using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Text.Json;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Formats;

/// <summary>One tensor inside a safetensors file, still in its on-disk dtype.</summary>
/// <param name="Name">Fully-qualified parameter name, e.g. <c>model.layers.0.mlp.up_proj.weight</c>.</param>
/// <param name="Dtype">Storage dtype.</param>
/// <param name="Shape">Dimensions, outermost first.</param>
/// <param name="Bytes">The raw little-endian bytes, mapped straight from the file.</param>
public readonly record struct WeightTensor(string Name, DType Dtype, int[] Shape, ReadOnlyMemory<byte> Bytes)
{
    /// <summary>Total number of elements.</summary>
    public int ElementCount => Tensor.ElementCount(Shape);

    /// <summary>Views the tensor as a <c>[rows, cols]</c> matrix in <c>nn.Linear</c> layout.</summary>
    public WeightMatrix AsMatrix()
    {
        if (Shape.Length != 2)
        {
            throw new InvalidOperationException($"'{Name}' has rank {Shape.Length}; expected a matrix.");
        }

        return WeightMatrix.Create(Bytes, Dtype, Shape[0], Shape[1], fromCheckpoint: true);
    }

    /// <summary>Views the tensor as a <c>[rows, cols]</c> matrix with an explicit shape.</summary>
    public WeightMatrix AsMatrix(int rows, int cols) => WeightMatrix.Create(Bytes, Dtype, rows, cols, fromCheckpoint: true);

    /// <summary>Widens the tensor into a newly allocated float32 array.</summary>
    public float[] ToFloats()
    {
        float[] result = new float[ElementCount];
        FloatConversion.ToFloat(Bytes.Span, Dtype, result);
        return result;
    }

    /// <summary>Widens the tensor into <paramref name="destination"/>.</summary>
    public void CopyTo(Span<float> destination) => FloatConversion.ToFloat(Bytes.Span, Dtype, destination);
}

/// <summary>
/// Reader for the <see href="https://github.com/huggingface/safetensors">safetensors</see>
/// container: an 8-byte little-endian header length, a JSON header, then the tensor payloads.
/// </summary>
/// <remarks>
/// The file is memory-mapped, so a 1.8 GB checkpoint costs no managed allocation and the OS
/// pages weights in on demand.
/// </remarks>
public sealed class SafetensorsFile : IDisposable
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly Dictionary<string, WeightTensor> _tensors;
    private unsafe byte* _base;

    private SafetensorsFile(
        MemoryMappedFile file,
        MemoryMappedViewAccessor view,
        Dictionary<string, WeightTensor> tensors,
        Dictionary<string, string> metadata)
    {
        _file = file;
        _view = view;
        _tensors = tensors;
        Metadata = metadata;
    }

    /// <summary>Free-form <c>__metadata__</c> entries from the header.</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>Names of every tensor in the file.</summary>
    public IReadOnlyCollection<string> Names => _tensors.Keys;

    /// <summary>Number of tensors in the file.</summary>
    public int Count => _tensors.Count;

    /// <summary>Opens <paramref name="path"/> and parses its header.</summary>
    public static unsafe SafetensorsFile Open(string path)
    {
        long fileLength = new FileInfo(path).Length;
        if (fileLength < 8)
        {
            throw new InvalidDataException($"'{path}' is too short to be a safetensors file.");
        }

        MemoryMappedFile file = MemoryMappedFile.CreateFromFile(
            path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);

        MemoryMappedViewAccessor view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        byte* basePointer = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePointer);

        try
        {
            long headerLength = BinaryPrimitives.ReadInt64LittleEndian(new ReadOnlySpan<byte>(basePointer, 8));
            if (headerLength <= 0 || headerLength + 8 > fileLength)
            {
                throw new InvalidDataException($"'{path}' has an implausible header length of {headerLength}.");
            }

            var headerBytes = new ReadOnlySpan<byte>(basePointer + 8, (int)headerLength);
            long dataStart = 8 + headerLength;

            var tensors = new Dictionary<string, WeightTensor>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);

            using JsonDocument header = JsonDocument.Parse(headerBytes.ToArray());
            foreach (JsonProperty entry in header.RootElement.EnumerateObject())
            {
                if (entry.Name == "__metadata__")
                {
                    foreach (JsonProperty meta in entry.Value.EnumerateObject())
                    {
                        metadata[meta.Name] = meta.Value.ToString();
                    }

                    continue;
                }

                DType dtype = DTypeExtensions.FromSafetensors(entry.Value.GetProperty("dtype").GetString()!);

                JsonElement shapeElement = entry.Value.GetProperty("shape");
                int[] shape = new int[shapeElement.GetArrayLength()];
                for (int i = 0; i < shape.Length; i++)
                {
                    shape[i] = shapeElement[i].GetInt32();
                }

                JsonElement offsets = entry.Value.GetProperty("data_offsets");
                long begin = offsets[0].GetInt64();
                long end = offsets[1].GetInt64();

                if (begin < 0 || end < begin || dataStart + end > fileLength)
                {
                    throw new InvalidDataException($"Tensor '{entry.Name}' has out-of-range data offsets.");
                }

                long expected = (long)Tensor.ElementCount(shape) * dtype.ByteSize();
                if (end - begin != expected)
                {
                    throw new InvalidDataException(
                        $"Tensor '{entry.Name}' spans {end - begin} bytes but its shape and dtype need {expected}.");
                }

                var manager = new UnmanagedMemoryManager<byte>(basePointer + dataStart + begin, (int)(end - begin));
                tensors[entry.Name] = new WeightTensor(entry.Name, dtype, shape, manager.Memory);
            }

            var result = new SafetensorsFile(file, view, tensors, metadata) { _base = basePointer };
            return result;
        }
        catch
        {
            view.SafeMemoryMappedViewHandle.ReleasePointer();
            view.Dispose();
            file.Dispose();
            throw;
        }
    }

    /// <summary>Gets a tensor by name.</summary>
    /// <exception cref="KeyNotFoundException">The file has no such tensor.</exception>
    public WeightTensor this[string name] =>
        _tensors.TryGetValue(name, out WeightTensor tensor)
            ? tensor
            : throw new KeyNotFoundException($"Checkpoint has no tensor named '{name}'.");

    /// <summary>Tries to get a tensor by name.</summary>
    public bool TryGet(string name, out WeightTensor tensor) => _tensors.TryGetValue(name, out tensor);

    /// <summary>Whether the file contains <paramref name="name"/>.</summary>
    public bool Contains(string name) => _tensors.ContainsKey(name);

    /// <inheritdoc />
    public unsafe void Dispose()
    {
        _tensors.Clear();
        if (_base is not null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _base = null;
        }

        _view.Dispose();
        _file.Dispose();
    }
}
