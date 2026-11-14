using System.Collections.Concurrent;
using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats;

namespace PaddleOcrSharp.Models;

/// <summary>
/// Named access to a checkpoint's parameters.
/// </summary>
/// <remarks>
/// Matrices stay in their on-disk dtype and are read straight from the memory map. Vectors
/// (norm scales, biases) are small, are touched on every row of every layer, and are therefore
/// widened to float32 once and cached.
/// </remarks>
public sealed class WeightStore : IDisposable
{
    private readonly SafetensorsFile _file;
    private readonly ConcurrentDictionary<string, float[]> _vectorCache = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, WeightMatrix> _quantized = new(StringComparer.Ordinal);
    private readonly bool _ownsFile;
    private readonly WeightQuantization _quantization;

    private WeightStore(SafetensorsFile file, bool ownsFile, WeightQuantization quantization)
    {
        _file = file;
        _ownsFile = ownsFile;
        _quantization = quantization;
    }

    /// <summary>Opens the safetensors file at <paramref name="path"/>.</summary>
    /// <param name="path">The checkpoint.</param>
    /// <param name="quantization">How to hold the weights that <see cref="Matrix"/> is told may be quantised.</param>
    public static WeightStore Open(string path, WeightQuantization quantization = WeightQuantization.None) =>
        new(SafetensorsFile.Open(path), ownsFile: true, quantization);

    /// <summary>Wraps an already-open file without taking ownership of it.</summary>
    public static WeightStore Wrap(SafetensorsFile file) => new(file, ownsFile: false, WeightQuantization.None);

    /// <summary>Every parameter name in the checkpoint.</summary>
    public IReadOnlyCollection<string> Names => _file.Names;

    /// <summary>Whether <paramref name="name"/> exists.</summary>
    public bool Contains(string name) => _file.Contains(name);

    /// <summary>Gets a 2-D parameter as a weight matrix in <c>nn.Linear</c> layout.</summary>
    /// <remarks>
    /// A tensor stored as int8 is read with its scales, which sit beside it as <c>name.scale</c>,
    /// whatever the store was opened with. A bfloat16 one is quantised as it is read when the
    /// store asks for <see cref="WeightQuantization.Int8"/> and the caller says it may be.
    /// </remarks>
    /// <param name="name">The parameter name.</param>
    /// <param name="quantizable">Whether this weight may be quantised on load.</param>
    public WeightMatrix Matrix(string name, bool quantizable = false)
    {
        WeightTensor tensor = _file[name];

        if (tensor.Dtype == DType.Int8)
        {
            float[] scales = Vector(name + ".scale");
            int rows = tensor.Shape[0];
            int cols = tensor.Shape[1];
            int groups = scales.Length / rows;
            return WeightMatrix.CreateInt8(tensor.Bytes, scales, cols / groups, rows, cols, fromCheckpoint: true);
        }

        if (quantizable && _quantization == WeightQuantization.Int8 && tensor.Dtype == DType.BFloat16 && tensor.Shape.Length == 2)
        {
            return _quantized.GetOrAdd(name, static (_, t) => t.AsMatrix().QuantizeInt8(), tensor);
        }

        return tensor.AsMatrix();
    }

    /// <summary>
    /// Gets a parameter of any rank as a <c>[rows, cols]</c> matrix, folding trailing dimensions
    /// into the columns. Used for convolution kernels stored as <c>[out, in, kh, kw]</c>.
    /// </summary>
    public WeightMatrix MatrixFlattened(string name)
    {
        WeightTensor tensor = _file[name];
        int rows = tensor.Shape[0];
        int cols = tensor.ElementCount / rows;
        return tensor.AsMatrix(rows, cols);
    }

    /// <summary>Gets a parameter widened to float32 and cached.</summary>
    public float[] Vector(string name) => _vectorCache.GetOrAdd(name, static (key, self) =>
        self._file[key].ToFloats(), this);

    /// <summary>Gets a parameter widened to float32, or an empty array if it is absent.</summary>
    public float[] OptionalVector(string name) => _file.Contains(name) ? Vector(name) : [];

    /// <summary>Shape of <paramref name="name"/>.</summary>
    public int[] Shape(string name) => _file[name].Shape;

    /// <summary>Raw tensor access, for callers that need the dtype or the bytes.</summary>
    public WeightTensor Tensor(string name) => _file[name];

    /// <inheritdoc />
    public void Dispose()
    {
        _vectorCache.Clear();
        _quantized.Clear();
        if (_ownsFile)
        {
            _file.Dispose();
        }
    }
}
