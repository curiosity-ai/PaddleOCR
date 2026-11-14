using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace PaddleOcrSharp.Core;

/// <summary>
/// A dense, row-major float32 tensor. Activations flow through the model as these; weights stay
/// in their on-disk dtype and are exposed through <see cref="PaddleOcrSharp.Formats.WeightTensor"/>.
/// </summary>
/// <remarks>
/// Instances rented from the pool must be disposed. Instances created from an existing array
/// (<see cref="Wrap(float[], int[])"/>) are not pooled and disposing them is a no-op.
/// </remarks>
[DebuggerDisplay("Tensor {ShapeString,nq}")]
public sealed class Tensor : IDisposable
{
    private float[] _array;
    private int _offset;
    private readonly bool _pooled;

    private Tensor(float[] array, int[] shape, int length, bool pooled, int offset = 0)
    {
        _array = array;
        _offset = offset;
        _pooled = pooled;
        Shape = shape;
        Length = length;
    }

    /// <summary>Dimensions, outermost first.</summary>
    public int[] Shape { get; private set; }

    /// <summary>Total number of elements.</summary>
    public int Length { get; private set; }

    /// <summary>Number of dimensions.</summary>
    public int Rank => Shape.Length;

    /// <summary>The elements, in row-major order.</summary>
    public Span<float> Span
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _array.AsSpan(_offset, Length);
    }

    /// <summary>The elements as a <see cref="Memory{T}"/>.</summary>
    public Memory<float> Memory => _array.AsMemory(_offset, Length);

    /// <summary>Size of the last dimension, or <see cref="Length"/> for a rank-0 tensor.</summary>
    public int LastDim => Shape.Length == 0 ? Length : Shape[^1];

    /// <summary>Number of rows, i.e. <see cref="Length"/> divided by <see cref="LastDim"/>.</summary>
    public int RowCount => LastDim == 0 ? 0 : Length / LastDim;

    private string ShapeString => $"[{string.Join(", ", Shape)}]";

    /// <summary>Allocates a pooled tensor of the given <paramref name="shape"/>.</summary>
    public static Tensor Rent(params int[] shape)
    {
        int length = ElementCount(shape);
        float[] array = TensorPool.RentAlignableArray(length);
        int offset = length == 0 ? 0 : TensorPool.AlignmentOffset(array);
        return new Tensor(array, shape, length, pooled: true, offset);
    }

    /// <summary>Allocates a pooled tensor of the given <paramref name="shape"/>, zero-filled.</summary>
    public static Tensor RentZeroed(params int[] shape)
    {
        Tensor tensor = Rent(shape);
        tensor.Span.Clear();
        return tensor;
    }

    /// <summary>Allocates a non-pooled tensor of the given <paramref name="shape"/>, zero-filled.</summary>
    public static Tensor Zeros(params int[] shape)
    {
        int length = ElementCount(shape);
        return new Tensor(new float[length], shape, length, pooled: false);
    }

    /// <summary>Wraps an existing array without copying. The tensor does not own the array.</summary>
    public static Tensor Wrap(float[] array, params int[] shape)
    {
        int length = ElementCount(shape);
        if (array.Length < length)
        {
            throw new ArgumentException(
                $"Array of length {array.Length} cannot hold a tensor of {length} elements.",
                nameof(array));
        }

        return new Tensor(array, shape, length, pooled: false);
    }

    /// <summary>Copies <paramref name="values"/> into a new non-pooled tensor.</summary>
    public static Tensor From(ReadOnlySpan<float> values, params int[] shape)
    {
        Tensor tensor = Zeros(shape);
        values[..tensor.Length].CopyTo(tensor.Span);
        return tensor;
    }

    /// <summary>Number of elements implied by <paramref name="shape"/>.</summary>
    public static int ElementCount(ReadOnlySpan<int> shape)
    {
        long count = 1;
        foreach (int dim in shape)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(dim, nameof(shape));
            count *= dim;
        }

        if (count > int.MaxValue)
        {
            throw new ArgumentException($"Tensor with {count} elements exceeds the addressable range.");
        }

        return (int)count;
    }

    /// <summary>Row <paramref name="index"/> of a tensor viewed as <c>[rows, LastDim]</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> Row(int index) => _array.AsSpan(_offset + (index * LastDim), LastDim);

    /// <summary>A contiguous slice of <paramref name="count"/> rows starting at <paramref name="start"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> Rows(int start, int count) => _array.AsSpan(_offset + (start * LastDim), count * LastDim);

    /// <summary>
    /// Reinterprets the tensor with a new shape. The element count must be unchanged; one
    /// dimension may be <c>-1</c> and is inferred.
    /// </summary>
    public Tensor Reshape(params int[] shape)
    {
        int inferred = -1;
        long known = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] == -1)
            {
                if (inferred >= 0)
                {
                    throw new ArgumentException("At most one dimension may be inferred.", nameof(shape));
                }

                inferred = i;
            }
            else
            {
                known *= shape[i];
            }
        }

        int[] resolved = (int[])shape.Clone();
        if (inferred >= 0)
        {
            if (known == 0 || Length % known != 0)
            {
                throw new ArgumentException($"Cannot reshape {Length} elements to [{string.Join(",", shape)}].");
            }

            resolved[inferred] = (int)(Length / known);
            known *= resolved[inferred];
        }

        if (known != Length)
        {
            throw new ArgumentException(
                $"Cannot reshape {Length} elements to [{string.Join(",", shape)}] ({known} elements).");
        }

        Shape = resolved;
        return this;
    }

    /// <summary>Deep copy into a fresh non-pooled tensor.</summary>
    public Tensor Clone()
    {
        Tensor copy = Zeros((int[])Shape.Clone());
        Span.CopyTo(copy.Span);
        return copy;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_pooled && _array.Length != 0)
        {
            TensorPool.Return(_array);
            _array = [];
            _offset = 0;
            Length = 0;
        }
    }
}
