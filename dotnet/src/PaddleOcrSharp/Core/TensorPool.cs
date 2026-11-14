using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace PaddleOcrSharp.Core;

/// <summary>
/// A scoped rental of a pooled <see cref="float"/> buffer.
/// </summary>
/// <remarks>
/// The rented array is normally larger than <see cref="Length"/>; always go through
/// <see cref="Span"/> so kernels never see the slack at the end. <see cref="Span"/> also starts on
/// a 64-byte boundary — see <see cref="TensorPool.AlignmentOffset(float[])"/>.
/// </remarks>
public readonly struct PooledBuffer : IDisposable
{
    private readonly float[] _array;
    private readonly int _offset;

    internal PooledBuffer(float[] array, int length, int offset = 0)
    {
        _array = array;
        Length = length;
        _offset = offset;
    }

    /// <summary>Number of usable elements.</summary>
    public int Length { get; }

    /// <summary>The usable portion of the rented array.</summary>
    public Span<float> Span => _array.AsSpan(_offset, Length);

    /// <summary>The usable portion of the rented array as a <see cref="Memory{T}"/>.</summary>
    public Memory<float> Memory => _array.AsMemory(_offset, Length);

    /// <summary>Returns the buffer to the pool.</summary>
    public void Dispose()
    {
        if (_array is not null)
        {
            TensorPool.Return(_array);
        }
    }
}

/// <summary>
/// Central <see cref="ArrayPool{T}"/> wrapper for the transient activation buffers the model
/// allocates per layer. Keeping these pooled is what lets a decode step run allocation-free.
/// </summary>
public static class TensorPool
{
    // `ArrayPool<T>.Shared`, measured rather than assumed. The comment this replaces said the
    // shared pool caps its buckets at 1 MiB and that activations here are far larger, so the pool
    // had to be our own. The first half was true of .NET Framework and has not been true for
    // years: the shared pool round-trips a 1 GiB float array with zero allocation. The second half
    // of the reasoning then made things worse, because `ArrayPool.Create` keeps only
    // `maxArraysPerBucket` buffers of a size and drops the rest — with 16, thirty-six live buffers
    // of one size lose twenty on every round, which is exactly the shape a graph run and a
    // key/value cache both have. Measured over a rent-and-return round of N live 4 MiB buffers:
    //
    //   N   shared   created(16)
    //   16   0 MiB     0 MiB
    //   36   0 MiB    16 MiB
    //   64   0 MiB   128 MiB
    //
    // The shared pool also trims under GC pressure, where a created one holds its buffers for the
    // life of the process. This type stays as the one place the pooling policy is stated.

    /// <summary>Rents a buffer of at least <paramref name="length"/> floats.</summary>
    /// <param name="length">Number of usable elements.</param>
    /// <param name="clear">When <see langword="true"/>, zeroes the usable portion before returning.</param>
    public static PooledBuffer Rent(int length, bool clear = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);

        float[] array = length == 0 ? [] : ArrayPool<float>.Shared.Rent(length + AlignmentSlack);
        int offset = length == 0 ? 0 : AlignmentOffset(array);
        if (clear && length > 0)
        {
            array.AsSpan(offset, length).Clear();
        }

        return new PooledBuffer(array, length, offset);
    }

    /// <summary>Elements of slack a rental carries so its usable region can start on a cache line.</summary>
    internal const int AlignmentSlack = 15;

    /// <summary>
    /// How many elements into <paramref name="array"/> the next 64-byte boundary lies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A .NET array's elements start sixteen bytes past the object header, so no array is 64-byte
    /// aligned, and a 512-bit load from it straddles two cache lines on every access. In the GEMM's
    /// four-by-four tile that doubles the load-port work — sixteen line-split loads against eight
    /// whole ones per sixteen multiply-adds — and measured as 331 GFLOP/s against 450 for the same
    /// kernel on an aligned weight panel (four threads, 1368 rows by 1152 by 4304). Alignment
    /// moves where the data sits and nothing else, so no output changes.
    /// </para>
    /// <para>
    /// The address is read without pinning, so a collection that relocates a small array between
    /// the rental and its use could leave the region misaligned. That costs speed and never
    /// correctness, and arrays of the sizes where it matters live in the large-object heap, which
    /// does not move.
    /// </para>
    /// </remarks>
    /// <param name="array">A rented array of at least <see cref="AlignmentSlack"/> spare elements.</param>
    internal static unsafe int AlignmentOffset(float[] array)
    {
        nint address = (nint)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(array));
        return (int)(((64 - (address & 63)) & 63) >> 2);
    }

    /// <summary>Rents a raw float array of at least <paramref name="length"/> elements.</summary>
    internal static float[] RentArray(int length) => length == 0 ? [] : ArrayPool<float>.Shared.Rent(length);

    /// <summary>
    /// Rents a raw float array with <see cref="AlignmentSlack"/> spare elements, for a caller that
    /// slices its own cache-line-aligned region out of it with <see cref="AlignmentOffset(float[])"/>.
    /// </summary>
    internal static float[] RentAlignableArray(int length) =>
        length == 0 ? [] : ArrayPool<float>.Shared.Rent(length + AlignmentSlack);

    /// <summary>Rents a raw byte buffer of at least <paramref name="length"/> bytes.</summary>
    /// <remarks>
    /// For a buffer that has to be read as bytes as well as written as floats — a
    /// a weight matrix over pooled storage, which is how the
    /// convolution hands its im2col columns to <see cref="Gemm.Linear"/> without copying them.
    /// </remarks>
    /// <param name="length">Number of usable bytes.</param>
    public static byte[] RentBytes(int length) => length == 0 ? [] : ArrayPool<byte>.Shared.Rent(length);

    /// <summary>Returns a byte buffer previously obtained from <see cref="RentBytes"/>.</summary>
    /// <param name="array">The buffer to return.</param>
    public static void ReturnBytes(byte[] array)
    {
        if (array.Length != 0)
        {
            ArrayPool<byte>.Shared.Return(array);
        }
    }

    /// <summary>Rents a buffer of at least <paramref name="length"/> ints.</summary>
    public static int[] RentInts(int length) => length == 0 ? [] : ArrayPool<int>.Shared.Rent(length);

    /// <summary>Returns an int buffer previously obtained from <see cref="RentInts"/>.</summary>
    public static void ReturnInts(int[] array)
    {
        if (array.Length != 0)
        {
            ArrayPool<int>.Shared.Return(array);
        }
    }

    /// <summary>Rents a raw long array of at least <paramref name="length"/> elements.</summary>
    internal static long[] RentLongs(int length) => length == 0 ? [] : ArrayPool<long>.Shared.Rent(length);

    /// <summary>Returns a long array previously obtained from <see cref="RentLongs"/>.</summary>
    internal static void ReturnLongs(long[] array)
    {
        if (array.Length != 0)
        {
            ArrayPool<long>.Shared.Return(array);
        }
    }

    /// <summary>Returns an array previously obtained from this pool.</summary>
    internal static void Return(float[] array)
    {
        if (array.Length != 0)
        {
            ArrayPool<float>.Shared.Return(array);
        }
    }
}
