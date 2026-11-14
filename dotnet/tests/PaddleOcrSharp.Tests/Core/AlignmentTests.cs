using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Tests.Core;

/// <summary>
/// Pooled activations start on a cache line, and where they sit changes nothing they compute.
/// </summary>
public class AlignmentTests
{
    private static unsafe bool IsAligned(Span<float> span) =>
        ((nint)Unsafe.AsPointer(ref MemoryMarshal.GetReference(span)) & 63) == 0;

    /// <summary>A rental starts on a 64-byte boundary at the sizes where it matters.</summary>
    /// <remarks>
    /// Only large rentals are asserted. Those live in the large-object heap, which does not move;
    /// a small array sits in gen0, where a collection triggered by anything else running at the
    /// same time may relocate it after its offset was taken. That costs a little speed and never
    /// correctness, which is why <see cref="TensorPool.AlignmentOffset(float[])"/> does not pin.
    /// </remarks>
    [Theory]
    [InlineData(30_000)]
    [InlineData(1152 * 1368)]
    [InlineData(4304 * 1368)]
    public void LargeRentalsStartOnACacheLine(int length)
    {
        for (int i = 0; i < 20; i++)
        {
            using PooledBuffer buffer = TensorPool.Rent(length);
            Assert.Equal(length, buffer.Span.Length);
            Assert.True(IsAligned(buffer.Span));

            using Tensor tensor = Tensor.Rent(length);
            Assert.Equal(length, tensor.Span.Length);
            Assert.True(IsAligned(tensor.Span));
            Assert.True(IsAligned(tensor.Rows(0, 1)));
        }
    }

    /// <summary>Rentals of every size, small ones included, are the length asked for.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(1000)]
    public void SmallRentalsAreTheLengthAskedFor(int length)
    {
        using PooledBuffer buffer = TensorPool.Rent(length, clear: true);
        Assert.Equal(length, buffer.Span.Length);
        Assert.Equal(length, buffer.Memory.Length);
        Assert.All(buffer.Span.ToArray(), value => Assert.Equal(0f, value));

        using Tensor tensor = Tensor.Rent(length);
        Assert.Equal(length, tensor.Span.Length);
    }

    /// <summary>The whole usable region is writable and does not run past the rental.</summary>
    [Fact]
    public void TheUsableRegionIsFullyAddressable()
    {
        const int Length = 12_345;
        using Tensor tensor = Tensor.Rent(Length);
        tensor.Span.Fill(1f);
        Assert.Equal(Length, tensor.Span.ToArray().Count(value => value == 1f));
        Assert.Equal(Length, tensor.Memory.Length);
    }

    /// <summary>A disposed tensor can still be asked for its (empty) span.</summary>
    [Fact]
    public void ADisposedTensorHasAnEmptySpan()
    {
        Tensor tensor = Tensor.Rent(1000);
        tensor.Dispose();
        Assert.True(tensor.Span.IsEmpty);
    }

    /// <summary>
    /// <see cref="Gemm.Linear"/> gives bit-identical results whether or not its activations are
    /// aligned: alignment moves data and nothing else.
    /// </summary>
    [Theory]
    [InlineData(8, 1152, 256)]
    [InlineData(37, 1024, 130)]
    [InlineData(355, 1024, 96)]
    public void LinearIsBitIdenticalForAlignedAndUnalignedInputs(int rows, int inner, int cols)
    {
        var random = new Random(rows);
        float[] values = new float[rows * inner];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)((random.NextDouble() * 2) - 1);
        }

        ushort[] raw = new ushort[cols * inner];
        for (int i = 0; i < raw.Length; i++)
        {
            raw[i] = (ushort)(0x3C00 + random.Next(0x7FF));
        }

        WeightMatrix weight = WeightMatrix.Create(
            MemoryMarshal.AsBytes<ushort>(raw).ToArray(), DType.BFloat16, cols, inner);

        // A plain array, which is not aligned, against a pooled tensor, which is.
        float[] plain = (float[])values.Clone();
        using Tensor pooled = Tensor.Rent(rows, inner);
        values.CopyTo(pooled.Span);

        float[] expected = new float[rows * cols];
        float[] actual = new float[rows * cols];
        Gemm.Linear(plain, rows, inner, weight, default, expected, cols);
        Gemm.Linear(pooled.Memory, rows, inner, weight, default, actual, cols);

        Assert.Equal(
            expected.Select(BitConverter.SingleToInt32Bits),
            actual.Select(BitConverter.SingleToInt32Bits));
    }
}
