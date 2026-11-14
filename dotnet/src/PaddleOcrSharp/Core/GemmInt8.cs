using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace PaddleOcrSharp.Core;

/// <summary>
/// <see cref="Gemm.Linear"/> with the activations quantised to int8 as well as the weights, run on the
/// integer dot-product instruction (<c>vpdpbusd</c>, AVX-VNNI) instead of float32 multiply-adds.
/// </summary>
/// <remarks>
/// <para>
/// A 256-bit <c>vpdpbusd</c> does thirty-two multiply-adds, and a four-by-four tile of them ran at
/// 207-211 GMAC/s a thread against 67 for the float32 dot tile it replaces: 3.1 times, in isolation.
/// The weights are used where they lie — a row of int8 values is already the contiguous reduction
/// vector the tile wants — so nothing is widened, packed or transposed, and the working set of a
/// tile is two 4.6 KB runs that sit in L1.
/// </para>
/// <para>
/// The activations are quantised once per call, one scale per row (per token): <c>q = round(x / s)</c>
/// with <c>s = max|x| / 127</c>, then shifted by 128 to the unsigned bytes the instruction wants.
/// The shift is undone with the weights' row sums, which are fixed and are computed once per
/// matrix: <c>Σ q·w = Σ (q + 128)·w − 128·Σ w</c>. The sums are exact in int32 for any reduction
/// length this model has, and each output is then <c>(sum) · s_x · s_w + bias</c>.
/// </para>
/// <para>
/// This needs the weights' scale to be one per row, so that one integer sum per output carries all the
/// way to the end; a grouped scale would need a conversion and a multiply-add every group. It is
/// <b>not</b> bit-identical to anything else here and it is not meant to be: it is the integer
/// arithmetic the <c>PADDLEOCR_SHARP_FAKE_A_BITS</c> experiment emulated, whose effect on recognition is
/// recorded in <c>CLAUDE.md</c>. It runs only for a matrix loaded with
/// <see cref="Models.WeightQuantization.Int8Compute"/>, on a machine with AVX-VNNI.
/// </para>
/// </remarks>
internal static unsafe class GemmInt8
{
    /// <summary>Fewer rows than this and the product is bound by the weights, not the arithmetic.</summary>
    internal const int MinRows = 32;

    /// <summary>Whether the instruction exists here.</summary>
    internal static bool Available => AvxVnni.IsSupported;

    /// <summary>
    /// Whether the tiles run on 512-bit vectors (<c>AvxVnni.V512</c>, .NET 11). The sums are integers, so
    /// the choice changes speed and nothing else; <c>PADDLEOCR_SHARP_INT8_WIDE=0</c> keeps the 256-bit
    /// tile on a machine that has both.
    /// </summary>
    internal static readonly bool Wide = Simd.Use512
        && AvxVnni.V512.IsSupported
        && Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_INT8_WIDE") != "0";

    /// <summary>Whether a product of <paramref name="rows"/> activation rows against <paramref name="weight"/> takes this path.</summary>
    internal static bool Applies(int rows, WeightMatrix weight) =>
        Available
        && rows >= MinRows
        && weight.ComputeInt8
        && weight.Dtype == DType.Int8
        && weight.GroupSize == weight.Cols
        && (weight.Cols & 15) == 0;

    /// <summary>Whether every projection in <paramref name="parts"/> takes this path.</summary>
    internal static bool AppliesToAll(int rows, Gemm.LinearPart[] parts)
    {
        foreach (Gemm.LinearPart part in parts)
        {
            if (!Applies(rows, part.Weight))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Several projections of one activation matrix; the activations are quantised once.</summary>
    internal static void LinearMany(ReadOnlyMemory<float> x, int rows, int inner, Gemm.LinearPart[] parts)
    {
        int paddedRows = (rows + 3) & ~3;
        using PooledBuffer quantised = TensorPool.Rent(((paddedRows * inner) + 3) / 4);
        using PooledBuffer rowScales = TensorPool.Rent(paddedRows);
        Quantise(x, rows, paddedRows, inner, quantised.Memory, rowScales.Memory);

        Gemm.PanelPlan plan = Gemm.PanelPlan.Create(parts, inner);
        RunPanels(plan, quantised.Memory, rowScales.Memory, rows, inner);
    }

    // Lambdas live in their own methods, as everywhere in the GEMM.
    private static void Quantise(
        ReadOnlyMemory<float> x, int rows, int paddedRows, int inner, Memory<float> quantised, Memory<float> scales) =>
        Parallelism.For((paddedRows + 15) / 16, block =>
        {
            Span<byte> destination = MemoryMarshal.AsBytes(quantised.Span);
            for (int r = block * 16; r < Math.Min(paddedRows, (block + 1) * 16); r++)
            {
                if (r < rows)
                {
                    scales.Span[r] = QuantiseRow(x.Span.Slice(r * inner, inner), destination.Slice(r * inner, inner));
                }
                else
                {
                    destination.Slice(r * inner, inner).Fill(128);
                    scales.Span[r] = 0f;
                }
            }
        });

    /// <summary>Quantises one row to unsigned bytes (value + 128) and returns its scale.</summary>
    private static float QuantiseRow(ReadOnlySpan<float> row, Span<byte> destination)
    {
        float peak = 0f;
        foreach (float value in row)
        {
            peak = Math.Max(peak, Math.Abs(value));
        }

        float scale = peak == 0f ? 1f : peak / 127f;
        for (int i = 0; i < row.Length; i++)
        {
            destination[i] = (byte)(int)(Math.Clamp(MathF.Round(row[i] / scale), -127f, 127f) + 128f);
        }

        return scale;
    }

    private static void RunPanels(Gemm.PanelPlan plan, ReadOnlyMemory<float> quantised, ReadOnlyMemory<float> scales, int rows, int inner) =>
        Parallelism.For(plan.Total, index =>
        {
            (int part, int start, int count) = plan.Locate(index);
            Gemm.LinearPart p = plan.Parts[part];
            RunPanel(
                MemoryMarshal.AsBytes(quantised.Span), scales.Span, rows, inner, p.Weight, p.Weight.GetRowSums(),
                p.Bias.Span, p.Output.Span, p.Cols, start, count);
        });

    private static void RunPanel(
        ReadOnlySpan<byte> x,
        ReadOnlySpan<float> rowScales,
        int rows,
        int inner,
        WeightMatrix weight,
        ReadOnlySpan<int> weightSums,
        ReadOnlySpan<float> bias,
        Span<float> y,
        int cols,
        int colStart,
        int colCount)
    {
        ReadOnlySpan<byte> values = weight.Int8Values;
        ReadOnlySpan<float> weightScales = weight.Int8Scales;
        int* sums = stackalloc int[16];

        fixed (byte* px = x)
        fixed (byte* pw = values)
        fixed (int* ws = weightSums)
        fixed (float* sx = rowScales, sw = weightScales, py = y, pb = bias)
        {
            bool hasBias = !bias.IsEmpty;
            bool wide = Wide;
            int paddedRows = (rows + 3) & ~3;

            for (int m = 0; m < paddedRows; m += 4)
            {
                int live = Math.Min(4, rows - m);
                byte* xm = px + ((long)m * inner);

                int c = 0;
                for (; c + 4 <= colCount; c += 4)
                {
                    int n = colStart + c;
                    if (wide)
                    {
                        Tile4x4Wide(xm, inner, (sbyte*)(pw + ((long)n * inner)), inner, sums);
                    }
                    else
                    {
                        Tile4x4(xm, inner, (sbyte*)(pw + ((long)n * inner)), inner, sums);
                    }

                   
                    Store(sums, live, 4, m, n, sx, sw, ws, py, cols, hasBias ? pb : null);
                }

                for (; c < colCount; c++)
                {
                    int n = colStart + c;
                    if (wide)
                    {
                        Tile4x1Wide(xm, inner, (sbyte*)(pw + ((long)n * inner)), sums);
                    }
                    else
                    {
                        Tile4x1(xm, inner, (sbyte*)(pw + ((long)n * inner)), sums);
                    }

                   
                    Store(sums, live, 1, m, n, sx, sw, ws, py, cols, hasBias ? pb : null);
                }
            }
        }
    }

    /// <summary>Turns a tile's integer sums (<c>sums[i * columns + j]</c>) into outputs.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(
        int* sums, int live, int columns, int m, int n, float* rowScales, float* weightScales, int* weightSums,
        float* y, int cols, float* bias)
    {
        for (int i = 0; i < live; i++)
        {
            float sx = rowScales[m + i];
            float* destination = y + ((long)(m + i) * cols) + n;
            for (int j = 0; j < columns; j++)
            {
                float value = (sums[(i * columns) + j] - (128 * weightSums[n + j])) * (sx * weightScales[n + j]);
                destination[j] = bias is null ? value : value + bias[n + j];
            }
        }
    }

    /// <summary>
    /// Four activation rows against four weight rows on 256-bit vectors: sixteen int32 accumulators of
    /// eight lanes, each step thirty-two bytes of each row. The reduction length is a multiple of
    /// sixteen, not necessarily of thirty-two; a trailing sixteen are zero-extended to a full vector
    /// so they accumulate in place. The sums are exact integers, so every kernel here gives the same
    /// ones whatever order it adds them in.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Tile4x4(byte* x, int k, sbyte* w, int stride, int* sums)
    {
        byte* x0 = x, x1 = x + k, x2 = x + (2L * k), x3 = x + (3L * k);
        sbyte* w0 = w, w1 = w + stride, w2 = w + (2L * stride), w3 = w + (3L * stride);

        Vector256<int> a00 = default, a01 = default, a02 = default, a03 = default;
        Vector256<int> a10 = default, a11 = default, a12 = default, a13 = default;
        Vector256<int> a20 = default, a21 = default, a22 = default, a23 = default;
        Vector256<int> a30 = default, a31 = default, a32 = default, a33 = default;

        int i = 0;
        for (; i + 32 <= k; i += 32)
        {
            Vector256<sbyte> b0 = Vector256.Load(w0 + i);
            Vector256<sbyte> b1 = Vector256.Load(w1 + i);
            Vector256<sbyte> b2 = Vector256.Load(w2 + i);
            Vector256<sbyte> b3 = Vector256.Load(w3 + i);

            Vector256<byte> v = Vector256.Load(x0 + i);
            a00 = AvxVnni.MultiplyWideningAndAdd(a00, v, b0);
            a01 = AvxVnni.MultiplyWideningAndAdd(a01, v, b1);
            a02 = AvxVnni.MultiplyWideningAndAdd(a02, v, b2);
            a03 = AvxVnni.MultiplyWideningAndAdd(a03, v, b3);

            v = Vector256.Load(x1 + i);
            a10 = AvxVnni.MultiplyWideningAndAdd(a10, v, b0);
            a11 = AvxVnni.MultiplyWideningAndAdd(a11, v, b1);
            a12 = AvxVnni.MultiplyWideningAndAdd(a12, v, b2);
            a13 = AvxVnni.MultiplyWideningAndAdd(a13, v, b3);

            v = Vector256.Load(x2 + i);
            a20 = AvxVnni.MultiplyWideningAndAdd(a20, v, b0);
            a21 = AvxVnni.MultiplyWideningAndAdd(a21, v, b1);
            a22 = AvxVnni.MultiplyWideningAndAdd(a22, v, b2);
            a23 = AvxVnni.MultiplyWideningAndAdd(a23, v, b3);

            v = Vector256.Load(x3 + i);
            a30 = AvxVnni.MultiplyWideningAndAdd(a30, v, b0);
            a31 = AvxVnni.MultiplyWideningAndAdd(a31, v, b1);
            a32 = AvxVnni.MultiplyWideningAndAdd(a32, v, b2);
            a33 = AvxVnni.MultiplyWideningAndAdd(a33, v, b3);
        }

        if (i < k)
        {
            // One sixteen-byte step remains, zero-extended: the upper half contributes nothing.
            Vector256<sbyte> b0 = Vector256.Create(Vector128.Load(w0 + i), Vector128<sbyte>.Zero);
            Vector256<sbyte> b1 = Vector256.Create(Vector128.Load(w1 + i), Vector128<sbyte>.Zero);
            Vector256<sbyte> b2 = Vector256.Create(Vector128.Load(w2 + i), Vector128<sbyte>.Zero);
            Vector256<sbyte> b3 = Vector256.Create(Vector128.Load(w3 + i), Vector128<sbyte>.Zero);

            Vector256<byte> v = Vector256.Create(Vector128.Load(x0 + i), Vector128<byte>.Zero);
            a00 = AvxVnni.MultiplyWideningAndAdd(a00, v, b0);
            a01 = AvxVnni.MultiplyWideningAndAdd(a01, v, b1);
            a02 = AvxVnni.MultiplyWideningAndAdd(a02, v, b2);
            a03 = AvxVnni.MultiplyWideningAndAdd(a03, v, b3);

            v = Vector256.Create(Vector128.Load(x1 + i), Vector128<byte>.Zero);
            a10 = AvxVnni.MultiplyWideningAndAdd(a10, v, b0);
            a11 = AvxVnni.MultiplyWideningAndAdd(a11, v, b1);
            a12 = AvxVnni.MultiplyWideningAndAdd(a12, v, b2);
            a13 = AvxVnni.MultiplyWideningAndAdd(a13, v, b3);

            v = Vector256.Create(Vector128.Load(x2 + i), Vector128<byte>.Zero);
            a20 = AvxVnni.MultiplyWideningAndAdd(a20, v, b0);
            a21 = AvxVnni.MultiplyWideningAndAdd(a21, v, b1);
            a22 = AvxVnni.MultiplyWideningAndAdd(a22, v, b2);
            a23 = AvxVnni.MultiplyWideningAndAdd(a23, v, b3);

            v = Vector256.Create(Vector128.Load(x3 + i), Vector128<byte>.Zero);
            a30 = AvxVnni.MultiplyWideningAndAdd(a30, v, b0);
            a31 = AvxVnni.MultiplyWideningAndAdd(a31, v, b1);
            a32 = AvxVnni.MultiplyWideningAndAdd(a32, v, b2);
            a33 = AvxVnni.MultiplyWideningAndAdd(a33, v, b3);
        }

        Vector128.Store(ReduceFour(a00, a01, a02, a03), sums + 0);
        Vector128.Store(ReduceFour(a10, a11, a12, a13), sums + 4);
        Vector128.Store(ReduceFour(a20, a21, a22, a23), sums + 8);
        Vector128.Store(ReduceFour(a30, a31, a32, a33), sums + 12);
    }

    /// <summary>
    /// <see cref="Tile4x4"/> on 512-bit vectors (<c>AvxVnni.V512</c>, new in .NET 11): sixty-four bytes of
    /// each row a step, so half as many steps for the same arithmetic. What does not fill a 64-byte
    /// step — sixteen, thirty-two or forty-eight bytes — runs on 256-bit vectors after the accumulators
    /// are folded to that width, which is also where the reduction starts. Integer sums: bit-identical
    /// to the 256-bit tile.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Tile4x4Wide(byte* x, int k, sbyte* w, int stride, int* sums)
    {
        byte* x0 = x, x1 = x + k, x2 = x + (2L * k), x3 = x + (3L * k);
        sbyte* w0 = w, w1 = w + stride, w2 = w + (2L * stride), w3 = w + (3L * stride);

        Vector512<int> a00 = default, a01 = default, a02 = default, a03 = default;
        Vector512<int> a10 = default, a11 = default, a12 = default, a13 = default;
        Vector512<int> a20 = default, a21 = default, a22 = default, a23 = default;
        Vector512<int> a30 = default, a31 = default, a32 = default, a33 = default;

        int i = 0;
        for (; i + 64 <= k; i += 64)
        {
            Vector512<sbyte> b0 = Vector512.Load(w0 + i);
            Vector512<sbyte> b1 = Vector512.Load(w1 + i);
            Vector512<sbyte> b2 = Vector512.Load(w2 + i);
            Vector512<sbyte> b3 = Vector512.Load(w3 + i);

            Vector512<byte> v = Vector512.Load(x0 + i);
            a00 = AvxVnni.V512.MultiplyWideningAndAdd(a00, v, b0);
            a01 = AvxVnni.V512.MultiplyWideningAndAdd(a01, v, b1);
            a02 = AvxVnni.V512.MultiplyWideningAndAdd(a02, v, b2);
            a03 = AvxVnni.V512.MultiplyWideningAndAdd(a03, v, b3);

            v = Vector512.Load(x1 + i);
            a10 = AvxVnni.V512.MultiplyWideningAndAdd(a10, v, b0);
            a11 = AvxVnni.V512.MultiplyWideningAndAdd(a11, v, b1);
            a12 = AvxVnni.V512.MultiplyWideningAndAdd(a12, v, b2);
            a13 = AvxVnni.V512.MultiplyWideningAndAdd(a13, v, b3);

            v = Vector512.Load(x2 + i);
            a20 = AvxVnni.V512.MultiplyWideningAndAdd(a20, v, b0);
            a21 = AvxVnni.V512.MultiplyWideningAndAdd(a21, v, b1);
            a22 = AvxVnni.V512.MultiplyWideningAndAdd(a22, v, b2);
            a23 = AvxVnni.V512.MultiplyWideningAndAdd(a23, v, b3);

            v = Vector512.Load(x3 + i);
            a30 = AvxVnni.V512.MultiplyWideningAndAdd(a30, v, b0);
            a31 = AvxVnni.V512.MultiplyWideningAndAdd(a31, v, b1);
            a32 = AvxVnni.V512.MultiplyWideningAndAdd(a32, v, b2);
            a33 = AvxVnni.V512.MultiplyWideningAndAdd(a33, v, b3);
        }

        // Fold to 256 bits; the remainder, if any, accumulates there.
        Vector256<int> f00 = a00.GetLower() + a00.GetUpper(), f01 = a01.GetLower() + a01.GetUpper(), f02 = a02.GetLower() + a02.GetUpper(), f03 = a03.GetLower() + a03.GetUpper();
        Vector256<int> f10 = a10.GetLower() + a10.GetUpper(), f11 = a11.GetLower() + a11.GetUpper(), f12 = a12.GetLower() + a12.GetUpper(), f13 = a13.GetLower() + a13.GetUpper();
        Vector256<int> f20 = a20.GetLower() + a20.GetUpper(), f21 = a21.GetLower() + a21.GetUpper(), f22 = a22.GetLower() + a22.GetUpper(), f23 = a23.GetLower() + a23.GetUpper();
        Vector256<int> f30 = a30.GetLower() + a30.GetUpper(), f31 = a31.GetLower() + a31.GetUpper(), f32 = a32.GetLower() + a32.GetUpper(), f33 = a33.GetLower() + a33.GetUpper();

        if (i + 32 <= k)
        {
            Vector256<sbyte> b0 = Vector256.Load(w0 + i);
            Vector256<sbyte> b1 = Vector256.Load(w1 + i);
            Vector256<sbyte> b2 = Vector256.Load(w2 + i);
            Vector256<sbyte> b3 = Vector256.Load(w3 + i);

            Vector256<byte> v = Vector256.Load(x0 + i);
            f00 = AvxVnni.MultiplyWideningAndAdd(f00, v, b0);
            f01 = AvxVnni.MultiplyWideningAndAdd(f01, v, b1);
            f02 = AvxVnni.MultiplyWideningAndAdd(f02, v, b2);
            f03 = AvxVnni.MultiplyWideningAndAdd(f03, v, b3);

            v = Vector256.Load(x1 + i);
            f10 = AvxVnni.MultiplyWideningAndAdd(f10, v, b0);
            f11 = AvxVnni.MultiplyWideningAndAdd(f11, v, b1);
            f12 = AvxVnni.MultiplyWideningAndAdd(f12, v, b2);
            f13 = AvxVnni.MultiplyWideningAndAdd(f13, v, b3);

            v = Vector256.Load(x2 + i);
            f20 = AvxVnni.MultiplyWideningAndAdd(f20, v, b0);
            f21 = AvxVnni.MultiplyWideningAndAdd(f21, v, b1);
            f22 = AvxVnni.MultiplyWideningAndAdd(f22, v, b2);
            f23 = AvxVnni.MultiplyWideningAndAdd(f23, v, b3);

            v = Vector256.Load(x3 + i);
            f30 = AvxVnni.MultiplyWideningAndAdd(f30, v, b0);
            f31 = AvxVnni.MultiplyWideningAndAdd(f31, v, b1);
            f32 = AvxVnni.MultiplyWideningAndAdd(f32, v, b2);
            f33 = AvxVnni.MultiplyWideningAndAdd(f33, v, b3);
            i += 32;
        }

        if (i < k)
        {
            // One sixteen-byte step remains, zero-extended.
            Vector256<sbyte> c0 = Vector256.Create(Vector128.Load(w0 + i), Vector128<sbyte>.Zero);
            Vector256<sbyte> c1 = Vector256.Create(Vector128.Load(w1 + i), Vector128<sbyte>.Zero);
            Vector256<sbyte> c2 = Vector256.Create(Vector128.Load(w2 + i), Vector128<sbyte>.Zero);
            Vector256<sbyte> c3 = Vector256.Create(Vector128.Load(w3 + i), Vector128<sbyte>.Zero);

            Vector256<byte> u = Vector256.Create(Vector128.Load(x0 + i), Vector128<byte>.Zero);
            f00 = AvxVnni.MultiplyWideningAndAdd(f00, u, c0);
            f01 = AvxVnni.MultiplyWideningAndAdd(f01, u, c1);
            f02 = AvxVnni.MultiplyWideningAndAdd(f02, u, c2);
            f03 = AvxVnni.MultiplyWideningAndAdd(f03, u, c3);

            u = Vector256.Create(Vector128.Load(x1 + i), Vector128<byte>.Zero);
            f10 = AvxVnni.MultiplyWideningAndAdd(f10, u, c0);
            f11 = AvxVnni.MultiplyWideningAndAdd(f11, u, c1);
            f12 = AvxVnni.MultiplyWideningAndAdd(f12, u, c2);
            f13 = AvxVnni.MultiplyWideningAndAdd(f13, u, c3);

            u = Vector256.Create(Vector128.Load(x2 + i), Vector128<byte>.Zero);
            f20 = AvxVnni.MultiplyWideningAndAdd(f20, u, c0);
            f21 = AvxVnni.MultiplyWideningAndAdd(f21, u, c1);
            f22 = AvxVnni.MultiplyWideningAndAdd(f22, u, c2);
            f23 = AvxVnni.MultiplyWideningAndAdd(f23, u, c3);

            u = Vector256.Create(Vector128.Load(x3 + i), Vector128<byte>.Zero);
            f30 = AvxVnni.MultiplyWideningAndAdd(f30, u, c0);
            f31 = AvxVnni.MultiplyWideningAndAdd(f31, u, c1);
            f32 = AvxVnni.MultiplyWideningAndAdd(f32, u, c2);
            f33 = AvxVnni.MultiplyWideningAndAdd(f33, u, c3);
        }

        Vector128.Store(ReduceFour(f00, f01, f02, f03), sums + 0);
        Vector128.Store(ReduceFour(f10, f11, f12, f13), sums + 4);
        Vector128.Store(ReduceFour(f20, f21, f22, f23), sums + 8);
        Vector128.Store(ReduceFour(f30, f31, f32, f33), sums + 12);
    }

    /// <summary>Four activation rows against one weight row, for the last few columns of a ragged panel.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Tile4x1(byte* x, int k, sbyte* w, int* sums)
    {
        byte* x0 = x, x1 = x + k, x2 = x + (2L * k), x3 = x + (3L * k);
        Vector256<int> a0 = default, a1 = default, a2 = default, a3 = default;

        int i = 0;
        for (; i + 32 <= k; i += 32)
        {
            Vector256<sbyte> b = Vector256.Load(w + i);
            a0 = AvxVnni.MultiplyWideningAndAdd(a0, Vector256.Load(x0 + i), b);
            a1 = AvxVnni.MultiplyWideningAndAdd(a1, Vector256.Load(x1 + i), b);
            a2 = AvxVnni.MultiplyWideningAndAdd(a2, Vector256.Load(x2 + i), b);
            a3 = AvxVnni.MultiplyWideningAndAdd(a3, Vector256.Load(x3 + i), b);
        }

        if (i < k)
        {
            Vector256<sbyte> c = Vector256.Create(Vector128.Load(w + i), Vector128<sbyte>.Zero);
            a0 = AvxVnni.MultiplyWideningAndAdd(a0, Vector256.Create(Vector128.Load(x0 + i), Vector128<byte>.Zero), c);
            a1 = AvxVnni.MultiplyWideningAndAdd(a1, Vector256.Create(Vector128.Load(x1 + i), Vector128<byte>.Zero), c);
            a2 = AvxVnni.MultiplyWideningAndAdd(a2, Vector256.Create(Vector128.Load(x2 + i), Vector128<byte>.Zero), c);
            a3 = AvxVnni.MultiplyWideningAndAdd(a3, Vector256.Create(Vector128.Load(x3 + i), Vector128<byte>.Zero), c);
        }

        Vector128.Store(ReduceFour(a0, a1, a2, a3), sums);
    }

    /// <summary><see cref="Tile4x1"/> on 512-bit vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Tile4x1Wide(byte* x, int k, sbyte* w, int* sums)
    {
        byte* x0 = x, x1 = x + k, x2 = x + (2L * k), x3 = x + (3L * k);
        Vector512<int> a0 = default, a1 = default, a2 = default, a3 = default;

        int i = 0;
        for (; i + 64 <= k; i += 64)
        {
            Vector512<sbyte> b = Vector512.Load(w + i);
            a0 = AvxVnni.V512.MultiplyWideningAndAdd(a0, Vector512.Load(x0 + i), b);
            a1 = AvxVnni.V512.MultiplyWideningAndAdd(a1, Vector512.Load(x1 + i), b);
            a2 = AvxVnni.V512.MultiplyWideningAndAdd(a2, Vector512.Load(x2 + i), b);
            a3 = AvxVnni.V512.MultiplyWideningAndAdd(a3, Vector512.Load(x3 + i), b);
        }

        Vector256<int> f0 = a0.GetLower() + a0.GetUpper(), f1 = a1.GetLower() + a1.GetUpper();
        Vector256<int> f2 = a2.GetLower() + a2.GetUpper(), f3 = a3.GetLower() + a3.GetUpper();

        if (i + 32 <= k)
        {
            Vector256<sbyte> b = Vector256.Load(w + i);
            f0 = AvxVnni.MultiplyWideningAndAdd(f0, Vector256.Load(x0 + i), b);
            f1 = AvxVnni.MultiplyWideningAndAdd(f1, Vector256.Load(x1 + i), b);
            f2 = AvxVnni.MultiplyWideningAndAdd(f2, Vector256.Load(x2 + i), b);
            f3 = AvxVnni.MultiplyWideningAndAdd(f3, Vector256.Load(x3 + i), b);
            i += 32;
        }

        if (i < k)
        {
            Vector256<sbyte> c = Vector256.Create(Vector128.Load(w + i), Vector128<sbyte>.Zero);
            f0 = AvxVnni.MultiplyWideningAndAdd(f0, Vector256.Create(Vector128.Load(x0 + i), Vector128<byte>.Zero), c);
            f1 = AvxVnni.MultiplyWideningAndAdd(f1, Vector256.Create(Vector128.Load(x1 + i), Vector128<byte>.Zero), c);
            f2 = AvxVnni.MultiplyWideningAndAdd(f2, Vector256.Create(Vector128.Load(x2 + i), Vector128<byte>.Zero), c);
            f3 = AvxVnni.MultiplyWideningAndAdd(f3, Vector256.Create(Vector128.Load(x3 + i), Vector128<byte>.Zero), c);
        }

        Vector128.Store(ReduceFour(f0, f1, f2, f3), sums);
    }

    /// <summary>
    /// The horizontal sums of four accumulators, as one vector: <c>[Σa, Σb, Σc, Σd]</c>. Three
    /// <c>vphaddd</c> and one add, where four separate <see cref="Vector256.Sum{T}"/> calls are about
    /// five times the instructions — and a tile of sixteen accumulators pays for this once per tile.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> ReduceFour(Vector256<int> a, Vector256<int> b, Vector256<int> c, Vector256<int> d)
    {
        Vector256<int> ab = Avx2.HorizontalAdd(a, b);
        Vector256<int> cd = Avx2.HorizontalAdd(c, d);
        Vector256<int> abcd = Avx2.HorizontalAdd(ab, cd);
        return abcd.GetLower() + abcd.GetUpper();
    }
}
