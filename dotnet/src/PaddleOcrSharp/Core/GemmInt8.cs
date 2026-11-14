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
            int paddedRows = (rows + 3) & ~3;

            for (int m = 0; m < paddedRows; m += 4)
            {
                int live = Math.Min(4, rows - m);
                byte* xm = px + ((long)m * inner);

                int c = 0;
                for (; c + 4 <= colCount; c += 4)
                {
                    int n = colStart + c;
                    Tile4x4(xm, inner, (sbyte*)(pw + ((long)n * inner)), inner, sums);
                    Store(sums, live, 4, m, n, sx, sw, ws, py, cols, hasBias ? pb : null);
                }

                for (; c < colCount; c++)
                {
                    int n = colStart + c;
                    Tile4x1(xm, inner, (sbyte*)(pw + ((long)n * inner)), sums);
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
    /// Four activation rows against four weight rows: sixteen int32 accumulators of eight lanes, each
    /// step thirty-two bytes of each row. The reduction length is not assumed to be a multiple of
    /// thirty-two; a trailing sixteen are taken on 128-bit lanes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile4x4(byte* x, int k, sbyte* w, int stride, int* sums)
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

        sums[0] = Vector256.Sum(a00);
        sums[1] = Vector256.Sum(a01);
        sums[2] = Vector256.Sum(a02);
        sums[3] = Vector256.Sum(a03);
        sums[4] = Vector256.Sum(a10);
        sums[5] = Vector256.Sum(a11);
        sums[6] = Vector256.Sum(a12);
        sums[7] = Vector256.Sum(a13);
        sums[8] = Vector256.Sum(a20);
        sums[9] = Vector256.Sum(a21);
        sums[10] = Vector256.Sum(a22);
        sums[11] = Vector256.Sum(a23);
        sums[12] = Vector256.Sum(a30);
        sums[13] = Vector256.Sum(a31);
        sums[14] = Vector256.Sum(a32);
        sums[15] = Vector256.Sum(a33);

        if (i < k)
        {
            // k is a multiple of sixteen: one 128-bit step remains.
            Vector128<int> t00 = default, t01 = default, t02 = default, t03 = default;
            Vector128<int> t10 = default, t11 = default, t12 = default, t13 = default;
            Vector128<int> t20 = default, t21 = default, t22 = default, t23 = default;
            Vector128<int> t30 = default, t31 = default, t32 = default, t33 = default;

            Vector128<sbyte> c0 = Vector128.Load(w0 + i);
            Vector128<sbyte> c1 = Vector128.Load(w1 + i);
            Vector128<sbyte> c2 = Vector128.Load(w2 + i);
            Vector128<sbyte> c3 = Vector128.Load(w3 + i);

            Vector128<byte> u = Vector128.Load(x0 + i);
            t00 = AvxVnni.MultiplyWideningAndAdd(t00, u, c0);
            t01 = AvxVnni.MultiplyWideningAndAdd(t01, u, c1);
            t02 = AvxVnni.MultiplyWideningAndAdd(t02, u, c2);
            t03 = AvxVnni.MultiplyWideningAndAdd(t03, u, c3);
            u = Vector128.Load(x1 + i);
            t10 = AvxVnni.MultiplyWideningAndAdd(t10, u, c0);
            t11 = AvxVnni.MultiplyWideningAndAdd(t11, u, c1);
            t12 = AvxVnni.MultiplyWideningAndAdd(t12, u, c2);
            t13 = AvxVnni.MultiplyWideningAndAdd(t13, u, c3);
            u = Vector128.Load(x2 + i);
            t20 = AvxVnni.MultiplyWideningAndAdd(t20, u, c0);
            t21 = AvxVnni.MultiplyWideningAndAdd(t21, u, c1);
            t22 = AvxVnni.MultiplyWideningAndAdd(t22, u, c2);
            t23 = AvxVnni.MultiplyWideningAndAdd(t23, u, c3);
            u = Vector128.Load(x3 + i);
            t30 = AvxVnni.MultiplyWideningAndAdd(t30, u, c0);
            t31 = AvxVnni.MultiplyWideningAndAdd(t31, u, c1);
            t32 = AvxVnni.MultiplyWideningAndAdd(t32, u, c2);
            t33 = AvxVnni.MultiplyWideningAndAdd(t33, u, c3);

            sums[0] += Vector128.Sum(t00);
            sums[1] += Vector128.Sum(t01);
            sums[2] += Vector128.Sum(t02);
            sums[3] += Vector128.Sum(t03);
            sums[4] += Vector128.Sum(t10);
            sums[5] += Vector128.Sum(t11);
            sums[6] += Vector128.Sum(t12);
            sums[7] += Vector128.Sum(t13);
            sums[8] += Vector128.Sum(t20);
            sums[9] += Vector128.Sum(t21);
            sums[10] += Vector128.Sum(t22);
            sums[11] += Vector128.Sum(t23);
            sums[12] += Vector128.Sum(t30);
            sums[13] += Vector128.Sum(t31);
            sums[14] += Vector128.Sum(t32);
            sums[15] += Vector128.Sum(t33);
        }
    }

    /// <summary>Four activation rows against one weight row, for the last few columns of a ragged panel.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile4x1(byte* x, int k, sbyte* w, int* sums)
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

        sums[0] = Vector256.Sum(a0);
        sums[1] = Vector256.Sum(a1);
        sums[2] = Vector256.Sum(a2);
        sums[3] = Vector256.Sum(a3);

        if (i < k)
        {
            Vector128<sbyte> c = Vector128.Load(w + i);
            sums[0] += Vector128.Sum(AvxVnni.MultiplyWideningAndAdd(Vector128<int>.Zero, Vector128.Load(x0 + i), c));
            sums[1] += Vector128.Sum(AvxVnni.MultiplyWideningAndAdd(Vector128<int>.Zero, Vector128.Load(x1 + i), c));
            sums[2] += Vector128.Sum(AvxVnni.MultiplyWideningAndAdd(Vector128<int>.Zero, Vector128.Load(x2 + i), c));
            sums[3] += Vector128.Sum(AvxVnni.MultiplyWideningAndAdd(Vector128<int>.Zero, Vector128.Load(x3 + i), c));
        }
    }
}
