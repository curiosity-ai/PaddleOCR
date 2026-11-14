using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace PaddleOcrSharp.Core;

/// <summary>
/// <see cref="Gemm.Linear"/> for many activation rows, with the vector lanes along the
/// <i>rows</i> instead of along the reduction.
/// </summary>
/// <remarks>
/// <para>
/// The dot-form tile that <see cref="Gemm"/> otherwise uses keeps sixteen lane-wise partial sums
/// per output and reduces them at the end. That final reduction, and the load mix that goes with
/// it, cap the tile at about 137 GFLOP/s a thread on the machine this was measured on, whatever
/// its shape. Here each vector holds one weight column's running sum for sixteen activation rows,
/// so a reduction step is one broadcast of a weight and one fused multiply-add per vector, there
/// is nothing to reduce afterwards, and a tile of thirty-two rows by eight columns keeps sixteen
/// independent chains in flight: 172-177 GFLOP/s a thread in isolation.
/// </para>
/// <para>
/// <b>This is not bit-identical to the dot form.</b> Every output here accumulates in a single
/// sequential float32 chain over the reduction, where the dot form accumulates sixteen lane chains
/// and sums them in a fixed tree. Both are float32 throughout, but they round differently, and the
/// sequential chain is about 2.4 times less accurate against a float64 reference (mean relative
/// error 1.7e-6 against 7.5e-7 at a reduction of 1152).
/// </para>
/// <para>
/// <b>It is off unless asked for</b>, with <c>PADDLEOCR_SHARP_GEMM=outer</c>, because the
/// project's bar is byte-identical output and parity over a corpus that was not available when
/// this was written. What was measured, on a synthetic A4 page at fifteen resolutions from 20 to
/// 400 dpi: the markdown is byte-identical to the dot form's at every one of them, including the
/// garbled 20-35 dpi outputs where greedy decoding is least stable; the vision tower's final
/// embeddings differ by a relative L2 of 4e-6 (a text block) to 5e-5 (a table block); and the
/// GEMMs run 11-27% faster, which is 2-6% of a page. See "Alignment and the worker team" in
/// <c>CLAUDE.md</c>.
/// </para>
/// <para>
/// The activations are packed once per call into groups of thirty-two rows laid out
/// <c>[k][32]</c> — the last group zero-padded, so every row takes the same arithmetic — and the
/// weights stay in the row-major panel the dot form already builds. A column of weights is read by
/// broadcast, so nothing about the weights needs to be aligned or transposed, and any reduction
/// length works, not only multiples of sixteen.
/// </para>
/// </remarks>
internal static class GemmOuter
{
    /// <summary>Fewer rows than this and the zero padding wastes too much; the dot form handles them.</summary>
    internal const int MinRows = 32;

    private const int GroupRows = 32;
    private const int Lanes = 16;

    /// <summary>Row groups swept per weight tile, so their packed slice stays in L2.</summary>
    private const int GroupBlock = 4;

    /// <summary>
    /// Reduction steps per pass. A group of thirty-two rows is 128 bytes a step, so four groups over
    /// 1152 steps is 590 KB and stays in L2 while every column tile sweeps it; over 4304 steps it
    /// is 2.2 MB and does not, and the two large-reduction shapes gained 2% and 7% where the
    /// others gained 22%. Longer reductions are cut into chunks of this length with the
    /// accumulators carried through memory between them, which is the same sequential chain
    /// resumed, so the arithmetic is unchanged by it.
    /// </summary>
    private const int Chunk = 1152;

    /// <summary>Whether the outer-product form is in use at all: only when <c>PADDLEOCR_SHARP_GEMM=outer</c>.</summary>
    internal static readonly bool Enabled = string.Equals(
        Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_GEMM"), "outer", StringComparison.OrdinalIgnoreCase);

    /// <summary>Whether a product with this many activation rows takes this path.</summary>
    internal static bool Applies(int rows) => Enabled && Simd.Use512 && rows >= MinRows;

    /// <summary>Computes <c>y = x · Wᵀ + bias</c> in the outer-product form.</summary>
    /// <param name="x">Activations, <c>[rows, inner]</c>.</param>
    /// <param name="rows">Number of activation rows.</param>
    /// <param name="inner">Reduction length.</param>
    /// <param name="weight">Weight, <c>[cols, inner]</c>.</param>
    /// <param name="bias">Optional bias of length <paramref name="cols"/>.</param>
    /// <param name="y">Output, <c>[rows, cols]</c>.</param>
    /// <param name="cols">Number of output columns.</param>
    internal static void Linear(
        ReadOnlyMemory<float> x,
        int rows,
        int inner,
        WeightMatrix weight,
        ReadOnlyMemory<float> bias,
        Memory<float> y,
        int cols)
    {
        int groups = (rows + GroupRows - 1) / GroupRows;
        using PooledBuffer packed = TensorPool.Rent(checked(groups * GroupRows * inner));
        PackAll(x, rows, inner, groups, packed.Memory);

        int panel = Gemm.ChoosePanelWidth(inner, cols, weight.Dtype);
        int panels = (cols + panel - 1) / panel;
        RunPanels(panels, panel, packed.Memory, groups, rows, inner, weight, bias, y, cols);
    }

    // The lambdas live in their own methods: a method that contains one allocates its display
    // class on entry whichever branch runs.
    private static void PackAll(ReadOnlyMemory<float> x, int rows, int inner, int groups, Memory<float> packed) =>
        Parallelism.For(groups, group => PackGroup(x.Span, rows, inner, group, packed.Span));

    private static void RunPanels(
        int panels,
        int panel,
        ReadOnlyMemory<float> packed,
        int groups,
        int rows,
        int inner,
        WeightMatrix weight,
        ReadOnlyMemory<float> bias,
        Memory<float> y,
        int cols) =>
        Parallelism.For(panels, index =>
        {
            int start = index * panel;
            RunPanel(
                packed.Span, groups, rows, inner, weight, bias.Span, y.Span, cols, start, Math.Min(panel, cols - start));
        });

    /// <summary>
    /// Lays one group of thirty-two activation rows out as <c>[k][32]</c>: every reduction step's
    /// values for the group are one contiguous 128 bytes.
    /// </summary>
    private static unsafe void PackGroup(ReadOnlySpan<float> x, int rows, int inner, int group, Span<float> packed)
    {
        fixed (float* px = x, pp = packed)
        {
            float* destination = pp + ((long)group * GroupRows * inner);

            // Sixteen reduction steps at a time, so the thirty-two destination lines a block
            // touches stay in L1 while the rows are walked.
            for (int k0 = 0; k0 < inner; k0 += 16)
            {
                int end = Math.Min(inner, k0 + 16);
                for (int r = 0; r < GroupRows; r++)
                {
                    int row = (group * GroupRows) + r;
                    if (row < rows)
                    {
                        float* source = px + ((long)row * inner);
                        for (int k = k0; k < end; k++)
                        {
                            destination[(k * GroupRows) + r] = source[k];
                        }
                    }
                    else
                    {
                        for (int k = k0; k < end; k++)
                        {
                            destination[(k * GroupRows) + r] = 0f;
                        }
                    }
                }
            }
        }
    }

    private static unsafe void RunPanel(
        ReadOnlySpan<float> packed,
        int groups,
        int rows,
        int inner,
        WeightMatrix weight,
        ReadOnlySpan<float> bias,
        Span<float> y,
        int cols,
        int colStart,
        int colCount)
    {
        bool inPlace = weight.TryGetFloats(out ReadOnlySpan<float> stored);
        using PooledBuffer scratch = inPlace ? default : TensorPool.Rent(colCount * inner);
        if (!inPlace)
        {
            weight.CopyRows(colStart, colCount, scratch.Span);
        }

        ReadOnlySpan<float> panel = inPlace ? stored.Slice(colStart * inner, colCount * inner) : scratch.Span;

        int chunks = (inner + Chunk - 1) / Chunk;

        // One chunk needs a tile's accumulators only for as long as the tile runs. More need every
        // (group, column) pair's thirty-two running sums held between passes.
        using PooledBuffer carried = chunks > 1 ? TensorPool.Rent(groups * colCount * GroupRows) : default;
        float* local = stackalloc float[8 * GroupRows];

        fixed (float* pw = panel, pp = packed, py = y, pb = bias, pc = carried.Span)
        {
            float* bias_ = bias.IsEmpty ? null : pb;

            for (int chunk = 0; chunk < chunks; chunk++)
            {
                int k0 = chunk * Chunk;
                int length = Math.Min(Chunk, inner - k0);
                bool accumulate = chunk > 0;
                bool last = chunk == chunks - 1;

                for (int firstGroup = 0; firstGroup < groups; firstGroup += GroupBlock)
                {
                    int lastGroup = Math.Min(groups, firstGroup + GroupBlock);

                    int column = 0;
                    for (; column + 8 <= colCount; column += 8)
                    {
                        for (int group = firstGroup; group < lastGroup; group++)
                        {
                            float* tile = chunks > 1 ? pc + (((long)group * colCount) + column) * GroupRows : local;
                            float* source = pp + ((long)group * GroupRows * inner) + ((long)k0 * GroupRows);
                            float* weights = pw + ((long)column * inner) + k0;
                            if (accumulate)
                            {
                                Tile32x8Resume(source, length, weights, inner, tile);
                            }
                            else
                            {
                                Tile32x8(source, length, weights, inner, tile);
                            }

                            if (last)
                            {
                                Scatter(tile, 8, group, rows, py, cols, colStart + column, bias_);
                            }
                        }
                    }

                    for (; column + 4 <= colCount; column += 4)
                    {
                        for (int group = firstGroup; group < lastGroup; group++)
                        {
                            float* tile = chunks > 1 ? pc + (((long)group * colCount) + column) * GroupRows : local;
                            float* source = pp + ((long)group * GroupRows * inner) + ((long)k0 * GroupRows);
                            float* weights = pw + ((long)column * inner) + k0;
                            if (accumulate)
                            {
                                Tile32x4Resume(source, length, weights, inner, tile);
                            }
                            else
                            {
                                Tile32x4(source, length, weights, inner, tile);
                            }

                            if (last)
                            {
                                Scatter(tile, 4, group, rows, py, cols, colStart + column, bias_);
                            }
                        }
                    }

                    for (; column < colCount; column++)
                    {
                        for (int group = firstGroup; group < lastGroup; group++)
                        {
                            float* tile = chunks > 1 ? pc + (((long)group * colCount) + column) * GroupRows : local;
                            float* source = pp + ((long)group * GroupRows * inner) + ((long)k0 * GroupRows);
                            float* weights = pw + ((long)column * inner) + k0;
                            if (accumulate)
                            {
                                Tile32x1Resume(source, length, weights, tile);
                            }
                            else
                            {
                                Tile32x1(source, length, weights, tile);
                            }

                            if (last)
                            {
                                Scatter(tile, 1, group, rows, py, cols, colStart + column, bias_);
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Copies a tile's accumulators, held column-major (<c>results[c * 32 + r]</c>), into the
    /// row-major output, adding the bias on the way — the same single float32 add the dot form
    /// applies afterwards — and skipping the zero-padded rows of the last group.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Scatter(
        float* results, int tileColumns, int group, int rows, float* y, int cols, int column, float* bias)
    {
        int first = group * GroupRows;
        int count = Math.Min(GroupRows, rows - first);

        for (int r = 0; r < count; r++)
        {
            float* destination = y + ((long)(first + r) * cols) + column;
            for (int c = 0; c < tileColumns; c++)
            {
                float value = results[(c * GroupRows) + r];
                destination[c] = bias is null ? value : value + bias[column + c];
            }
        }
    }

    /// <summary>
    /// Thirty-two rows by eight weight columns: sixteen accumulators, each one weight column's
    /// running sum for sixteen rows, advanced one reduction step at a time.
    /// </summary>
    /// <param name="packed">One packed group, <c>[k][32]</c>.</param>
    /// <param name="k">Reduction length.</param>
    /// <param name="w">First of eight consecutive weight rows.</param>
    /// <param name="stride">Distance between weight rows.</param>
    /// <param name="results">Receives <c>[8][32]</c>.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile32x8(float* packed, int k, float* w, int stride, float* results)
    {
        float* w0 = w;
        float* w1 = w + stride;
        float* w2 = w + (2L * stride);
        float* w3 = w + (3L * stride);
        float* w4 = w + (4L * stride);
        float* w5 = w + (5L * stride);
        float* w6 = w + (6L * stride);
        float* w7 = w + (7L * stride);

        Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default;
        Vector512<float> c20 = default, c21 = default, c30 = default, c31 = default;
        Vector512<float> c40 = default, c41 = default, c50 = default, c51 = default;
        Vector512<float> c60 = default, c61 = default, c70 = default, c71 = default;

        for (nint i = 0; i < k; i++)
        {
            Vector512<float> lo = Vector512.Load(packed + (i * GroupRows));
            Vector512<float> hi = Vector512.Load(packed + (i * GroupRows) + Lanes);

            Vector512<float> b = Vector512.Create(w0[i]);
            c00 = Vector512.FusedMultiplyAdd(b, lo, c00);
            c01 = Vector512.FusedMultiplyAdd(b, hi, c01);
            b = Vector512.Create(w1[i]);
            c10 = Vector512.FusedMultiplyAdd(b, lo, c10);
            c11 = Vector512.FusedMultiplyAdd(b, hi, c11);
            b = Vector512.Create(w2[i]);
            c20 = Vector512.FusedMultiplyAdd(b, lo, c20);
            c21 = Vector512.FusedMultiplyAdd(b, hi, c21);
            b = Vector512.Create(w3[i]);
            c30 = Vector512.FusedMultiplyAdd(b, lo, c30);
            c31 = Vector512.FusedMultiplyAdd(b, hi, c31);
            b = Vector512.Create(w4[i]);
            c40 = Vector512.FusedMultiplyAdd(b, lo, c40);
            c41 = Vector512.FusedMultiplyAdd(b, hi, c41);
            b = Vector512.Create(w5[i]);
            c50 = Vector512.FusedMultiplyAdd(b, lo, c50);
            c51 = Vector512.FusedMultiplyAdd(b, hi, c51);
            b = Vector512.Create(w6[i]);
            c60 = Vector512.FusedMultiplyAdd(b, lo, c60);
            c61 = Vector512.FusedMultiplyAdd(b, hi, c61);
            b = Vector512.Create(w7[i]);
            c70 = Vector512.FusedMultiplyAdd(b, lo, c70);
            c71 = Vector512.FusedMultiplyAdd(b, hi, c71);
        }

        c00.Store(results);
        c01.Store(results + Lanes);
        c10.Store(results + GroupRows);
        c11.Store(results + GroupRows + Lanes);
        c20.Store(results + (2 * GroupRows));
        c21.Store(results + (2 * GroupRows) + Lanes);
        c30.Store(results + (3 * GroupRows));
        c31.Store(results + (3 * GroupRows) + Lanes);
        c40.Store(results + (4 * GroupRows));
        c41.Store(results + (4 * GroupRows) + Lanes);
        c50.Store(results + (5 * GroupRows));
        c51.Store(results + (5 * GroupRows) + Lanes);
        c60.Store(results + (6 * GroupRows));
        c61.Store(results + (6 * GroupRows) + Lanes);
        c70.Store(results + (7 * GroupRows));
        c71.Store(results + (7 * GroupRows) + Lanes);
    }

    /// <summary>
    /// <see cref="Tile32x8"/> resumed from the sums a previous chunk of the reduction left in
    /// <paramref name="results"/>. A separate method rather than a flag because the flag, a
    /// conditional load of sixteen accumulators ahead of the loop, measured as a slower loop.
    /// </summary>
    /// <param name="packed">One packed group, <c>[k][32]</c>.</param>
    /// <param name="k">Reduction length.</param>
    /// <param name="w">First of eight consecutive weight rows.</param>
    /// <param name="stride">Distance between weight rows.</param>
    /// <param name="results">Holds the running sums, <c>[8][32]</c>, and receives the new ones.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile32x8Resume(float* packed, int k, float* w, int stride, float* results)
    {
        float* w0 = w;
        float* w1 = w + stride;
        float* w2 = w + (2L * stride);
        float* w3 = w + (3L * stride);
        float* w4 = w + (4L * stride);
        float* w5 = w + (5L * stride);
        float* w6 = w + (6L * stride);
        float* w7 = w + (7L * stride);

        Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default;
        Vector512<float> c20 = default, c21 = default, c30 = default, c31 = default;
        Vector512<float> c40 = default, c41 = default, c50 = default, c51 = default;
        Vector512<float> c60 = default, c61 = default, c70 = default, c71 = default;
        c00 = Vector512.Load(results);
        c01 = Vector512.Load(results + Lanes);
        c10 = Vector512.Load(results + GroupRows);
        c11 = Vector512.Load(results + GroupRows + Lanes);
        c20 = Vector512.Load(results + (2 * GroupRows));
        c21 = Vector512.Load(results + (2 * GroupRows) + Lanes);
        c30 = Vector512.Load(results + (3 * GroupRows));
        c31 = Vector512.Load(results + (3 * GroupRows) + Lanes);
        c40 = Vector512.Load(results + (4 * GroupRows));
        c41 = Vector512.Load(results + (4 * GroupRows) + Lanes);
        c50 = Vector512.Load(results + (5 * GroupRows));
        c51 = Vector512.Load(results + (5 * GroupRows) + Lanes);
        c60 = Vector512.Load(results + (6 * GroupRows));
        c61 = Vector512.Load(results + (6 * GroupRows) + Lanes);
        c70 = Vector512.Load(results + (7 * GroupRows));
        c71 = Vector512.Load(results + (7 * GroupRows) + Lanes);

        for (nint i = 0; i < k; i++)
        {
            Vector512<float> lo = Vector512.Load(packed + (i * GroupRows));
            Vector512<float> hi = Vector512.Load(packed + (i * GroupRows) + Lanes);

            Vector512<float> b = Vector512.Create(w0[i]);
            c00 = Vector512.FusedMultiplyAdd(b, lo, c00);
            c01 = Vector512.FusedMultiplyAdd(b, hi, c01);
            b = Vector512.Create(w1[i]);
            c10 = Vector512.FusedMultiplyAdd(b, lo, c10);
            c11 = Vector512.FusedMultiplyAdd(b, hi, c11);
            b = Vector512.Create(w2[i]);
            c20 = Vector512.FusedMultiplyAdd(b, lo, c20);
            c21 = Vector512.FusedMultiplyAdd(b, hi, c21);
            b = Vector512.Create(w3[i]);
            c30 = Vector512.FusedMultiplyAdd(b, lo, c30);
            c31 = Vector512.FusedMultiplyAdd(b, hi, c31);
            b = Vector512.Create(w4[i]);
            c40 = Vector512.FusedMultiplyAdd(b, lo, c40);
            c41 = Vector512.FusedMultiplyAdd(b, hi, c41);
            b = Vector512.Create(w5[i]);
            c50 = Vector512.FusedMultiplyAdd(b, lo, c50);
            c51 = Vector512.FusedMultiplyAdd(b, hi, c51);
            b = Vector512.Create(w6[i]);
            c60 = Vector512.FusedMultiplyAdd(b, lo, c60);
            c61 = Vector512.FusedMultiplyAdd(b, hi, c61);
            b = Vector512.Create(w7[i]);
            c70 = Vector512.FusedMultiplyAdd(b, lo, c70);
            c71 = Vector512.FusedMultiplyAdd(b, hi, c71);
        }

        c00.Store(results);
        c01.Store(results + Lanes);
        c10.Store(results + GroupRows);
        c11.Store(results + GroupRows + Lanes);
        c20.Store(results + (2 * GroupRows));
        c21.Store(results + (2 * GroupRows) + Lanes);
        c30.Store(results + (3 * GroupRows));
        c31.Store(results + (3 * GroupRows) + Lanes);
        c40.Store(results + (4 * GroupRows));
        c41.Store(results + (4 * GroupRows) + Lanes);
        c50.Store(results + (5 * GroupRows));
        c51.Store(results + (5 * GroupRows) + Lanes);
        c60.Store(results + (6 * GroupRows));
        c61.Store(results + (6 * GroupRows) + Lanes);
        c70.Store(results + (7 * GroupRows));
        c71.Store(results + (7 * GroupRows) + Lanes);
    }

    /// <summary>Thirty-two rows by four weight columns, for a panel whose width is not a multiple of eight.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile32x4(float* packed, int k, float* w, int stride, float* results)
    {
        float* w0 = w;
        float* w1 = w + stride;
        float* w2 = w + (2L * stride);
        float* w3 = w + (3L * stride);

        Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default;
        Vector512<float> c20 = default, c21 = default, c30 = default, c31 = default;
        for (nint i = 0; i < k; i++)
        {
            Vector512<float> lo = Vector512.Load(packed + (i * GroupRows));
            Vector512<float> hi = Vector512.Load(packed + (i * GroupRows) + Lanes);

            Vector512<float> b = Vector512.Create(w0[i]);
            c00 = Vector512.FusedMultiplyAdd(b, lo, c00);
            c01 = Vector512.FusedMultiplyAdd(b, hi, c01);
            b = Vector512.Create(w1[i]);
            c10 = Vector512.FusedMultiplyAdd(b, lo, c10);
            c11 = Vector512.FusedMultiplyAdd(b, hi, c11);
            b = Vector512.Create(w2[i]);
            c20 = Vector512.FusedMultiplyAdd(b, lo, c20);
            c21 = Vector512.FusedMultiplyAdd(b, hi, c21);
            b = Vector512.Create(w3[i]);
            c30 = Vector512.FusedMultiplyAdd(b, lo, c30);
            c31 = Vector512.FusedMultiplyAdd(b, hi, c31);
        }

        c00.Store(results);
        c01.Store(results + Lanes);
        c10.Store(results + GroupRows);
        c11.Store(results + GroupRows + Lanes);
        c20.Store(results + (2 * GroupRows));
        c21.Store(results + (2 * GroupRows) + Lanes);
        c30.Store(results + (3 * GroupRows));
        c31.Store(results + (3 * GroupRows) + Lanes);
    }

    /// <summary><see cref="Tile32x4"/> resumed from the sums in <paramref name="results"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile32x4Resume(float* packed, int k, float* w, int stride, float* results)
    {
        float* w0 = w;
        float* w1 = w + stride;
        float* w2 = w + (2L * stride);
        float* w3 = w + (3L * stride);

        Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default;
        Vector512<float> c20 = default, c21 = default, c30 = default, c31 = default;
        c00 = Vector512.Load(results);
        c01 = Vector512.Load(results + Lanes);
        c10 = Vector512.Load(results + GroupRows);
        c11 = Vector512.Load(results + GroupRows + Lanes);
        c20 = Vector512.Load(results + (2 * GroupRows));
        c21 = Vector512.Load(results + (2 * GroupRows) + Lanes);
        c30 = Vector512.Load(results + (3 * GroupRows));
        c31 = Vector512.Load(results + (3 * GroupRows) + Lanes);

        for (nint i = 0; i < k; i++)
        {
            Vector512<float> lo = Vector512.Load(packed + (i * GroupRows));
            Vector512<float> hi = Vector512.Load(packed + (i * GroupRows) + Lanes);

            Vector512<float> b = Vector512.Create(w0[i]);
            c00 = Vector512.FusedMultiplyAdd(b, lo, c00);
            c01 = Vector512.FusedMultiplyAdd(b, hi, c01);
            b = Vector512.Create(w1[i]);
            c10 = Vector512.FusedMultiplyAdd(b, lo, c10);
            c11 = Vector512.FusedMultiplyAdd(b, hi, c11);
            b = Vector512.Create(w2[i]);
            c20 = Vector512.FusedMultiplyAdd(b, lo, c20);
            c21 = Vector512.FusedMultiplyAdd(b, hi, c21);
            b = Vector512.Create(w3[i]);
            c30 = Vector512.FusedMultiplyAdd(b, lo, c30);
            c31 = Vector512.FusedMultiplyAdd(b, hi, c31);
        }

        c00.Store(results);
        c01.Store(results + Lanes);
        c10.Store(results + GroupRows);
        c11.Store(results + GroupRows + Lanes);
        c20.Store(results + (2 * GroupRows));
        c21.Store(results + (2 * GroupRows) + Lanes);
        c30.Store(results + (3 * GroupRows));
        c31.Store(results + (3 * GroupRows) + Lanes);
    }

    /// <summary>Thirty-two rows by one weight column, for the last few columns of a ragged panel.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile32x1(float* packed, int k, float* w, float* results)
    {
        Vector512<float> c0 = default, c1 = default;
        for (nint i = 0; i < k; i++)
        {
            Vector512<float> b = Vector512.Create(w[i]);
            c0 = Vector512.FusedMultiplyAdd(b, Vector512.Load(packed + (i * GroupRows)), c0);
            c1 = Vector512.FusedMultiplyAdd(b, Vector512.Load(packed + (i * GroupRows) + Lanes), c1);
        }

        c0.Store(results);
        c1.Store(results + Lanes);
    }
    /// <summary><see cref="Tile32x1"/> resumed from the sums in <paramref name="results"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile32x1Resume(float* packed, int k, float* w, float* results)
    {
        Vector512<float> c0 = default, c1 = default;
        c0 = Vector512.Load(results);
        c1 = Vector512.Load(results + Lanes);

        for (nint i = 0; i < k; i++)
        {
            Vector512<float> b = Vector512.Create(w[i]);
            c0 = Vector512.FusedMultiplyAdd(b, Vector512.Load(packed + (i * GroupRows)), c0);
            c1 = Vector512.FusedMultiplyAdd(b, Vector512.Load(packed + (i * GroupRows) + Lanes), c1);
        }

        c0.Store(results);
        c1.Store(results + Lanes);
    }
}
