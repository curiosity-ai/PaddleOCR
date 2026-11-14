using System.Diagnostics;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Models.Vision;

/// <summary>
/// Scaled dot-product attention over a single packed sequence, laid out per head.
/// </summary>
/// <remarks>
/// Scores are materialised a block of query rows at a time, so a long sequence never needs a
/// full <c>[tokens, tokens]</c> matrix — a 5120-patch page would be 100 MB per head — while the
/// block still gives the keys and values enough reuse to stay in cache across it. A row at a
/// time streamed the whole key matrix past for every single query. Work is split across heads.
/// </remarks>
public static class Attention
{
    /// <summary>
    /// Computes <c>softmax(Q · Kᵀ · scale) · V</c> for every head.
    /// </summary>
    /// <param name="queries">Queries as <c>[heads][tokens][headDim]</c>.</param>
    /// <param name="keys">Keys, same layout.</param>
    /// <param name="values">Values, <c>[heads][tokens][rowStride]</c>.</param>
    /// <param name="output">Receives the result, <c>[heads][tokens][rowStride]</c>.</param>
    /// <param name="heads">Number of heads.</param>
    /// <param name="tokens">Sequence length.</param>
    /// <param name="headDim">Width of one head.</param>
    /// <param name="scale">Softmax scale, normally <c>1 / sqrt(headDim)</c>.</param>
    /// <param name="rowStride">
    /// Floats per token in <paramref name="values"/> and <paramref name="output"/>, at least
    /// <paramref name="headDim"/>; a multiple of sixteen keeps every row on a cache line.
    /// </param>
    /// <param name="profile">
    /// Receives the parts' cost when supplied. Attention runs a thread per head, so nothing
    /// outside it can attribute its time; each thread accumulates its own and they are summed.
    /// </param>
    public static void Bidirectional(
        ReadOnlyMemory<float> queries,
        ReadOnlyMemory<float> keys,
        ReadOnlyMemory<float> values,
        Memory<float> output,
        int heads,
        int tokens,
        int headDim,
        float scale,
        int rowStride,
        StageProfile? profile = null)
    {
        // Wide enough that a block amortises the key and value traffic, narrow enough that the
        // scores stay well inside L2 (16 x 5120 floats is 320 KB at the largest page we accept).
        const int Block = 16;

        long transposeTicks = 0, scoreTicks = 0, softmaxTicks = 0, valueTicks = 0;

        // Every row the kernels read or write as a vector starts on a cache line: the transposed
        // keys and the score rows are padded to a multiple of sixteen floats, and the values and
        // the output keep `rowStride` floats per token (80 for a head width of 72, which is five
        // lines where 72 floats is four and a half). A 512-bit load that straddles two lines costs
        // two, and these kernels issue nothing else. Padding moves data and changes no arithmetic.
        int keyStride = (tokens + 15) & ~15;

        Parallelism.For(heads, head =>
        {
            int headOffset = head * tokens * headDim;
            int paddedOffset = head * tokens * rowStride;

            ReadOnlyMemory<float> k = keys.Slice(headOffset, tokens * headDim);
            ReadOnlyMemory<float> v = values.Slice(paddedOffset, tokens * rowStride);

            using PooledBuffer buffer = TensorPool.Rent(Block * keyStride);
            Memory<float> scores = buffer.Memory;

            // One transpose of this head's keys serves every row-block below, of which there are
            // `tokens / Block`. See AttentionKernels for why the score product wants them this
            // way round.
            using PooledBuffer keysT = TensorPool.Rent(headDim * keyStride);
            long mark = Stopwatch.GetTimestamp();
            AttentionKernels.TransposeHead(k.Span, tokens, headDim, keysT.Span, keyStride);
            long transpose = Stopwatch.GetTimestamp() - mark;
            long score = 0, soft = 0, value = 0;

            for (int start = 0; start < tokens; start += Block)
            {
                int rows = Math.Min(Block, tokens - start);
                Memory<float> block = scores[..(rows * keyStride)];

                mark = Stopwatch.GetTimestamp();
                AttentionKernels.Scores(
                    queries.Span.Slice(headOffset + (start * headDim), rows * headDim),
                    rows,
                    headDim,
                    keysT.Span,
                    tokens,
                    keyStride,
                    block.Span,
                    keyStride);
                score += Stopwatch.GetTimestamp() - mark;

                mark = Stopwatch.GetTimestamp();
                Span<float> rowsSpan = block.Span;
                for (int row = 0; row < rows; row++)
                {
                    Kernels.ScaledSoftmax(rowsSpan.Slice(row * keyStride, tokens), scale);
                }

                soft += Stopwatch.GetTimestamp() - mark;
                mark = Stopwatch.GetTimestamp();

                AttentionKernels.Values(
                    block.Span,
                    keyStride,
                    rows,
                    tokens,
                    v.Span,
                    headDim,
                    rowStride,
                    output.Span.Slice(paddedOffset + (start * rowStride), rows * rowStride),
                    rowStride);
                value += Stopwatch.GetTimestamp() - mark;
            }

            if (profile is not null)
            {
                Interlocked.Add(ref transposeTicks, transpose);
                Interlocked.Add(ref scoreTicks, score);
                Interlocked.Add(ref softmaxTicks, soft);
                Interlocked.Add(ref valueTicks, value);
            }
        });

        if (profile is not null)
        {
            profile.AddThreadTicks("attention transpose", transposeTicks, heads);
            profile.AddThreadTicks("attention scores", scoreTicks, heads);
            profile.AddThreadTicks("attention softmax", softmaxTicks, heads);
            profile.AddThreadTicks("attention values", valueTicks, heads);
        }
    }
}
