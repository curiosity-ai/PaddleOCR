namespace PaddleOcrSharp.Models;

/// <summary>How the language model's weights are held in memory.</summary>
/// <remarks>
/// <para>
/// Decoding is bound by the bytes it streams — 646 MB of bfloat16 weights a token — so narrower
/// weights make it faster. <see cref="Int8"/> keeps every decoder projection and the output head as
/// signed bytes with one float32 scale per 128 weights, which halves that traffic. The arithmetic
/// stays float32, and nothing is calibrated: it is plain round-to-nearest.
/// </para>
/// <para>
/// The vision tower is not quantised on load, because it is bound by arithmetic and a smaller
/// weight does not make it faster. A checkpoint that was quantised ahead of time, by
/// <c>tools/quantization/quantize_checkpoint.py</c>, is read as it is stored whatever is asked
/// for here, tower included.
/// </para>
/// </remarks>
public enum WeightQuantization
{
    /// <summary>Weights stay as the checkpoint stores them: bfloat16 for the shipped model.</summary>
    None = 0,

    /// <summary>
    /// The decoder's projections and the output head are quantised to int8 as they are loaded, unless
    /// the checkpoint already holds them quantised.
    /// </summary>
    Int8 = 1,

    /// <summary>
    /// <see cref="Int8"/>, and also the vision tower, the projector and the decoder's prefill run in
    /// integer arithmetic: weights have one scale per row, the activations of any product with at
    /// least 32 rows are quantised per row as well, and the multiply-adds use the AVX-VNNI
    /// dot-product instruction, which is about three times the float32 tile's rate. Quantising the
    /// activations costs more accuracy than quantising the weights; see <c>CLAUDE.md</c>. On a
    /// machine without AVX-VNNI this is weights-only, and its output can differ slightly from a
    /// machine that has it.
    /// </summary>
    Int8Compute = 2,
}
