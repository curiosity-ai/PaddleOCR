namespace PaddleOcrSharp.Models;

/// <summary>How the model's weights are held in memory.</summary>
/// <remarks>
/// <para>
/// Decoding is bound by the bytes it streams — 646 MB of bfloat16 weights a token — and the vision
/// tower and the prefill are bound by arithmetic, so one choice of weight format serves both:
/// <see cref="Int8Compute"/> keeps every projection as signed bytes and runs the large products in
/// integer arithmetic. Nothing is calibrated: it is plain round-to-nearest.
/// </para>
/// <para>
/// A weight-only variant, which quantised just the decoder, was dropped: it made decode 28% faster
/// and a page 3%, which is not worth a second checkpoint to ship and a second set of results to
/// explain. See <c>CLAUDE.md</c>.
/// </para>
/// </remarks>
public enum WeightQuantization
{
    /// <summary>Weights stay as the checkpoint stores them: bfloat16 for the shipped model.</summary>
    None = 0,

    /// <summary>
    /// The vision tower, the projector and the decoder's projections and output head are int8
    /// with one scale per row. The activations of any product with at least 32 rows are quantised
    /// per row as well, and the multiply-adds use the AVX-VNNI dot-product instruction, which is
    /// about three times the float32 tile's rate; decode, whose products have few rows, streams
    /// half the bytes and keeps float32 activations. Quantising the activations costs more
    /// accuracy than quantising the weights; see <c>CLAUDE.md</c>. On a machine without AVX-VNNI
    /// this is weights-only, and its output can differ slightly from a machine that has it.
    /// A checkpoint that was quantised ahead of time, by <c>tools/quantization/quantize_checkpoint.py</c>
    /// and published as <see cref="Download.ModelCatalog.PaddleOcrVL16Int8Compute"/>, is read as it
    /// is stored; a bfloat16 one is quantised as it loads.
    /// </summary>
    Int8Compute = 1,
}
