using PaddleOcrSharp.Core;
using PaddleOcrSharp.Imaging;

namespace PaddleOcrSharp.Models.Vision;

/// <summary>
/// The NaViT-style SigLIP vision encoder of PaddleOCR-VL, followed by the 2×2 patch-merging
/// projector that maps its output into the language model's embedding space.
/// </summary>
/// <remarks>
/// <para>
/// Port of <c>PaddleOCRVisionTransformer</c> and <c>Projector</c> from
/// <c>modeling_paddleocr_vl.py</c>, configured the way
/// <c>PaddleOCRVLForConditionalGeneration.forward</c> calls them:
/// <c>interpolate_pos_encoding=True</c>, <c>use_rope=True</c>, <c>window_size=-1</c> and
/// <c>return_pooler_output=False</c>. The attention-pooling head is therefore never executed and
/// its weights are ignored.
/// </para>
/// <para>
/// Upstream packs several crops into one sequence and relies on <c>flash_attn_varlen</c>'s
/// <c>cu_seqlens</c> to keep attention block-diagonal. Encoding one image per call is equivalent
/// and avoids building the block mask.
/// </para>
/// </remarks>
public sealed class VisionTower
{
    private readonly VisionConfig _config;
    private readonly WeightStore _weights;
    private readonly VisionRotaryEmbedding _rope;
    private readonly PositionEmbeddingInterpolator _positions;
    private readonly LayerWeights[] _layers;
    private readonly float[] _postNormWeight;
    private readonly float[] _postNormBias;
    private readonly Core.WeightMatrix _patchEmbedding;
    private readonly float[] _patchEmbeddingBias;
    private readonly ProjectorWeights _projector;

    /// <summary>Loads the tower from a checkpoint.</summary>
    /// <param name="weights">Open checkpoint.</param>
    /// <param name="config">Vision hyper-parameters.</param>
    /// <param name="languageHiddenSize">Width of the language model's embeddings (1024).</param>
    /// <param name="prefix">Parameter-name prefix; the shipped checkpoint uses <c>visual.</c>.</param>
    public VisionTower(
        WeightStore weights,
        VisionConfig config,
        int languageHiddenSize,
        string prefix = "visual.")
    {
        _config = config;
        _weights = weights;
        _rope = new VisionRotaryEmbedding(config.HeadDim, config.RopeTheta);

        string model = prefix + "vision_model.";

        _patchEmbedding = weights.MatrixFlattened(model + "embeddings.patch_embedding.weight");
        _patchEmbeddingBias = weights.Vector(model + "embeddings.patch_embedding.bias");
        _positions = new PositionEmbeddingInterpolator(
            weights.Vector(model + "embeddings.position_embedding.weight"),
            config.PositionGridSize,
            config.HiddenSize);

        _layers = new LayerWeights[config.NumHiddenLayers];
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            string layer = $"{model}encoder.layers.{i}.";
            _layers[i] = new LayerWeights
            {
                Norm1Weight = weights.Vector(layer + "layer_norm1.weight"),
                Norm1Bias = weights.Vector(layer + "layer_norm1.bias"),
                Norm2Weight = weights.Vector(layer + "layer_norm2.weight"),
                Norm2Bias = weights.Vector(layer + "layer_norm2.bias"),
                Query = weights.Matrix(layer + "self_attn.q_proj.weight", WeightRole.Vision),
                QueryBias = weights.Vector(layer + "self_attn.q_proj.bias"),
                Key = weights.Matrix(layer + "self_attn.k_proj.weight", WeightRole.Vision),
                KeyBias = weights.Vector(layer + "self_attn.k_proj.bias"),
                Value = weights.Matrix(layer + "self_attn.v_proj.weight", WeightRole.Vision),
                ValueBias = weights.Vector(layer + "self_attn.v_proj.bias"),
                Output = weights.Matrix(layer + "self_attn.out_proj.weight", WeightRole.Vision),
                OutputBias = weights.Vector(layer + "self_attn.out_proj.bias"),
                Fc1 = weights.Matrix(layer + "mlp.fc1.weight", WeightRole.Vision),
                Fc1Bias = weights.Vector(layer + "mlp.fc1.bias"),
                Fc2 = weights.Matrix(layer + "mlp.fc2.weight", WeightRole.Vision),
                Fc2Bias = weights.Vector(layer + "mlp.fc2.bias"),
            };
        }

        _postNormWeight = weights.Vector(model + "post_layernorm.weight");
        _postNormBias = weights.Vector(model + "post_layernorm.bias");

        _projector = new ProjectorWeights
        {
            PreNormWeight = weights.Vector("mlp_AR.pre_norm.weight"),
            PreNormBias = weights.Vector("mlp_AR.pre_norm.bias"),
            Linear1 = weights.Matrix("mlp_AR.linear_1.weight", WeightRole.Vision),
            Linear1Bias = weights.Vector("mlp_AR.linear_1.bias"),
            Linear2 = weights.Matrix("mlp_AR.linear_2.weight", WeightRole.Vision),
            Linear2Bias = weights.Vector("mlp_AR.linear_2.bias"),
        };

        LanguageHiddenSize = languageHiddenSize;
        MergedPatchWidth = config.HiddenSize * config.SpatialMergeSize * config.SpatialMergeSize;
    }

    /// <summary>Width of the language model's token embeddings.</summary>
    public int LanguageHiddenSize { get; }

    /// <summary>Width of one merged 2×2 patch group, the projector's input width.</summary>
    public int MergedPatchWidth { get; }

    /// <summary>
    /// Encodes one preprocessed image and projects it into the language model's embedding space.
    /// </summary>
    /// <returns>
    /// A tensor of <c>[grid.TokenCount(mergeSize), languageHiddenSize]</c>. The caller owns it.
    /// </returns>
    public Tensor Encode(PreprocessedImage image, VisionTrace? trace = null, StageProfile? profile = null)
    {
        using Tensor hidden = RunEncoder(image, trace, profile);
        return Project(hidden, image.Grid, trace);
    }

    /// <summary>
    /// Runs patch embedding, the 27 encoder layers and the final layer norm, returning the
    /// per-patch hidden states of shape <c>[patches, hiddenSize]</c>.
    /// </summary>
    public Tensor RunEncoder(PreprocessedImage image, VisionTrace? trace = null, StageProfile? profile = null)
    {
        ImageGrid grid = image.Grid;
        int tokens = grid.PatchCount;
        int width = _config.HiddenSize;
        int heads = _config.NumAttentionHeads;
        int headDim = _config.HeadDim;

        Tensor hidden = Tensor.Rent(tokens, width);

        Gemm.Linear(
            image.PixelValues.Memory,
            tokens,
            _config.PatchVectorLength,
            _patchEmbedding,
            _patchEmbeddingBias,
            hidden.Memory,
            width);

        ReadOnlySpan<float> positions = _positions.Get(grid.Height, grid.Width);
        for (int t = 0; t < grid.Temporal; t++)
        {
            Span<float> slice = hidden.Span.Slice(t * grid.Height * grid.Width * width, grid.Height * grid.Width * width);
            Kernels.AddInPlace(slice, positions);
        }

        trace?.Record("embeddings", hidden.Span, tokens, width);

        using PooledBuffer cos = TensorPool.Rent(tokens * headDim);
        using PooledBuffer sin = TensorPool.Rent(tokens * headDim);
        _rope.Fill(grid.Height, grid.Width, cos.Span, sin.Span);

        using PooledBuffer normed = TensorPool.Rent(tokens * width);
        using PooledBuffer queries = TensorPool.Rent(tokens * width);
        using PooledBuffer keys = TensorPool.Rent(tokens * width);
        // Per-head rows of the values and the attention output are padded to whole cache lines.
        int rowStride = (headDim + 15) & ~15;
        using PooledBuffer values = TensorPool.Rent(heads * tokens * rowStride);
        using PooledBuffer attention = TensorPool.Rent(heads * tokens * rowStride);
        using PooledBuffer packed = TensorPool.Rent(tokens * width);
        using PooledBuffer packedKeys = TensorPool.Rent(tokens * width);
        using PooledBuffer packedValues = TensorPool.Rent(tokens * width);
        using PooledBuffer intermediate = TensorPool.Rent(tokens * _config.IntermediateSize);

        float scale = 1f / MathF.Sqrt(headDim);

        for (int layerIndex = 0; layerIndex < _layers.Length; layerIndex++)
        {
            LayerWeights layer = _layers[layerIndex];

            using (profile?.Measure("norm"))
            {
                hidden.Span.CopyTo(normed.Span);
                Norms.LayerNormParallel(normed.Memory, width, layer.Norm1Weight, layer.Norm1Bias, _config.LayerNormEps);
            }

            // One call for the three projections: the activations are packed once, and the three
            // sets of column panels run in one parallel region.
            using (profile?.Measure("qkv gemm"))
            {
                Gemm.LinearMany(
                    normed.Memory,
                    tokens,
                    width,
                    [
                        new Gemm.LinearPart(layer.Query, layer.QueryBias, packed.Memory),
                        new Gemm.LinearPart(layer.Key, layer.KeyBias, packedKeys.Memory),
                        new Gemm.LinearPart(layer.Value, layer.ValueBias, packedValues.Memory),
                    ]);
            }

            using (profile?.Measure("rope+split"))
            {
                SplitHeadsWithRope(packed.Memory, queries.Memory, tokens, heads, headDim, cos.Memory, sin.Memory);
                SplitHeadsWithRope(packedKeys.Memory, keys.Memory, tokens, heads, headDim, cos.Memory, sin.Memory);
                SplitHeads(packedValues.Memory, values.Memory, tokens, heads, headDim, rowStride);
            }

            using (profile?.Measure("attention"))
            {
                // The parts are recorded from inside, as thread-ticks: attention runs a thread
                // per head, so nothing outside it can see where its time went.
                Attention.Bidirectional(
                    queries.Memory, keys.Memory, values.Memory, attention.Memory,
                    heads, tokens, headDim, scale, rowStride, profile);
            }

            using (profile?.Measure("rope+split"))
            {
                MergeHeads(attention.Memory, packed.Memory, tokens, heads, headDim, rowStride);
            }

            using (profile?.Measure("out gemm"))
            {
                Gemm.Linear(packed.Memory, tokens, width, layer.Output, layer.OutputBias, normed.Memory, width);
            }

            using (profile?.Measure("residual"))
            {
                Kernels.AddInPlace(hidden.Span, normed.Span);
            }

            using (profile?.Measure("norm"))
            {
                hidden.Span.CopyTo(normed.Span);
                Norms.LayerNormParallel(normed.Memory, width, layer.Norm2Weight, layer.Norm2Bias, _config.LayerNormEps);
            }

            using (profile?.Measure("mlp gemm"))
            {
                Gemm.Linear(
                    normed.Memory, tokens, width, layer.Fc1, layer.Fc1Bias, intermediate.Memory, _config.IntermediateSize);
            }

            using (profile?.Measure("gelu"))
            {
                Kernels.GeluTanhParallel(intermediate.Memory);
            }

            using (profile?.Measure("mlp gemm"))
            {
                Gemm.Linear(
                    intermediate.Memory, tokens, _config.IntermediateSize, layer.Fc2, layer.Fc2Bias, normed.Memory, width);
            }

            using (profile?.Measure("residual"))
            {
                Kernels.AddInPlace(hidden.Span, normed.Span);
            }

            trace?.Record($"layer{layerIndex}", hidden.Span, tokens, width);
        }

        Norms.LayerNormParallel(hidden.Memory, width, _postNormWeight, _postNormBias, _config.LayerNormEps);
        trace?.Record("post_layernorm", hidden.Span, tokens, width);

        return hidden;
    }

    /// <summary>
    /// Applies <c>mlp_AR</c>: pre-norm, 2×2 spatial merge, then the two-layer MLP.
    /// </summary>
    /// <remarks>
    /// The merge follows upstream's
    /// <c>rearrange("(t h p1 w p2) d -&gt; (t h w) (p1 p2 d)")</c>: patch <c>(2h + p1, 2w + p2)</c>
    /// contributes to output token <c>(h, w)</c> at feature offset <c>(2·p1 + p2) · d</c>.
    /// </remarks>
    public Tensor Project(Tensor hidden, ImageGrid grid, VisionTrace? trace = null)
    {
        int width = _config.HiddenSize;
        int merge = _config.SpatialMergeSize;
        int mergedHeight = grid.Height / merge;
        int mergedWidth = grid.Width / merge;
        int outputTokens = grid.Temporal * mergedHeight * mergedWidth;

        using PooledBuffer normed = TensorPool.Rent(grid.PatchCount * width);
        hidden.Span[..(grid.PatchCount * width)].CopyTo(normed.Span);

        // Upstream's Projector uses its own LayerNorm with eps 1e-5, not the tower's 1e-6.
        Norms.LayerNormParallel(
            normed.Memory, width, _projector.PreNormWeight, _projector.PreNormBias, 1e-5f);

        using PooledBuffer merged = TensorPool.Rent(outputTokens * MergedPatchWidth);
        Span<float> mergedSpan = merged.Span;
        ReadOnlySpan<float> normedSpan = normed.Span;

        for (int t = 0; t < grid.Temporal; t++)
        {
            int frameBase = t * grid.Height * grid.Width;
            for (int h = 0; h < mergedHeight; h++)
            {
                for (int w = 0; w < mergedWidth; w++)
                {
                    int outputIndex = ((t * mergedHeight) + h) * mergedWidth + w;
                    Span<float> target = mergedSpan.Slice(outputIndex * MergedPatchWidth, MergedPatchWidth);

                    for (int p1 = 0; p1 < merge; p1++)
                    {
                        for (int p2 = 0; p2 < merge; p2++)
                        {
                            int sourceIndex = frameBase + (((h * merge) + p1) * grid.Width) + ((w * merge) + p2);
                            normedSpan.Slice(sourceIndex * width, width)
                                .CopyTo(target.Slice((((p1 * merge) + p2) * width), width));
                        }
                    }
                }
            }
        }

        trace?.Record("projector_merged", mergedSpan, outputTokens, MergedPatchWidth);

        using PooledBuffer hiddenProjection = TensorPool.Rent(outputTokens * MergedPatchWidth);
        Gemm.Linear(
            merged.Memory,
            outputTokens,
            MergedPatchWidth,
            _projector.Linear1,
            _projector.Linear1Bias,
            hiddenProjection.Memory,
            MergedPatchWidth);

        Kernels.GeluErf(hiddenProjection.Span);

        Tensor result = Tensor.Rent(outputTokens, LanguageHiddenSize);
        Gemm.Linear(
            hiddenProjection.Memory,
            outputTokens,
            MergedPatchWidth,
            _projector.Linear2,
            _projector.Linear2Bias,
            result.Memory,
            LanguageHiddenSize);

        trace?.Record("projector", result.Span, outputTokens, LanguageHiddenSize);
        return result;
    }

    /// <summary>
    /// Re-lays one projection from <c>[token, head, d]</c> to <c>[head, token, d]</c>, applying
    /// the 2-D rotary embedding to each head's row on the way.
    /// </summary>
    /// <remarks>
    /// The three shuffles below move <c>patches x 1152</c> floats each — 24 MB a layer at a
    /// full-budget block, four times over — and each head's slice is independent of every other's,
    /// so they are split across the same worker count as the matrix products beside them. They
    /// ran on one core while the GEMMs used four.
    /// </remarks>
    private void SplitHeadsWithRope(
        ReadOnlyMemory<float> packed,
        Memory<float> destination,
        int tokens,
        int heads,
        int headDim,
        ReadOnlyMemory<float> cos,
        ReadOnlyMemory<float> sin)
    {
        int width = heads * headDim;
        Parallelism.For(heads, head =>
        {
            int headBase = head * tokens * headDim;
            ReadOnlySpan<float> source = packed.Span;
            Span<float> target = destination.Span;
            ReadOnlySpan<float> cosines = cos.Span;
            ReadOnlySpan<float> sines = sin.Span;

            for (int token = 0; token < tokens; token++)
            {
                Span<float> row = target.Slice(headBase + (token * headDim), headDim);
                source.Slice((token * width) + (head * headDim), headDim).CopyTo(row);
                _rope.Apply(
                    row, cosines.Slice(token * headDim, headDim), sines.Slice(token * headDim, headDim));
            }
        });
    }

    /// <summary>Re-lays one projection from <c>[token, head, d]</c> to <c>[head, token, d]</c>.</summary>
    private static void SplitHeads(
        ReadOnlyMemory<float> packed,
        Memory<float> destination,
        int tokens,
        int heads,
        int headDim,
        int rowStride)
    {
        int width = heads * headDim;
        Parallelism.For(heads, head =>
        {
            int headBase = head * tokens * rowStride;
            ReadOnlySpan<float> source = packed.Span;
            Span<float> target = destination.Span;

            for (int token = 0; token < tokens; token++)
            {
                source.Slice((token * width) + (head * headDim), headDim)
                    .CopyTo(target.Slice(headBase + (token * rowStride), headDim));
            }
        });
    }

    /// <summary>The inverse of <see cref="SplitHeads"/>, back to <c>[token, head, d]</c>.</summary>
    private static void MergeHeads(
        ReadOnlyMemory<float> perHead,
        Memory<float> destination,
        int tokens,
        int heads,
        int headDim,
        int rowStride)
    {
        int width = heads * headDim;
        Parallelism.For(heads, head =>
        {
            int headBase = head * tokens * rowStride;
            ReadOnlySpan<float> source = perHead.Span;
            Span<float> target = destination.Span;

            for (int token = 0; token < tokens; token++)
            {
                source.Slice(headBase + (token * rowStride), headDim)
                    .CopyTo(target.Slice((token * width) + (head * headDim), headDim));
            }
        });
    }

    private sealed class LayerWeights
    {
        public required float[] Norm1Weight { get; init; }

        public required float[] Norm1Bias { get; init; }

        public required float[] Norm2Weight { get; init; }

        public required float[] Norm2Bias { get; init; }

        public required Core.WeightMatrix Query { get; init; }

        public required float[] QueryBias { get; init; }

        public required Core.WeightMatrix Key { get; init; }

        public required float[] KeyBias { get; init; }

        public required Core.WeightMatrix Value { get; init; }

        public required float[] ValueBias { get; init; }

        public required Core.WeightMatrix Output { get; init; }

        public required float[] OutputBias { get; init; }

        public required Core.WeightMatrix Fc1 { get; init; }

        public required float[] Fc1Bias { get; init; }

        public required Core.WeightMatrix Fc2 { get; init; }

        public required float[] Fc2Bias { get; init; }
    }

    private sealed class ProjectorWeights
    {
        public required float[] PreNormWeight { get; init; }

        public required float[] PreNormBias { get; init; }

        public required Core.WeightMatrix Linear1 { get; init; }

        public required float[] Linear1Bias { get; init; }

        public required Core.WeightMatrix Linear2 { get; init; }

        public required float[] Linear2Bias { get; init; }
    }
}
