namespace PaddleOcrSharp.Download;

/// <summary>One file inside a model directory.</summary>
/// <param name="Path">Path relative to the model's directory.</param>
/// <param name="Required">Whether the model cannot be used without it.</param>
/// <param name="RemotePath">
/// Directory under the endpoint to fetch this one file from, when it is not the model's own
/// <see cref="ModelDescriptor.RemotePath"/>. A variant that differs from another model in a
/// single file — a quantised checkpoint beside the same tokenizer and configs — names only that
/// file here instead of mirroring a second copy of the rest.
/// </param>
public readonly record struct ModelFile(string Path, bool Required = true, string? RemotePath = null);

/// <summary>A downloadable model.</summary>
/// <param name="Name">Short local name, also the cache directory.</param>
/// <param name="RemotePath">
/// Path under the download endpoint, e.g. <c>paddleocr-vl</c> for
/// <c>https://models.curiosity.ai/paddleocr-vl/</c>. Nested paths are allowed, so a mirror that
/// keeps several versions side by side can use <c>paddleocr-vl/1.6</c>.
/// </param>
/// <param name="Files">Files to fetch.</param>
public sealed record ModelDescriptor(
    string Name,
    string RemotePath,
    IReadOnlyList<ModelFile> Files);

/// <summary>The models the pipeline can fetch.</summary>
/// <remarks>
/// Each entry names a directory in the model mirror <see cref="ModelDownloader"/> reads from,
/// which holds a copy of the files the upstream Hugging Face repository named beside it
/// publishes. The upstream repositories remain the provenance of the weights; they are not what
/// the pipeline downloads from.
/// </remarks>
public static class ModelCatalog
{
    /// <summary>
    /// The 0.9B vision-language model that does the actual recognition. Mirrors
    /// <c>PaddlePaddle/PaddleOCR-VL-1.6</c>.
    /// </summary>
    public static ModelDescriptor PaddleOcrVL16 { get; } = new(
        "PaddleOCR-VL-1.6",
        "paddleocr-vl",
        [
            new ModelFile("config.json"),
            new ModelFile("generation_config.json", Required: false),
            new ModelFile("model.safetensors"),
            new ModelFile("preprocessor_config.json"),
            new ModelFile("processor_config.json", Required: false),
            new ModelFile("tokenizer.json"),
            new ModelFile("tokenizer_config.json", Required: false),
            new ModelFile("added_tokens.json", Required: false),
            new ModelFile("special_tokens_map.json", Required: false),
            new ModelFile("chat_template.jinja", Required: false),
        ]);

    /// <summary>
    /// <see cref="PaddleOcrVL16"/> with its weights replaced by the int8 checkpoint that
    /// <see cref="Models.WeightQuantization.Int8Compute"/> runs from: per-row scales, tower and
    /// decoder both quantised, 1.1 GB against 1.9 GB. Only <c>model.safetensors</c> differs, so
    /// it is the only file the mirror holds under <c>paddleocr-vl/int8-compute/</c>; the
    /// tokenizer and the configs are the original model's and are fetched from there.
    /// </summary>
    public static ModelDescriptor PaddleOcrVL16Int8Compute { get; } = new(
        "PaddleOCR-VL-1.6-int8-compute",
        PaddleOcrVL16.RemotePath,
        [
            .. PaddleOcrVL16.Files.Select(file => file.Path == "model.safetensors"
                ? file with { RemotePath = $"{PaddleOcrVL16.RemotePath}/int8-compute" }
                : file),
        ]);

    /// <summary>
    /// The VL checkpoint that matches <paramref name="quantization"/>: the original for
    /// <see cref="Models.WeightQuantization.None"/>, the pre-quantised one for
    /// <see cref="Models.WeightQuantization.Int8Compute"/>.
    /// </summary>
    public static ModelDescriptor VisionLanguageFor(Models.WeightQuantization quantization) => quantization switch
    {
        Models.WeightQuantization.None => PaddleOcrVL16,
        Models.WeightQuantization.Int8Compute => PaddleOcrVL16Int8Compute,
        _ => throw new ArgumentOutOfRangeException(nameof(quantization), quantization, null),
    };

    /// <summary>
    /// The RT-DETR layout detector used by the 1.6 pipeline. Mirrors
    /// <c>PaddlePaddle/PP-DocLayoutV3</c>.
    /// </summary>
    public static ModelDescriptor PpDocLayoutV3 { get; } = new(
        "PP-DocLayoutV3",
        "pp-doclayoutv3",
        [
            new ModelFile("inference.json"),
            new ModelFile("inference.pdiparams"),
            new ModelFile("inference.yml"),
        ]);

    /// <summary>
    /// Document orientation classifier, used when doc preprocessing is enabled. Mirrors
    /// <c>PaddlePaddle/PP-LCNet_x1_0_doc_ori</c>.
    /// </summary>
    public static ModelDescriptor DocOrientationClassifier { get; } = new(
        "PP-LCNet_x1_0_doc_ori",
        "pp-lcnet-x1-0-doc-ori",
        [
            new ModelFile("inference.json"),
            new ModelFile("inference.pdiparams"),
            new ModelFile("inference.yml"),
            new ModelFile("config.json", Required: false),
        ]);

    /// <summary>
    /// Document unwarping model, used when doc preprocessing is enabled. Mirrors
    /// <c>PaddlePaddle/UVDoc</c>.
    /// </summary>
    public static ModelDescriptor DocUnwarping { get; } = new(
        "UVDoc",
        "uvdoc",
        [
            new ModelFile("inference.json"),
            new ModelFile("inference.pdiparams"),
            new ModelFile("inference.yml"),
            new ModelFile("config.json", Required: false),
        ]);

    /// <summary>Every model the catalogue knows about.</summary>
    public static IReadOnlyList<ModelDescriptor> All { get; } =
        [PaddleOcrVL16, PaddleOcrVL16Int8Compute, PpDocLayoutV3, DocOrientationClassifier, DocUnwarping];

    /// <summary>Finds a model by <see cref="ModelDescriptor.Name"/>, case-insensitively.</summary>
    public static ModelDescriptor? Find(string name) =>
        All.FirstOrDefault(model => string.Equals(model.Name, name, StringComparison.OrdinalIgnoreCase));
}
