using System.Diagnostics;
using PaddleOcrSharp.Core;
using System.Buffers;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Models.Preprocessing;

namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// The PaddleOCR-VL-1.6 document parsing pipeline: detect layout, crop each region, recognise it
/// with the VL model, then assemble the page.
/// </summary>
/// <remarks>
/// Port of <c>_PaddleOCRVLPipeline</c> in PaddleX. The block dispatch (which instruction and which
/// pixel budget each label gets), the table figure handling, the repetition guard and the
/// OTSL→HTML conversion follow <c>pipeline.py</c> and <c>uilts.py</c>.
/// </remarks>
public sealed class DocumentParser : IDisposable
{
    private readonly PaddleOcrVLModel _model;
    private readonly LayoutDetector? _layout;
    private readonly bool _ownsModel;

    private readonly DocOrientationClassifier? _orientation;
    private readonly DocumentUnwarper? _unwarper;

    /// <summary>Creates a parser over an already-loaded model and detector.</summary>
    /// <param name="model">The vision-language model.</param>
    /// <param name="layout">The layout detector, or <see langword="null"/> for whole-page mode.</param>
    /// <param name="ownsModel">Whether disposing the parser disposes the model.</param>
    /// <param name="orientation">Optional page-orientation classifier.</param>
    /// <param name="unwarper">Optional page-flattening model.</param>
    public DocumentParser(
        PaddleOcrVLModel model,
        LayoutDetector? layout,
        bool ownsModel = false,
        DocOrientationClassifier? orientation = null,
        DocumentUnwarper? unwarper = null)
    {
        _model = model;
        _layout = layout;
        _ownsModel = ownsModel;
        _orientation = orientation;
        _unwarper = unwarper;
    }

    /// <summary>Loads the models from their directories.</summary>
    /// <param name="visionLanguageDirectory">Directory holding the PaddleOCR-VL checkpoint.</param>
    /// <param name="layoutDirectory">Directory holding PP-DocLayoutV3, or <see langword="null"/>.</param>
    /// <param name="orientationDirectory">Directory holding PP-LCNet_x1_0_doc_ori, or <see langword="null"/>.</param>
    /// <param name="unwarpingDirectory">Directory holding UVDoc, or <see langword="null"/>.</param>
    /// <param name="quantization">How to hold the language model's weights; see <see cref="WeightQuantization"/>.</param>
    public static DocumentParser Load(
        string visionLanguageDirectory,
        string? layoutDirectory,
        string? orientationDirectory = null,
        string? unwarpingDirectory = null,
        WeightQuantization quantization = WeightQuantization.None)
    {
        PaddleOcrVLModel model = PaddleOcrVLModel.Load(visionLanguageDirectory, quantization);
        LayoutDetector? layout = layoutDirectory is null ? null : LayoutDetector.Load(layoutDirectory);
        DocOrientationClassifier? orientation = orientationDirectory is null
            ? null
            : DocOrientationClassifier.Load(orientationDirectory);
        DocumentUnwarper? unwarper = unwarpingDirectory is null
            ? null
            : DocumentUnwarper.Load(unwarpingDirectory);

        return new DocumentParser(model, layout, ownsModel: true, orientation, unwarper);
    }

    /// <summary>Parses one page.</summary>
    /// <param name="page">The page image.</param>
    /// <param name="options">Pipeline settings.</param>
    /// <param name="pageIndex">Index recorded on the result.</param>
    /// <param name="progress">Optional per-block progress callback.</param>
    /// <param name="cancellationToken">Cancels between blocks.</param>
    public ParsedPage Parse(
        RgbImage page,
        DocumentParserOptions? options = null,
        int pageIndex = 0,
        IProgress<BlockProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        DocumentParserOptions settings = options ?? DocumentParserOptions.Default;

        // Ambient for the whole parse, so it reaches the kernels without a parameter on every
        // layer between here and them — and it is an AsyncLocal, so it survives the block loop
        // below putting a block's tower and decoder on pool threads.
        using Core.Parallelism.Scope parallelism = Core.Parallelism.Use(settings.Parallelism);
        using PageProfile.Scope whole = settings.StageProfile?.Measure("page") ?? default;

        RgbImage? prepared = null;
        try
        {
            if (settings.UseDocOrientationClassify && _orientation is not null)
            {
                using (settings.StageProfile?.Measure("orientation"))
                {
                    prepared = _orientation.Correct(page);
                }

                page = prepared;
            }

            if (settings.UseDocUnwarping && _unwarper is not null)
            {
                RgbImage flattened;
                using (settings.StageProfile?.Measure("unwarp"))
                {
                    flattened = _unwarper.Unwarp(page);
                }

                prepared?.Dispose();
                prepared = flattened;
                page = flattened;
            }

            return ParseCore(page, settings, pageIndex, progress, cancellationToken);
        }
        finally
        {
            prepared?.Dispose();
        }
    }

    private ParsedPage ParseCore(
        RgbImage page,
        DocumentParserOptions settings,
        int pageIndex,
        IProgress<BlockProgress>? progress,
        CancellationToken cancellationToken)
    {
        PageProfile? stages = settings.StageProfile;

        IReadOnlyList<LayoutBox> detected;
        using (stages?.Measure("layout"))
        {
            detected = settings.UseLayoutDetection && _layout is not null
                ? _layout.Detect(page, settings.Layout, settings.LayoutProfile)
                : [WholePage(page, settings.WholePageLabel)];
        }

        // The page's pictures are gathered before the overlap filter runs, which is what lets a
        // figure a table swallowed still be identified after the filter has dropped it.
        IReadOnlyList<DocumentFigure> figures = GatherFigures(detected, page.Width, page.Height);

        IReadOnlyList<LayoutBox> regions = settings.UseLayoutDetection && _layout is not null
            ? OverlapFilter.Apply(detected)
            : detected;

        var crops = new RgbImage[regions.Count];
        var sizes = new (int Width, int Height)[regions.Count];

        try
        {
            using (stages?.Measure("crop"))
            {
            for (int i = 0; i < regions.Count; i++)
            {
                LayoutBox region = regions[i].ClampTo(page.Width, page.Height);

                // Truncated, not rounded outward: `restructured_boxes` clamps each edge into the
                // page and then takes `int(...)` of all four, so the right and bottom edges fall
                // back to the pixel they are inside rather than the next one out.
                int left = (int)region.Left;
                int top = (int)region.Top;
                crops[i] = page.Crop(left, top, (int)region.Right, (int)region.Bottom);

                // A region with an outline keeps only what the outline covers; the rest of the
                // crop goes white so the model does not read a neighbour's text out of the
                // corners of a slanted or L-shaped block.
                if (region.Polygon is { Length: > 2 } outline)
                {
                    MaskToPolygon(crops[i], outline, left, top);
                }

                sizes[i] = (crops[i].Width, crops[i].Height);
            }
            }

            IReadOnlyCollection<string> nonMergeLabels = NonMergeLabels(settings);
            List<BlockGroup> groups = settings.MergeLayoutBlocks
                ? BlockMerger.Group(regions, sizes, nonMergeLabels)
                : [.. Enumerable.Range(0, regions.Count).Select(i => new BlockGroup([i], []))];

            // Grouping is also what settles the page's block order: a region caught inside a
            // merged run's span is emitted after the run, not in its original position.
            int[] emission = [.. groups.SelectMany(group => group.Indices)];

            var blocks = new ParsedBlock[regions.Count];
            var tokenizedByBlock = new IReadOnlyList<TokenizedFigure>?[regions.Count];
            var absorbed = new HashSet<string>(StringComparer.Ordinal);
            int completed = 0;

            // Everything a group needs before the model: the stacked crop, the table's figure
            // placeholders and the prepared image. Kept apart from the model call so a batch of
            // groups can be built, decoded together and finished together.
            (BlockGroup Group, int Primary, LayoutBox Region, BlockRequest Request,
                IReadOnlyList<TokenizedFigure> Tokenized) BuildWork(int groupIndex)
            {
                cancellationToken.ThrowIfCancellationRequested();

                BlockGroup group = groups[groupIndex];
                int primary = group.Indices[0];
                LayoutBox region = regions[primary].ClampTo(page.Width, page.Height);

                RgbImage mergedImage;
                using (stages?.Measure("stack"))
                {
                    mergedImage = group.Indices.Count == 1
                        ? crops[primary].Clone()
                        : ImageStacker.Stack(
                            [.. group.Indices.Select(index => crops[index])], group.Alignments);
                }

                using RgbImage merged = mergedImage;

                IReadOnlyList<TokenizedFigure> tokenized = [];
                RgbImage prepared = merged;

                if (region.Label == "table" && settings.TokenizeTableFigures)
                {
                    using (stages?.Measure("table-figures"))
                    {
                        (prepared, tokenized, IReadOnlyList<string> swallowed) =
                            TableFigureTokenizer.Tokenize(merged, region, figures);

                        lock (absorbed)
                        {
                            absorbed.UnionWith(swallowed);
                        }
                    }
                }

                try
                {
                    // `PrepareForModel` copies what it keeps, so both images go back here.
                    return (group, primary, region, PrepareForModel(prepared, region, settings), tokenized);
                }
                finally
                {
                    if (!ReferenceEquals(prepared, merged))
                    {
                        prepared.Dispose();
                    }
                }
            }

            void FinishWork(
                (BlockGroup Group, int Primary, LayoutBox Region, BlockRequest Request,
                    IReadOnlyList<TokenizedFigure> Tokenized) work,
                string raw)
            {
                blocks[work.Primary] = Complete(work.Request, raw, settings) with { GroupId = work.Group.GroupId };

                if (work.Tokenized.Count > 0)
                {
                    // Substituting the placeholders has to wait until every block is recognised,
                    // because a figure's own text goes in beside its image.
                    tokenizedByBlock[work.Primary] = work.Tokenized;
                }

                // Every region of a merged group keeps its box so the JSON still describes the
                // page, but only the first carries the recognised text.
                for (int i = 1; i < work.Group.Indices.Count; i++)
                {
                    int index = work.Group.Indices[i];
                    LayoutBox other = regions[index].ClampTo(page.Width, page.Height);
                    blocks[index] = new ParsedBlock(other.Label, other, string.Empty, other.ReadingOrder)
                    {
                        GroupId = work.Group.GroupId,
                    };
                }

                progress?.Report(new BlockProgress(
                    Interlocked.Increment(ref completed) - 1, groups.Count, work.Region.Label));
            }

            void RecognizeGroup(int groupIndex)
            {
                var work = BuildWork(groupIndex);
                using BlockRequest request = work.Request;

                string raw = string.Empty;
                if (request.NeedsModel)
                {
                    using (stages?.Measure("recognize"))
                    {
                        raw = _model.Recognize(
                            request.Prepared!,
                            request.Instruction,
                            BlockPrompt.Options(work.Region.Label, settings.PixelBudgets),
                            request.Generation,
                            settings.Profile,
                            work.Region.Label,
                            cancellationToken);
                    }
                }

                FinishWork(work, raw);
            }

            if (settings.DecodeBatch != 1 && groups.Count > 1)
            {
                // Batches are closed by the cache they would hold, not by a count: a page of forty
                // headings decodes in one pass where a page of full-budget blocks still splits.
                var pending = new List<(BlockGroup Group, int Primary, LayoutBox Region,
                    BlockRequest Request, IReadOnlyList<TokenizedFigure> Tokenized)>();
                long pendingBytes = 0;

                for (int groupIndex = 0; groupIndex < groups.Count; groupIndex++)
                {
                    var work = BuildWork(groupIndex);
                    long bytes = EstimateCacheBytes(work.Request, settings);

                    bool full = pending.Count > 0
                        && (pendingBytes + bytes > settings.DecodeBatchBytes
                            || (settings.DecodeBatch > 0 && pending.Count >= settings.DecodeBatch));

                    if (full)
                    {
                        RecognizeBatch(pending, FinishWork, settings, stages, cancellationToken);
                        pending = [];
                        pendingBytes = 0;
                    }

                    pending.Add(work);
                    pendingBytes += bytes;
                }

                if (pending.Count > 0)
                {
                    RecognizeBatch(pending, FinishWork, settings, stages, cancellationToken);
                }
            }
            else if (settings.BlockConcurrency > 1)
            {
                Parallel.For(
                    0,
                    groups.Count,
                    new ParallelOptions
                    {
                        MaxDegreeOfParallelism = settings.BlockConcurrency,
                        CancellationToken = cancellationToken,
                    },
                    RecognizeGroup);
            }
            else
            {
                for (int groupIndex = 0; groupIndex < groups.Count; groupIndex++)
                {
                    RecognizeGroup(groupIndex);
                }
            }

            using PageProfile.Scope assembly = stages?.Measure("assemble") ?? default;

            for (int i = 0; i < blocks.Length; i++)
            {
                if (tokenizedByBlock[i] is { Count: > 0 } tokenized)
                {
                    blocks[i] = blocks[i] with
                    {
                        Content = TableFigureTokenizer.Untokenize(
                            blocks[i].Content,
                            tokenized,
                            settings.Markdown.ImageDirectory,
                            path => blocks.FirstOrDefault(
                                block => block?.ImagePath == path)?.Content),
                    };
                }
            }

            // Figures that a table absorbed are described inside its HTML, so they no longer
            // belong to the page as separate blocks.
            IReadOnlyList<ParsedBlock> retained =
            [
                .. emission
                    .Select(index => blocks[index])
                    .Where(block => absorbed.Count == 0
                        || block.ImagePath is null
                        || !absorbed.Contains(block.ImagePath)),
            ];

            return new ParsedPage(pageIndex, page.Width, page.Height, BlockOrder.Assign(retained, settings.Markdown.IgnoredLabels));
        }
        finally
        {
            foreach (RgbImage? crop in crops)
            {
                crop?.Dispose();
            }
        }
    }

    /// <summary>
    /// Labels that are never merged with a neighbour: figures, tables, and whichever of charts and
    /// seals are being kept as images rather than recognised.
    /// </summary>
    private static string[] NonMergeLabels(DocumentParserOptions options) =>
        [.. SkippedByModel(options), "table"];

    /// <summary>
    /// Labels whose blocks are not sent to the model at all, because they are kept as pictures.
    /// </summary>
    /// <remarks><c>image_labels</c>.</remarks>
    private static List<string> SkippedByModel(DocumentParserOptions options)
    {
        var labels = options.UseOcrForImageBlocks ? [] : new List<string>(BlockLabels.ImageLabels);

        if (!options.UseChartRecognition)
        {
            labels.Add("chart");
        }

        if (!options.UseSealRecognition)
        {
            labels.Add("seal");
        }

        return labels;
    }

    /// <summary>
    /// Labels whose blocks get their crop attached as a picture.
    /// </summary>
    /// <remarks>
    /// <c>vis_image_labels</c>, and deliberately not the same set as
    /// <see cref="SkippedByModel"/>: the two decisions are independent upstream. A seal is always
    /// kept as a picture even when it is also recognised, and asking for OCR on figures gives a
    /// figure both a picture and recognised text rather than swapping one for the other.
    /// </remarks>
    private static bool KeptAsPicture(string label, DocumentParserOptions options) =>
        BlockLabels.ImageLabels.Contains(label)
        || label == "seal"
        || (label == "chart" && !options.UseChartRecognition);

    private ParsedBlock Recognize(
        RgbImage crop,
        LayoutBox region,
        DocumentParserOptions options,
        CancellationToken cancellationToken)
    {
        using BlockRequest request = PrepareForModel(crop, region, options);
        if (!request.NeedsModel)
        {
            return Complete(request, string.Empty, options);
        }

        VisionPreprocessorOptions preprocessing = BlockPrompt.Options(region.Label, options.PixelBudgets);
        string raw = _model.Recognize(
            request.Prepared!,
            request.Instruction,
            preprocessing,
            request.Generation,
            options.Profile,
            region.Label,
            cancellationToken);

        return Complete(request, raw, options);
    }

    /// <summary>
    /// Runs the vision tower over a batch's blocks, <c>BlockConcurrency</c> of them at a time.
    /// </summary>
    /// <remarks>
    /// The tower's own kernels spread over every core, but not perfectly: a block of a few hundred
    /// patches leaves the fourth core idle for part of every layer, because the norms, the head
    /// shuffles and the tails of each matrix product are too short to fill it. Encoding two or
    /// four blocks at once fills those gaps with another block's work, and each one's kernels are
    /// narrowed to match so the two settings do not multiply into more threads than cores. It is
    /// off by default because the win is a property of small blocks and the cost is holding
    /// several blocks' activations at once.
    /// </remarks>
    private void EncodeMembers(
        IReadOnlyList<(BlockGroup Group, int Primary, LayoutBox Region, BlockRequest Request,
            IReadOnlyList<TokenizedFigure> Tokenized)> works,
        IReadOnlyList<int> members,
        (PreprocessedImage Image, Tensor Features, TimeSpan Elapsed)[] encoded,
        DocumentParserOptions settings,
        CancellationToken cancellationToken)
    {
        int concurrency = Math.Clamp(settings.BlockConcurrency, 1, Math.Max(1, members.Count));
        if (concurrency == 1)
        {
            for (int r = 0; r < members.Count; r++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                encoded[r] = EncodeOne(works[members[r]].Request, works[members[r]].Region, settings);
            }

            return;
        }

        int workers = Math.Max(1, (Core.Parallelism.Options.MaxDegreeOfParallelism <= 0
            ? Environment.ProcessorCount
            : Core.Parallelism.Options.MaxDegreeOfParallelism) / concurrency);

        var inner = new ParallelOptions
        {
            MaxDegreeOfParallelism = workers,
            CancellationToken = cancellationToken,
        };

        Parallel.For(
            0,
            members.Count,
            new ParallelOptions { MaxDegreeOfParallelism = concurrency, CancellationToken = cancellationToken },
            r =>
            {
                using Core.Parallelism.Scope scope = Core.Parallelism.Use(inner);
                encoded[r] = EncodeOne(works[members[r]].Request, works[members[r]].Region, settings);
            });
    }

    /// <summary>Preprocesses and encodes one block's crop.</summary>
    private (PreprocessedImage Image, Tensor Features, TimeSpan Elapsed) EncodeOne(
        BlockRequest request,
        LayoutBox region,
        DocumentParserOptions settings)
    {
        long start = Stopwatch.GetTimestamp();
        PreprocessedImage image = VisionPreprocessor.Preprocess(
            request.Prepared!, BlockPrompt.Options(region.Label, settings.PixelBudgets));
        Tensor features = _model.Vision.Encode(image);
        return (image, features, Stopwatch.GetElapsedTime(start));
    }

    /// <summary>
    /// What one block's key/value cache will cost, close enough to size a batch by.
    /// </summary>
    /// <remarks>
    /// The prompt is one token per merged 2x2 patch group plus a dozen of instruction, and the
    /// patch count follows from the crop's area once the label's pixel budget has clamped it — all
    /// of which is known before the tower runs. The initial generation allowance matches what
    /// <c>GenerateBatch</c> reserves; a block that outgrows it simply grows its own cache.
    /// </remarks>
    private long EstimateCacheBytes(BlockRequest request, DocumentParserOptions settings)
    {
        if (!request.NeedsModel || request.Prepared is not { } crop)
        {
            return 0;
        }

        (int minimum, int maximum) = settings.PixelBudgets.For(request.Region.Label);
        long pixels = Math.Clamp((long)crop.Width * crop.Height, minimum, maximum);

        int patch = _model.Configuration.Vision.PatchSize;
        int merge = _model.Configuration.Vision.SpatialMergeSize;
        long promptTokens = (pixels / ((long)patch * patch * merge * merge)) + 32;

        Models.Language.LanguageConfig language = _model.Configuration.Language;
        long capacity = promptTokens + Math.Min(request.Generation.MaxNewTokens, 128);
        return capacity * language.NumHiddenLayers * 2 * language.KeyValueWidth * sizeof(float);
    }

    /// <summary>
    /// Encodes and prefills a batch of blocks, then decodes them together, one step for all.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The vision tower and the prefill are compute-bound and already use every core, so they run
    /// per block as before. The token loop is not: it reads 646 MB of weights to produce one
    /// token, so a page of small blocks spends its decode budget re-reading the same weights one
    /// block at a time. Stepping the batch together reads them once for the whole batch, and the
    /// batch costs as many steps as its longest block needs rather than as many as all of them
    /// together need.
    /// </para>
    /// <para>
    /// The cost is holding the batch's key/value caches at once, which is why the batch is sized
    /// rather than unbounded.
    /// </para>
    /// </remarks>
    private void RecognizeBatch(
        IReadOnlyList<(BlockGroup Group, int Primary, LayoutBox Region, BlockRequest Request,
            IReadOnlyList<TokenizedFigure> Tokenized)> works,
        Action<(BlockGroup Group, int Primary, LayoutBox Region, BlockRequest Request,
            IReadOnlyList<TokenizedFigure> Tokenized), string> finish,
        DocumentParserOptions settings,
        PageProfile? stages,
        CancellationToken cancellationToken)
    {
        var images = new List<PreprocessedImage>();
        var embeddings = new List<Tensor>();
        var requests = new List<PaddleOcrVLModel.BatchedRequest>();
        var members = new List<int>();
        var visionTimes = new List<TimeSpan>();

        try
        {
            using (stages?.Measure("recognize"))
            {
                for (int i = 0; i < works.Count; i++)
                {
                    if (works[i].Request.NeedsModel)
                    {
                        members.Add(i);
                    }
                }

                var encoded = new (PreprocessedImage Image, Tensor Features, TimeSpan Elapsed)[members.Count];
                EncodeMembers(works, members, encoded, settings, cancellationToken);

                for (int r = 0; r < members.Count; r++)
                {
                    images.Add(encoded[r].Image);
                    embeddings.Add(encoded[r].Features);
                    visionTimes.Add(encoded[r].Elapsed);
                    requests.Add(new PaddleOcrVLModel.BatchedRequest(
                        _model.BuildPrompt(encoded[r].Image.Grid, works[members[r]].Request.Instruction),
                        encoded[r].Features,
                        encoded[r].Image.Grid,
                        works[members[r]].Request.Generation));
                }

                List<int>[] generated = _model.GenerateBatch(requests, out GenerationStats[] stats, cancellationToken);

                if (settings.Profile is { } profile)
                {
                    for (int r = 0; r < requests.Count; r++)
                    {
                        ImageGrid grid = requests[r].Grid;
                        profile.Add(new RecognitionRecord(
                            works[members[r]].Region.Label,
                            grid.Height * grid.Width * grid.Temporal,
                            stats[r].PromptTokens,
                            stats[r].GeneratedTokens,
                            stats[r].HitTokenBudget,
                            stats[r].StoppedEarly,
                            visionTimes[r],
                            stats[r].Prefill,
                            stats[r].Decode,
                            stats[r].DecodeLogits,
                            0,
                            0));
                    }
                }

                int next = 0;
                for (int i = 0; i < works.Count; i++)
                {
                    string raw = works[i].Request.NeedsModel
                        ? _model.Tokenizer.Decode(
                            generated[next++], works[i].Request.Generation.SkipSpecialTokens)
                        : string.Empty;

                    finish(works[i], raw);
                }
            }
        }
        finally
        {
            foreach (Tensor features in embeddings)
            {
                features.Dispose();
            }

            foreach (PreprocessedImage image in images)
            {
                image.Dispose();
            }

            foreach (var work in works)
            {
                work.Request.Dispose();
            }
        }
    }

    /// <summary>
    /// Everything a block needs before the model sees it: the picture it keeps, whether it is sent
    /// at all, its instruction and its prepared crop.
    /// </summary>
    /// <remarks>
    /// Split out of <see cref="Recognize"/> so a batched decode can run this for several blocks,
    /// hand the model all of them at once, and finish each with <see cref="Complete"/>. The
    /// single-block path calls the three in a row and is unchanged by the split.
    /// </remarks>
    private BlockRequest PrepareForModel(RgbImage crop, LayoutBox region, DocumentParserOptions options)
    {
        byte[]? picture = null;
        string? picturePath = null;

        if (KeptAsPicture(region.Label, options))
        {
            using (options.StageProfile?.Measure("encode-figure"))
            {
                picture = ImageIO.EncodeJpeg(crop);
            }

            picturePath = FigurePath(region);
        }

        if (SkippedByModel(options).Contains(region.Label))
        {
            return new BlockRequest(region, crop.Width, crop.Height)
            {
                Picture = picture,
                PicturePath = picturePath,
            };
        }

        string instruction = region.Label == "spotting"
            ? BlockPrompt.Spotting
            : BlockPrompt.For(region.Label, options.UseChartRecognition, options.UseSealRecognition);

        RgbImage prepared;
        using (options.StageProfile?.Measure("prepare"))
        {
            prepared = Prepare(crop, region.Label, instruction);
        }

        return new BlockRequest(region, crop.Width, crop.Height)
        {
            Picture = picture,
            PicturePath = picturePath,
            Prepared = prepared,
            Instruction = instruction,

            // Spotting encodes coordinates as `<|LOC_n|>` tokens, which are special tokens:
            // dropping them during decoding would erase the geometry the mode exists to produce.
            Generation = region.Label == "spotting"
                ? options.Generation with { SkipSpecialTokens = false }
                : options.Generation,
            NeedsModel = true,
        };
    }

    /// <summary>Turns one block's raw model output back into a parsed block.</summary>
    private static ParsedBlock Complete(BlockRequest request, string raw, DocumentParserOptions options)
    {
        LayoutBox region = request.Region;

        if (!request.NeedsModel)
        {
            return new ParsedBlock(region.Label, region, string.Empty, region.ReadingOrder)
            {
                Image = request.Picture,
                ImagePath = request.PicturePath,
            };
        }

        string content = RepetitionTruncator.Truncate(
            raw,
            region.Label == "table"
                ? RepetitionTruncator.TableMinimumLength
                : RepetitionTruncator.BlockMinimumLength);

        content = NormalizeMathDelimiters(content, region.Label);

        if (region.Label == "table")
        {
            using (options.StageProfile?.Measure("otsl-to-html"))
            {
                string html = OtslTable.ToHtml(content);
                if (html.Length > 0)
                {
                    content = html;
                }
            }
        }

        IReadOnlyList<SpottedText> spotted = [];
        if (region.Label == "spotting")
        {
            (content, spotted) = Spotting.Parse(content, request.CropWidth, request.CropHeight);
        }

        return new ParsedBlock(region.Label, region, content, region.ReadingOrder)
        {
            SpottedText = spotted,
            Image = request.Picture,
            ImagePath = request.PicturePath,
        };
    }

    /// <summary>One block between its crop and its text.</summary>
    private sealed class BlockRequest(LayoutBox region, int cropWidth, int cropHeight) : IDisposable
    {
        public LayoutBox Region { get; } = region;

        public int CropWidth { get; } = cropWidth;

        public int CropHeight { get; } = cropHeight;

        public byte[]? Picture { get; init; }

        public string? PicturePath { get; init; }

        public RgbImage? Prepared { get; init; }

        public string Instruction { get; init; } = string.Empty;

        public GenerationOptions Generation { get; init; } = GenerationOptions.Default;

        public bool NeedsModel { get; init; }

        public void Dispose() => Prepared?.Dispose();
    }

    /// <summary>
    /// Applies the per-instruction crop preparation: formulas get their margins trimmed, and a
    /// small spotting crop is doubled with Lanczos so the coordinate grid has room to resolve.
    /// </summary>
    private static RgbImage Prepare(RgbImage crop, string label, string instruction)
    {
        if (instruction == BlockPrompt.Formula)
        {
            return CropMargin(crop);
        }

        if (label == "spotting"
            && crop.Width < Spotting.UpscaleBelow
            && crop.Height < Spotting.UpscaleBelow)
        {
            return PilResize.ResizeLanczos(crop, crop.Width * 2, crop.Height * 2);
        }

        return crop.Clone();
    }

    /// <summary>Whitens every pixel of <paramref name="crop"/> outside <paramref name="outline"/>.</summary>
    private static void MaskToPolygon(RgbImage crop, (float X, float Y)[] outline, int left, int top)
    {
        var shifted = new (int X, int Y)[outline.Length];
        for (int i = 0; i < outline.Length; i++)
        {
            shifted[i] = ((int)outline[i].X - left, (int)outline[i].Y - top);
        }

        bool[] inside = Polygons.Fill(shifted, crop.Width, crop.Height);

        for (int y = 0; y < crop.Height; y++)
        {
            Span<byte> row = crop.Row(y);
            int rowBase = y * crop.Width;
            for (int x = 0; x < crop.Width; x++)
            {
                if (inside[rowBase + x])
                {
                    continue;
                }

                row[x * 3] = 255;
                row[(x * 3) + 1] = 255;
                row[(x * 3) + 2] = 255;
            }
        }
    }

    /// <summary>
    /// The pictures a page contains, as <c>gather_imgs</c> collects them.
    /// </summary>
    /// <remarks>
    /// A narrower set of labels than the blocks that keep a picture: running-head and running-foot
    /// images are not gathered, and a seal is. This is the list a table's figure tokenisation
    /// works from, so the difference decides which pictures a table can swallow.
    /// </remarks>
    private static List<DocumentFigure> GatherFigures(
        IReadOnlyList<LayoutBox> detected,
        int pageWidth,
        int pageHeight)
    {
        var figures = new List<DocumentFigure>();

        foreach (LayoutBox box in detected)
        {
            if (box.Label is not ("image" or "figure" or "seal"))
            {
                continue;
            }

            LayoutBox clamped = box.ClampTo(pageWidth, pageHeight);
            if (clamped.Right <= clamped.Left || clamped.Bottom <= clamped.Top)
            {
                continue;
            }

            figures.Add(new DocumentFigure(FigurePath(box), box));
        }

        return figures;
    }

    /// <summary>
    /// File name a figure block is written to, as <c>construct_img_path</c>.
    /// </summary>
    /// <remarks>
    /// The box makes the name unique within a page, which is what lets a table's
    /// <c>[Fn]</c> placeholder be resolved back to the figure it covered. JPEG, because that is
    /// the extension upstream writes and the format its saver then infers from it.
    /// </remarks>
    private static string FigurePath(LayoutBox region) =>
        $"img_in_{region.Label}_box_{(int)region.Left}_{(int)region.Top}_{(int)region.Right}_{(int)region.Bottom}.jpg";

    /// <summary>
    /// Rewrites LaTeX delimiters the model emits into the <c>$</c> forms markdown renders.
    /// </summary>
    /// <remarks>
    /// Port of the delimiter fix-up in <c>_paddleocr_vl_assemble_parsing_results</c>: when the
    /// output uses <c>\(…\)</c> or <c>\[…\]</c>, existing <c>$</c> signs are currency and are
    /// stripped before the delimiters are converted.
    /// </remarks>
    private static string NormalizeMathDelimiters(string content, string label)
    {
        bool hasInline = content.Contains("\\(", StringComparison.Ordinal)
            && content.Contains("\\)", StringComparison.Ordinal);
        bool hasDisplay = content.Contains("\\[", StringComparison.Ordinal)
            && content.Contains("\\]", StringComparison.Ordinal);

        if (!hasInline && !hasDisplay)
        {
            return content;
        }

        string result = content
            .Replace("$", string.Empty, StringComparison.Ordinal)
            .Replace("\\(", " $ ", StringComparison.Ordinal)
            .Replace("\\)", " $", StringComparison.Ordinal)
            .Replace("\\[\\[", "\\[", StringComparison.Ordinal)
            .Replace("\\]\\]", "\\]", StringComparison.Ordinal)
            .Replace("\\[", " $$ ", StringComparison.Ordinal)
            .Replace("\\]", " $$ ", StringComparison.Ordinal);

        return label == "formula_number"
            ? result.Replace("$", string.Empty, StringComparison.Ordinal)
            : result;
    }

    /// <summary>
    /// Trims uniform margins from a formula crop so the model sees the glyphs at a useful scale.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Port of <c>crop_margin</c>, and of the <c>w &gt; 2 and h &gt; 2</c> test the caller applies
    /// to its result: a crop thinner than three pixels either way is discarded and the untrimmed
    /// region used instead.
    /// </para>
    /// <para>
    /// The grey is stretched to the full range before the threshold, so what counts as background
    /// depends on the crop's own contrast rather than on an absolute level — a faint formula on a
    /// grey ground is trimmed just as a black one on white is.
    /// </para>
    /// </remarks>
    internal static RgbImage CropMargin(RgbImage image)
    {
        int count = image.Width * image.Height;
        byte[] grey = ArrayPool<byte>.Shared.Rent(count);

        try
        {
            int minimum = 255;
            int maximum = 0;

            for (int y = 0; y < image.Height; y++)
            {
                ReadOnlySpan<byte> row = image.Row(y);
                int at = y * image.Width;

                for (int x = 0; x < image.Width; x++)
                {
                    int offset = x * 3;

                    // OpenCV's fixed-point BGR-to-grey, applied — as upstream applies it — to a
                    // buffer that is actually RGB, so red carries the blue weight and blue the
                    // red one. Reproduced rather than repaired: it decides which pixels the
                    // threshold keeps, and the model is fed whatever it decides.
                    int value = ((row[offset] * 1868)
                        + (row[offset + 1] * 9617)
                        + (row[offset + 2] * 4899)
                        + 8192) >> 14;

                    grey[at + x] = (byte)value;
                    minimum = Math.Min(minimum, value);
                    maximum = Math.Max(maximum, value);
                }
            }

            if (maximum == minimum)
            {
                return image.Clone();
            }

            Span<byte> stretched = stackalloc byte[256];
            double range = maximum - minimum;
            for (int value = minimum; value <= maximum; value++)
            {
                stretched[value] = (byte)(int)((value - minimum) / range * 255d);
            }

            int left = image.Width;
            int right = -1;
            int top = image.Height;
            int bottom = -1;

            for (int y = 0; y < image.Height; y++)
            {
                int at = y * image.Width;
                for (int x = 0; x < image.Width; x++)
                {
                    // `THRESH_BINARY_INV` keeps everything at or below the threshold, so the
                    // comparison is strictly greater rather than the other way round.
                    if (stretched[grey[at + x]] > 200)
                    {
                        continue;
                    }

                    left = Math.Min(left, x);
                    right = Math.Max(right, x);
                    top = Math.Min(top, y);
                    bottom = Math.Max(bottom, y);
                }
            }

            if (right < 0 || right - left + 1 <= 2 || bottom - top + 1 <= 2)
            {
                return image.Clone();
            }

            return image.Crop(left, top, right + 1, bottom + 1);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(grey);
        }
    }

    private static LayoutBox WholePage(RgbImage page, string label) =>
        new(-1, label, 1f, 0f, 0f, page.Width, page.Height, 0);

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsModel)
        {
            _model.Dispose();
        }

        _layout?.Dispose();
        _orientation?.Dispose();
        _unwarper?.Dispose();
    }
}

/// <summary>Progress of a page parse.</summary>
/// <param name="BlockIndex">Index of the block just recognised.</param>
/// <param name="BlockCount">Total number of blocks on the page.</param>
/// <param name="Label">Layout label of the block.</param>
public readonly record struct BlockProgress(int BlockIndex, int BlockCount, string Label);
