# PaddleOCR for .NET

**A pure C# port of the [PaddleOCR-VL-1.6](https://github.com/PaddlePaddle/PaddleOCR) document
parsing pipeline.** It turns a PDF or an image into markdown and JSON — text, headings, tables,
formulas, figures and reading order — on the CPU, with no ONNX Runtime, no Paddle Inference and
no libtorch.

[![NuGet](https://img.shields.io/nuget/v/PaddleOCR.svg?label=PaddleOCR)](https://www.nuget.org/packages/PaddleOCR)
[![NuGet](https://img.shields.io/nuget/v/PaddleOCR.Pdf.svg?label=PaddleOCR.Pdf)](https://www.nuget.org/packages/PaddleOCR.Pdf)
[![NuGet](https://img.shields.io/nuget/v/PaddleOCR.Cli.svg?label=paddleocr-sharp)](https://www.nuget.org/packages/PaddleOCR.Cli)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)

The tensor math, the model graphs, the tokenizer, the image pipeline and the weight loading are
all in this repository, under [`dotnet/`](dotnet/). The upstream Python sources are still here
too — they are the normative specification the port is checked against, line by line.

## Packages

| Package | What it is |
| --- | --- |
| [`PaddleOCR`](https://www.nuget.org/packages/PaddleOCR) | The library: tensors and SIMD kernels, the SigLIP vision tower, the ERNIE-4.5 decoder, the BPE tokenizer, the Paddle graph interpreter, the pipeline |
| [`PaddleOCR.Pdf`](https://www.nuget.org/packages/PaddleOCR.Pdf) | PDF page rasterisation — the one native dependency (PDFium) |
| [`PaddleOCR.Cli`](https://www.nuget.org/packages/PaddleOCR.Cli) | The `paddleocr-sharp` .NET tool |

```bash
dotnet add package PaddleOCR
dotnet add package PaddleOCR.Pdf     # only if you feed it PDFs
dotnet tool install --global PaddleOCR.Cli
```

The assemblies and the namespaces are `PaddleOcrSharp`, which is what the port is called in the
tree; the packages are the ones above. `net11.0` only — the port leans on
`System.Numerics.Tensors`, `Vector512<T>` and `TensorPrimitives`.

## The command-line tool

```bash
paddleocr-sharp download                              # fetch the checkpoints into the cache
paddleocr-sharp parse report.pdf --output-dir out/    # <name>.md, <name>.json and imgs/ per page
paddleocr-sharp parse page.png --format json          # to stdout instead
paddleocr-sharp recognize crop.png --prompt-label table
paddleocr-sharp bench                                 # measure the machine, then the stages
```

`parse` runs the shipped pipeline end to end:

1. **Layout detection** — PP-DocLayoutV3 finds the regions of a page and orders them.
2. **Block preparation** — overlapping detections are filtered against their mask outlines,
   paragraphs split across columns or around figures are stacked back into one image, and figures
   are extracted as images.
3. **Recognition** — each region goes to the 0.9B vision-language model with the instruction its
   label calls for (`OCR:`, `Table Recognition:`, `Formula Recognition:`, …).
4. **Assembly** — OTSL tables become HTML, LaTeX delimiters are normalised, runaway repetition is
   trimmed, and the page is rendered as markdown and JSON.

Models are fetched on demand into `~/.cache/paddleocr-sharp`; `--model-dir` and `--layout-dir`
point the loaders at directories you manage yourself, and `PADDLEOCR_SHARP_MODELS_URL` redirects
the downloader at another mirror. `paddleocr-sharp` with no arguments prints every option.

## As a library

```csharp
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Pipeline;

using var parser = DocumentParser.Load(visionLanguageDirectory, layoutDirectory);
using RgbImage page = ImageIO.Load("page.png");

ParsedPage parsed = parser.Parse(page);
Console.WriteLine(parsed.ToMarkdown());

foreach (ParsedBlock block in parsed.Blocks)
{
    Console.WriteLine($"{block.Label} {block.Box.Left},{block.Box.Top} -> {block.Content}");
}
```

A whole document, so that the stages needing every page can run:

```csharp
using PaddleOcrSharp.Pdf;

var pages = new List<ParsedPage>();
foreach (RgbImage page in PdfRasterizer.Render("report.pdf", dpi: 200))
{
    using (page) pages.Add(parser.Parse(page, pageIndex: pages.Count));
}

string markdown = new ParsedDocument(pages)
    .MergeTablesAcrossPages()   // rejoin a table a page break split in two
    .AssignTitleLevels()        // decide how deep each heading sits
    .ToMarkdown();
```

To recognise a single already-cropped region, skip the pipeline:

```csharp
using PaddleOcrSharp.Models;

using var model = PaddleOcrVLModel.Load(visionLanguageDirectory);
string text = model.Recognize(crop, BlockPrompt.Table);
```

Fetching the checkpoints from code:

```csharp
using PaddleOcrSharp.Download;

using var downloader = new ModelDownloader();
string vl = await downloader.EnsureAsync(ModelCatalog.PaddleOcrVL16);
string layout = await downloader.EnsureAsync(ModelCatalog.PpDocLayoutV3);
```

## What is inside

**`PaddleOCR-VL-1.6-0.9B` = NaViT-style SigLIP vision tower → 2×2 patch-merging projector →
ERNIE-4.5 decoder**, with **PP-DocLayoutV3** in front of it.

- **Vision tower** — 27 pre-norm encoder layers, 16 heads of 72 dims, native-resolution
  patchify with `smart_resize`, the pretrained 27×27 position grid bilinearly interpolated to the
  image's own grid, 2-D RoPE inside attention, attention block-diagonal per image.
- **Decoder** — 18 layers, hidden 1024, GQA ×8, SwiGLU, RMSNorm, and 3-D M-RoPE whose `t/h/w`
  position ids come from the image grid.
- **Layout** — an RT-DETR–style detector: HGNetV2-L backbone, hybrid encoder, 6-layer deformable
  decoder, 300 queries, 25 classes, plus mask and reading-order heads. Its `[N, 200, 200]` masks
  are not decoration: each is reduced to a polygon that decides whether two overlapping boxes are
  really the same region, and whites out everything outside the region in the block's crop. That
  needs `findContours`, `approxPolyDP`, `minAreaRect`, `fillPoly` and polygon intersection, all
  ported and checked against OpenCV and Shapely themselves.

The layout detector ships only as a Paddle inference graph — there is no PyTorch module tree to
port from — so the port interprets the exported graph directly, with ~65 of its own operator
kernels. The same interpreter runs UVDoc and the orientation classifier.

Weights stay in their on-disk dtype: a 0.9B model does not get a 4 GB float32 shadow copy. The
GEMM widens one bf16 column panel at a time and reuses it across every activation row.

Roughly half the port is not model code at all — it is the stages that decide what the models are
asked and what becomes of their answers: overlap filtering, block merging, formula crop
contrast and margin trimming, figure placeholders inside tables, repetition truncation, OTSL to
HTML, spotting coordinates to polygons, reading-order numbering, heading levels, cross-page table
merging, and the markdown writer.

## Correctness

Every stage is checked against the upstream Python rather than against expectations:

| Stage | How it is checked |
| --- | --- |
| Image pre-processing | byte-identical resize and `pixel_values` matching to 1e-6, over six image shapes |
| Vision tower | all 27 encoder layers plus the projector, against per-layer dumps |
| Tokenizer | encode and decode over 27 mixed-script cases, including byte fallback and the `<\|LOC_n\|>` tokens |
| Decoder | prompt ids, 3-D rope index, prefill logits and 24 greedy steps, token for token |
| Layout graph | every fetched tensor against Paddle's own inference run |
| Contours and polygons | the OpenCV and Shapely calls behind the mask head, against cv2 and Shapely themselves |
| Orientation classifier, UVDoc | both graphs against Paddle's own run, and both wrappers end to end |
| Markdown | the rendered page against PaddleX's own `MarkdownConverter`, over every label and six settings combinations |
| Cross-page tables | merge decisions and merged HTML against `merge_table.py`, over eleven page pairs |
| Heading levels | numbering styles, text heights and the final levels against `title_level.py` |
| Whole pipeline | block labels, boxes and recognised text for a rendered page |

```bash
cd dotnet && dotnet test
```

Tests that need the checkpoints or the generated `.npz` fixtures skip themselves when those are
absent, so a clean clone goes green. See
[`dotnet/tools/reference/README.md`](dotnet/tools/reference/README.md) for generating them.

## Performance

The port is faster than the pipeline it was ported from. Upstream's own `PaddleOCRVL16Pipeline`
and this library over the same three pages, back to back on the same machine, at each side's
defaults, checkpoint loading excluded on both sides:

| Page | Python | C# | |
| --- | --- | --- | --- |
| report (4 blocks) | 62.1 s | 33.6 s | 1.85x |
| benchmark (3 blocks, one table) | 49.7 s | 23.2 s | 2.14x |
| lines (1 block) | 20.1 s | 11.5 s | 1.75x |
| **total** | **131.9 s** | **68.3 s** | **1.93x** |

The markdown is byte-identical on all three, so this is the same work done in less time rather
than less work. `--block-concurrency 4` takes the total to about 59 s by recognising blocks in
parallel too, at the cost of holding several blocks' activations at once; it is not the default.

`paddleocr-sharp bench` measures the machine before it loads anything — the FMA rate at each
vector width, at one thread and at all of them, and the read bandwidth at each level of the
hierarchy — because stage times from a shared host are not comparable between runs without it.
It then reports per-stage timings, allocation, and a per-operator breakdown of the layout graph
including each operator's slowest single call with its result shape and Paddle module path.

Every tensor in the hot loops is pooled: a whole vision pass allocates about 1.2 MiB and a decode
step about 310 KiB, all of it framework bookkeeping for the parallel loops, none of it tensors.

## Native AOT

The CLI publishes as a self-contained native binary, no runtime installed:

```bash
cd dotnet
dotnet publish src/PaddleOcrSharp.Cli -c Release -r linux-x64 -p:PublishAot=true
```

Both libraries set `IsAotCompatible`, so the trim and AOT analysers run on every build and a
reflection-based API cannot creep in unnoticed.

## Repository layout

```
dotnet/
  src/PaddleOcrSharp/          Core/ Formats/ Imaging/ Text/ Models/ Pipeline/ Download/
  src/PaddleOcrSharp.Pdf/      PDF page rasterisation
  src/PaddleOcrSharp.Cli/      paddleocr-sharp
  tests/PaddleOcrSharp.Tests/  unit + numerical-parity tests
  tools/reference/             Python scripts that dump upstream reference tensors
.devops/build-nuget.yml        builds, tests and publishes the three packages
```

Everything else in the tree is the upstream PaddleOCR Python project, kept as the reference the
port is validated against. It is not modified by the port; its CI, pre-commit hooks and agent
skill files are disabled (`*.disabled`) for the duration.

## Scope

In scope: the PaddleOCR-VL **1.6** pipeline (`PaddleOCR-VL-1.6-0.9B` + `PP-DocLayoutV3`), document
pre-processing, layout detection, block-level recognition, and markdown/JSON assembly, on the CPU.

Out of scope: the legacy PP-OCRv3/v4/v5 detector+recognizer, PP-StructureV3, PP-ChatOCR, training
and fine-tuning, and the vLLM / SGLang / FastDeploy back-ends. GPU is a possible future backend.

## Licence

Apache-2.0, matching the upstream project and the model weights. See [LICENSE](LICENSE).

## Citation

The models this port runs are described in:

```bibtex
@misc{zhang2026paddleocrvl16expandingfrontierdocument,
      title={PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training},
      author={Zelun Zhang and Hongen Liu and Suyin Liang and Yubo Zhang and Yiqing Xiang and Jiaxuan Liu and Ting Sun and Manhui Lin and Yue Zhang and Changda Zhou and Tingquan Gao and Cheng Cui and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2026},
      eprint={2606.03264},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.03264},
}

@misc{cui2025paddleocrvlboostingmultilingualdocument,
      title={PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model},
      author={Cheng Cui and Ting Sun and Suyin Liang and Tingquan Gao and Zelun Zhang and Jiaxuan Liu and Xueqing Wang and Changda Zhou and Hongen Liu and Manhui Lin and Yue Zhang and Yubo Zhang and Handong Zheng and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2510.14528},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14528},
}
```

The original English PaddleOCR README is kept at [`readme/README_upstream.md`](readme/README_upstream.md).
