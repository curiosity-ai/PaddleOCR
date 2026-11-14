# PaddleOCR-VL — Pure C# Port (`dotnet/`)

This repository hosts, in addition to the upstream Python PaddleOCR sources, a **pure C#
re-implementation of the PaddleOCR-VL-1.6 document-parsing pipeline** under [`dotnet/`](dotnet/).

The C# port is self-contained: it owns the tensor math, the model graph, the tokenizer, the
image pipeline and the weight loading. **It does not use ONNX Runtime, Paddle Inference,
libtorch, or any other third-party inference engine.**

---

## Scope

| In scope | Out of scope |
| --- | --- |
| PaddleOCR-VL **1.6** pipeline (`PaddleOCR-VL-1.6-0.9B` + `PP-DocLayoutV3`) | PP-OCRv3/v4/v5 detector+recognizer (legacy models) |
| Doc pre-processing (orientation classify, unwarping) | Training / fine-tuning |
| Layout detection, block cropping, block-level VL recognition | PP-StructureV3, PP-ChatOCR, table-cell detectors |
| Markdown / JSON assembly, OTSL→HTML table conversion | Serving (HPS/Triton), distributed inference |
| CPU inference (SIMD-accelerated); GPU is a possible future backend | vLLM / SGLang / FastDeploy back-ends |

## Upstream references used

The port is validated line-by-line against these upstream sources. They are the **normative
specification** — when C# and Python disagree, Python is right.

| Component | Upstream reference |
| --- | --- |
| VL model graph | `PaddlePaddle/PaddleOCR-VL-1.6` on Hugging Face — `modeling_paddleocr_vl.py`, `configuration_paddleocr_vl.py` |
| Image pre-processing | same repo — `image_processing_paddleocr_vl.py` (`smart_resize`, patchify) |
| Prompt assembly | same repo — `processing_paddleocr_vl.py`, `chat_template.jinja` |
| Pipeline orchestration | `PaddlePaddle/PaddleX` — `paddlex/inference/pipelines/paddleocr_vl/{pipeline,uilts,result}.py` |
| Layout model | `PaddlePaddle/PP-DocLayoutV3` on Hugging Face; config in `paddlex/modules/base/utils/pdparams2safetensors/model_config.py` |
| Pipeline defaults | `deploy/paddleocr_vl_docker/pipeline_config_vllm.yaml` in this repo |

A read-only checkout of PaddleX and the Hugging Face model metadata is expected under
`/home/user/ref/` during development (see `dotnet/tools/reference/README.md`).

---

## Model architecture (as ported)

**`PaddleOCR-VL-1.6-0.9B` = NaViT-style SigLIP vision tower → 2×2 patch-merging projector → ERNIE-4.5 decoder.**

### Vision tower (`PaddleOCRVisionModel`, 27 layers)
- Native-resolution patchify: `smart_resize` to a multiple of `patch*merge = 28`,
  `min_pixels = 112_896`, `max_pixels = 1_003_520`, bicubic, `mean = std = 0.5`.
- `Conv2d(3, 1152, k=14, s=14)` per patch (equivalent to a `588 → 1152` GEMM).
- Absolute position embedding: the pretrained `27×27` grid is **bilinearly interpolated**
  (`align_corners=false`) to the image's `(grid_h, grid_w)` — `interpolate_pos_encoding=True`.
- 27 × pre-norm encoder layers: `LayerNorm(eps=1e-6)` → MHA(16 heads, head_dim 72) → `LayerNorm` → MLP(1152→4304→1152, `gelu_tanh`).
- 2-D RoPE inside attention: `SigLIPRotaryEmbedding(dim = head_dim/2 = 36, theta = 10000)`,
  frequencies indexed by `(row, col)`, concatenated then duplicated to 72 and applied
  `rotate_half`-style. Window attention is **disabled** (`window_size = -1`).
- Attention is **block-diagonal per image** (matches `flash_attn_varlen` with `cu_seqlens`).
- `post_layernorm`; the pooling head is **not** used at inference (`return_pooler_output=False`).

### Projector (`mlp_AR`)
`LayerNorm(1152, eps=1e-5)` → merge 2×2 spatial patches into one token (4608 dims) →
`Linear(4608, 4608)` → exact-erf `GELU` → `Linear(4608, 1024)`.

### Language model (`Ernie4_5Model`, 18 layers)
- `hidden 1024`, `heads 16`, `head_dim 128`, `kv_heads 2` (GQA ×8), `intermediate 3072`, SwiGLU.
- `RMSNorm(eps=1e-5)`, no biases, untied `lm_head`.
- **3-D M-RoPE**: `theta = 500_000`, `mrope_section = [16, 24, 24]`; `t/h/w` position ids come
  from `get_rope_index` — image tokens get a `(t, h, w)` grid, text tokens get `max+1` onward.
- Causal attention with a KV cache; greedy decoding by default.

### Layout (`PP-DocLayoutV3`)
An RT-DETR–style detector: HGNetV2-L backbone → hybrid encoder (`d_model 256`, 3 levels,
strides 8/16/32) → 6-layer deformable decoder, 300 queries, 25 classes, plus mask and
reading-order heads. Input `800×800`, `1/255` rescale, no mean/std, `NCHW`.

The graph has three fetches: boxes `[N, 7]`, a box count, and a `[N, 200, 200]` mask per query.
The masks are not decoration — upstream's default `layout_shape_mode = "auto"` reduces each to a
polygon and uses it both to decide whether two overlapping boxes are really the same region and
to white out everything outside the region in the block's crop. A slanted scan or an L-shaped
column is where that matters. Reproducing it needs `findContours`, `approxPolyDP`, `minAreaRect`,
`fillPoly` and Shapely's polygon intersection, all of which are ported in `Imaging/Contours.cs`
and `Imaging/Polygons.cs` and checked against the originals.

**This model is not hand-ported.** It ships only as a Paddle inference graph — there is no
upstream PyTorch module tree to port from — so `Models/Paddle` interprets the exported graph
directly. Every operator in that interpreter is our own kernel; what we take from Paddle is the
graph topology and the weights, both of which we would have to take anyway. The result is exact
by construction instead of by inspection, and the same interpreter also runs UVDoc and the
orientation classifier.

The interpreter covers ~65 operators: `conv2d` / `depthwise_conv2d` (im2col + GEMM, with a
direct path for depthwise), pooling, batch norm, layer norm, softmax, `matmul` / `bmm` / `einsum`,
`grid_sample` (which is what deformable attention is built on), bilinear and nearest
interpolation, `pad3d`, broadcasting element-wise ops and activations (`relu`, `hardswish`,
`hardsigmoid`, `prelu`, …), reductions, `top_k`, `argsort`, `gather_nd`, `index_put`,
`set_value` and the shape algebra.

Two of Paddle's inference-time conventions are easy to get wrong and are worth naming, because
both produce plausible-looking output rather than an error. `dropout` is not the identity at
inference when its `mode` is `downgrade_in_infer` — it scales by `1 - p`, and `p` arrives as an
input rather than an attribute. And the interpolation operators take their target size three
ways: an `OutSize` tensor, a `SizeTensor` *list* of rank-0 tensors (which is how a size read
from another tensor's shape at run time arrives), or a scale; UVDoc uses the list form.

---

## Project layout

```
dotnet/
  PaddleOcrSharp.slnx
  Directory.Build.props        # net11.0, unsafe, nullable, preview lang, package metadata
  Directory.Build.targets      # attaches the repository README to the packable projects
  src/
    PaddleOcrSharp/            # the library — everything below is here
      Core/                    # Tensor<T>, pooled buffers, SIMD kernels, GEMM
      Formats/                 # safetensors + paddle .pdiparams readers, bf16/f16
      Imaging/                 # SkiaSharp decode, smart_resize, normalize, patchify
      Text/                    # tokenizer (tokenizer.json BPE), chat template
      Models/Vision/           # SigLIP/NaViT encoder + projector
      Models/Language/         # ERNIE-4.5 decoder, KV cache, sampling
      Models/Layout/           # PP-DocLayoutV3 wrapper and detection post-processing
      Models/Paddle/           # Paddle PIR graph interpreter and its operator kernels
      Models/Preprocessing/    # orientation classifier, UVDoc unwarping
      Pipeline/                # orchestration, block prompts, markdown/OTSL assembly
      Download/                # model downloader (static HTTP mirror)
    PaddleOcrSharp.Pdf/        # PDF page rasterisation (the one native dependency)
    PaddleOcrSharp.Cli/        # `paddleocr-sharp` command-line front-end
  tests/
    PaddleOcrSharp.Tests/      # unit + numerical-parity tests
  tools/
    reference/                 # Python scripts that dump upstream reference tensors
```

## Two bicubic resizes, deliberately

The two model families reach C# through different Python image stacks, and their bicubic
resamplers disagree:

| | `PilResize` | `OpenCvResize` |
| --- | --- | --- |
| Used by | the VL model's `smart_resize` | the layout detector's 800×800 input |
| Mirrors | `PIL.Image.resize(BICUBIC)` | `cv2.resize(INTER_CUBIC)` |
| Kernel `a` | −0.5 | −0.75 |
| Downscale | support scaled by `in/out` (antialiased) | fixed support (aliased) |
| Border | kernel truncated | replicated |
| Arithmetic | Q22 fixed point, two uint8 passes | float32 |
| Parity | byte-exact | ≤1 level on ~0.02% of bytes |

SkiaSharp's resampler matches neither and is used only for decoding and encoding.

## Engineering conventions

- **.NET 11 / C# preview.** Use `System.Numerics.Tensors`, `Vector<T>` / `Vector512<T>`,
  `TensorPrimitives`, `ArrayPool<T>` and `MemoryPool<T>`. Hot loops must be allocation-free —
  and "allocation-free" includes the compiler's own: a method that contains a lambda *anywhere*
  allocates its display class on entry, whichever branch runs, so a hot method with a parallel
  and a serial path must keep the lambda in a separate method. That one detail was 437 KiB per
  vision attention layer.
- **Pool through `ArrayPool<T>.Shared`.** It pools buffers up to a gigabyte — the widely repeated
  1 MiB bucket cap is .NET Framework folklore — and it keeps every buffer of a size where
  `ArrayPool.Create` drops everything past `maxArraysPerBucket`. `TensorPool` is the one place that
  policy is stated. Measure the shared pool before adding another.
- **Weights stay in their on-disk dtype** (bf16) so a 0.9B model does not need a 4 GB float32
  shadow copy. The GEMM widens one column panel at a time and reuses it across every activation
  row; widening inside the inner loop instead costs more than the multiply-adds it feeds.
- **Every numerical stage is testable in isolation.** Each module exposes a deterministic entry
  point that the parity tests feed with `.npz` fixtures dumped from the Python reference. That is
  why the public surface is wider than a black-box library's would be; what is genuinely internal
  to one component — the Paddle operator kernels behind the graph interpreter — is `internal`.
  Public members are documented, and `GenerateDocumentationFile` makes a gap a build warning.
- **No `float` accumulation shortcuts** where upstream forces `float32` (softmax, RoPE,
  RMSNorm variance) — match upstream precision decisions exactly.
- **Stay native-AOT publishable.** Both libraries set `IsAotCompatible`, so the trim and AOT
  analysers run on every build; JSON goes through a source-generated context, not reflection.

## Validating against upstream

`dotnet/tools/reference/` holds Python scripts that load the real Hugging Face model, run a
stage, and dump inputs/outputs as `.npz`. `PaddleOcrSharp.Tests` loads the same `.npz` and
asserts the C# output matches within a stage-appropriate tolerance. Fixtures are generated on
demand (they are not committed) — see `dotnet/tools/reference/README.md`.

Tests that need fixtures are skipped, not failed, when the fixture directory is absent, so
`dotnet test` works on a clean clone.

## Where the time goes

> **How to measure, as opposed to what was measured, is a skill:**
> [`.claude/skills/measuring-performance`](.claude/skills/measuring-performance/SKILL.md).
> Read it before benchmarking, profiling, or accepting or rejecting a performance change. It
> covers calibrating the machine first, making sure the code under test is compiled, best-of-N
> with the spread, interleaved A/B runs against a control, the microbenchmark traps that report a
> fifth of the truth, and how to pin an output. Everything in this section is the *record* those
> rules produced.

`paddleocr-sharp bench` starts by measuring the machine, before it loads anything: the FMA rate
the hardware sustains at each vector width, at one thread and at every thread, and the read
bandwidth at each level of the hierarchy. A shared virtual machine is not a constant, so absolute
milliseconds from different days are not comparable: the run prints its own ceilings and the
per-sample spread beside every figure, and `--gemm true` reports each GEMM shape as a fraction of
the ceiling just measured. On the reference machine that spread runs from 10% to 40%, which is
wider than most differences worth acting on: nothing below was accepted without an A/B in one
sitting, with an untouched operator alongside it as a control.

Measured on 4 cores at ~3 GHz, a 980x392 page (1960 patches).

| Stage | Cost |
| --- | --- |
| Vision tower (1960 patches) | ~16 s |
| Decoder (503-token prefill + 32 tokens) | ~3.6 s |
| Layout graph | ~7.6 s |

### A whole page, including the stages no profile covered

`PageProfile` (printed by `parse --profile` beside the block table) names the rest: rasterise,
decode, layout, crop, stack, table-figures, prepare, recognize, encode-figure, otsl-to-html,
assemble, render. It exists because those three profiles between them left a hole — on a short
scanned page `parse --profile` reported 6.2 s of recognition inside a 20.7 s page and said nothing
about the other 14.5 s. `--layout-profile` adds the layout graph's own operator breakdown.

Over six pages of `test_documents/pdf_scanned` and `images/ocr_*`, one page each:

| Stage | Share of the corpus |
| --- | --- |
| vision tower | 67.5% |
| decode | 16.0% |
| layout graph | 8.3% |
| prefill | 8.1% |
| everything else | 0.2% |

**Scanned documents invert the picture the rest of this section was written against.** There, two
runaway decodes were 95% of a page; here the tower is two thirds of the corpus and decode is a
sixth. Dense scanned text gives every block a near-maximal patch count and almost nothing to say:
one block reached 4,840 patches and generated 258 tokens. The consequence for tuning is that
attention's share is not a constant — it is quadratic in patches where every other stage is
linear, so 36% at 1960 patches is 65% at 4900.

The same table settles two smaller questions. The pipeline's own image work — cropping, masking,
stacking, the markup conversions — is 0.2% of a page and is not worth attention. And one layout
detection allocated 4.5 GiB, which turned out to be the largest single thing wrong with the port;
see below.

### Against the original

The port is faster than the pipeline it was ported from. Running upstream's own
`PaddleOCRVL16Pipeline` and this library over the same three pages, back to back on the same
machine, at each side's defaults:

| Page | Python | C# | |
| --- | --- | --- | --- |
| report (4 blocks) | 62.1 s | 33.6 s | 1.85x |
| benchmark (3 blocks, one table) | 49.7 s | 23.2 s | 2.14x |
| lines (1 block) | 20.1 s | 11.5 s | 1.75x |
| **total** | **131.9 s** | **68.3 s** | **1.93x** |

The markdown is byte-identical on all three, so this is the same work done in less time rather
than less work. Both sides use the whole machine: upstream threads inside oneDNN and Paddle, and
this port inside its own GEMM. `--block-concurrency 4` takes the total to about 59 s by
recognising blocks in parallel as well, at the cost of holding several blocks' activations at
once; it is not the default.

Two caveats on that table. Loading the checkpoint is excluded on both sides — cold, the first
memory-mapped read of the 0.9B weights costs around 97 s here and is entirely page-cache traffic,
which is what an earlier and much less flattering comparison had accidentally folded into the C#
column. And the machine moves: the same Python run measured 123.5 s a few days earlier against
131.9 s here, which is why both sides are always measured in one sitting.

#### Measured again, stage by stage, against a genuine PaddleX v1.6 install

That table is end-to-end only, and end-to-end hides that the two halves of the pipeline go in
opposite directions. Re-measured on a different machine (4 cores, Xeon @ 2.10 GHz) against
`paddleocr.PaddleOCRVL(pipeline_version="v1.6", vl_rec_backend="native")` — the real wrapper over
the real PaddleX pipeline, pointed at the same two model directories the port loads, everything
else at upstream's defaults. One page per process pair, upstream then port back to back, so
machine drift stays inside a pair; model loading excluded on both sides.

| Page | upstream | port | |
| --- | --- | --- | --- |
| `ocr_test_original.png` (1 block, 672 patches) | 16.0 s | 13.8 s | 1.16x |
| `nougat_004_scanned.pdf` (6 blocks, 4.7k patches) | 83.8 s | 39.6 s | 2.12x |
| `code_and_formula` p1 (8 blocks, 18.5k patches) | 1016.6 s | 223.3 s | 4.55x |
| `ocr_image.jpg` (5 blocks, one table) | did not finish in 90 min | 74.4 s | — |

The markdown is identical on all three that finished, to the trailing newline the port adds and
upstream does not, so this is again the same work in less time. `ocr_image.jpg` was given a
90-minute budget twice and produced no page either time; it is reported as it was measured, not
diagnosed.

The ratio's spread across that table is not noise, and splitting the page says where it comes
from. Upstream's own two stages were timed by wrapping `layout_det_model.apply` and
`vl_rec_model.predict` in place (`__call__` on a PaddleX predictor delegates to `apply`, and a
dunder is looked up on the type, so `apply` is the hook that sees the work); the port's come from
`parse --profile`. As first measured:

| Page | layout: upstream | layout: port | VL: upstream | VL: port |
| --- | --- | --- | --- | --- |
| `ocr_test_original.png` | 2.0 s | 6.3 s | 11.5 s | 7.5 s |
| `nougat_004_scanned.pdf` | 1.7 s | 5.9 s | 73.3 s | 33.6 s |

**The port won the VL half by 1.5-2.2x and lost the layout half by about 3x.** Everything in the
end-to-end table follows from those two numbers: the layout deficit was a near-constant ~4 s, so it
was 30% of a trivial page and 2% of a heavy one, and the VL win is proportional to the work — which
is why a one-block page came out at 1.16x and an eight-block page at 4.55x.

Two structural differences behind the VL half, both worth stating because both invite the wrong
conclusion:

- **Upstream runs the VL model in fp32 on CPU.** `is_bfloat16_available` excludes CPU, so
  PaddleX's `doc_vlm` predictor casts the bf16 checkpoint up at load. Decoding is bandwidth-bound
  on weight streaming, so upstream re-reads 4 bytes per parameter per token where the port reads
  2. That kills the idea of moving the port to fp32 activations for parity — it would be a
  divergence from upstream's *numbers* in the name of matching its *dtype*, and it would cost
  roughly the whole decode win.
- **Loading.** Upstream builds its pipeline in 22-57 s at ~9 GB RSS (the fp32 shadow copy above);
  the port memory-maps the bf16 weights in 0.5-1.0 s. Excluded from every figure here, but it is
  the difference between a pipeline you can start per page and one you cannot.

And one that was simply a mistake on this side: **upstream renders PDF pages at 144 dpi**
(`PDFReader(zoom=2.0)` over the natural 72), where the port defaulted to 200. At 200 a page
carries 1.93x the pixels and so 1.93x the patches, so the port had been doing appreciably more
work than upstream on every PDF in the comparison and being timed against it anyway. The default
is now 144; `nougat_004_scanned.pdf` went from 50.6 s to 39.6 s on the change alone, and its
output matches upstream's at the matched resolution.

#### Closing the layout gap

That deficit is gone. Measured with both sides inside one shell invocation, so neither can be
attributed to a different machine than the other — upstream's three pages in one process, the
port's one process per page, as a user runs it:

| Page | upstream | port | |
| --- | --- | --- | --- |
| `ocr_test_original.png` | 16.6 s | 8.8 s | **1.89x** |
| `nougat_004_scanned.pdf` | 74.0 s | 37.3 s | **1.98x** |
| `code_and_formula_scanned.pdf` p1 | 1250.7 s | 165.6 s | **7.55x** |

| Page | layout: upstream | layout: port | VL: upstream | VL: port |
| --- | --- | --- | --- | --- |
| `ocr_test_original.png` | 3.0 s | 3.4 s | 13.5 s | 5.3 s |
| `nougat_004_scanned.pdf` | 2.2 s | 3.5 s | 71.7 s | 33.7 s |
| `code_and_formula_scanned.pdf` p1 | 4.0 s | 3.5 s | 1246.4 s | 162.1 s |

The port's layout column is a cold process every time; upstream's is warm after its first page, so
2.2 s is its steady state and 3.0 s its own cold one. Against the cold figure the port is at
parity; against the steady one it is within about 1.6x, from 3x. A warm detection in `bench` is
2.66-2.72 s against the 3.8 s it was.

Six changes did it, each A/B'd interleaved in one sitting, and the output is byte-identical to the
pre-optimisation build on all four test pages:

- **The convolutions reached the wrong GEMM.** `conv2d` is 157 GFLOP of a detection and ran at
  81 GFLOP/s, against the 250 the same machine sustains through `Gemm.Linear` on the vision
  tower's shapes. `Gemm.MatMul`'s `k × n` form accumulates whole output rows in memory — one load
  and one store per multiply-add — where `Linear` widens a column panel once and reduces four rows
  against four columns in registers. The convolution is now posed as the shape `Linear` wants: the
  filters as the activation rows and the im2col columns as the weight panel, which also lands the
  product in `[channel, pixel]` order, so it copies out rather than transposing. im2col runs over
  a block of output rows sized by a column budget — at 4 MB the GEMM measured 3767 ms over three
  detections against 2884 at 16 MB, and 32 MB was no better. **-10% of the stage.**
- **Almost nothing outside `conv2d` was threaded.** The element-wise kernels, the casts, the
  broadcast walks, the transposes and `where` all ran on one core while the convolutions used
  four. `Parallelism.Chunked` is now the one place that split is expressed, and the walks that
  needed to start somewhere other than zero get their counters from `Broadcast.Seed`.
  **-31% of the stage**, and the largest single item in this list.
- **`batch_norm_` was a scalar loop** with two freshly allocated per-channel arrays, and **`any`
  over a contiguous suffix** went through the generic reduction's per-element index arithmetic and
  `double` conversion. 156 -> 34 ms and 68 ms -> off the profile.
- **The depthwise convolution ran at 3.7 GFLOP/s**, bounds-checking each of a 5×5 window's
  twenty-five taps per output pixel. Its stride-1 interior now accumulates a vector of output
  columns in a register across the window: 149 -> 41 ms.
- **`Gemm.MatMul`'s direct tile is register-blocked** — four output rows by two vectors of columns
  held in registers for the whole reduction — which is the same store-port bound the vision
  tower's value product had. The deformable attention is where that lands: matmul 221 -> 178 ms.
  `Gemm.Linear` also stopped widening a float32 panel into scratch, which for float32 is a copy
  and nothing else, and the convolution hands it 2.1 GB of im2col columns a detection.
- **`einsum` rebuilt every operand's offset from the counters for each of 23 million terms.**
  The summed axes are the tail of the label order, so each output element owns one contiguous run
  of the walk: offsets are carried forward and the walk splits by output element. 88 -> 45 ms.

And one that is not a kernel at all. **Every run of the tool is a cold process** — it parses a
document and exits, so the first pass through the layout graph and the vision tower runs as tier-0
code and nothing ever reaches the steady state tiered compilation is designed for. Compiling
optimised up front, on a page: layout 5061/4721 -> 3194/3262 ms, the whole page 10712/10036 ->
8149/8352, recognition 5623/5287 -> 4909/5044. It is set on the tool alone, so a long-lived host
embedding the library keeps tiering, and `DOTNET_TieredCompilation=1` overrides it.

Post-everything the graph is still `conv2d`-bound, and more so than before:

| Operator | Share of a 2.69 s detection |
| --- | --- |
| `conv2d` | **60.0%** |
| `matmul`, `add`, `transpose` (deformable attention) | 14.6% |
| `depthwise_conv2d`, `batch_norm_`, `concat`, `grid_sample` | 6.7% |
| the mask head's `cast`, `multiply`, `where`, `slice`, `full_like` | 7.8% |

What remains in `conv2d` splits roughly 65% GEMM, 30% im2col fill, 5% copy-out, with the GEMM at
about 180 GFLOP/s. Closing the rest of it means a GEMM that blocks the reduction as well as the
output — `Linear` sweeps its panel once per four activation rows, which is 0.5 bytes per flop from
L2 whatever the panel width — and that is a change to the kernel every other stage depends on.


Both model halves are GEMM-bound, and the shape of the win is the same in each: give the inner
loop enough reuse that it is compute-bound rather than load-bound. `Gemm.Linear` widens a bf16
column panel once and reuses it across every activation row; `Gemm.MatMul` tiles the output and
picks its kernel from the operand layout; attention blocks 16 query rows at a time so the keys and
values stay in cache across the block; the convolution treats an output row as one GEMM against
its im2col columns rather than a dot product per output pixel.

The layout graph is not GEMM-bound, and for a while it was not bound by anything defensible. The
mask head works on `[1, 300, 200, 200]` tensors — twelve million elements — and its casts,
comparisons and fills ran element at a time. Vectorising them took the stage from 9.3 s to 7.6 s
with `conv2d` flat as a control, and moved the convolutions from a fifth of the graph to a third.
What was actually left, though, was neither: the graph was allocation-bound, and pooling its
intermediates took it from 7.5 s to 4.7 s — see "Where the allocations go".

`PirProfile` (printed by `bench`) reports each operator's total, its slowest single call, and that
call's result shape and Paddle module path. The shape column is what makes the layout graph's cost
legible.

### A page's cost is not the sum of its blocks

`RecognitionProfile` (printed by `parse --profile`) is the VL half's answer to those two: one row
per block, giving the patches encoded, the prompt length, the **tokens generated**, the split
between vision, prefill and decode, the share of decode spent in the output head, and the bytes
allocated. The generated-token count is the column that matters, because a block's cost is set by
how much it says rather than by how big it is.

Decoding is bandwidth-bound on weight streaming, which is what makes that so. Every generated token
re-reads the 18 decoder layers (255M parameters) and the untied `lm_head` (106M), so **721 MB of
bf16 weights per token** before the key/value cache is counted — and the cache adds another 302 MB
per token once the context reaches 8k. The profile shows both halves of that directly: a block
decoding a few dozen tokens runs at ~40 ms/token with the output head taking 19-20% of decode,
while a block that reached 8k runs at ~85 ms/token with the head down to 9% — same weights, same
head, twice the cost, because the cache it re-reads has grown past the weights themselves. The head
is not worth attacking either way: greedy selection needs every logit, so its 212 MB is
irreducible.

The consequence is that a decoder which stops converging is not a quality problem with a
performance footnote, it is the performance problem. Measured on `equations.docx` page 1, fifteen
blocks, before the early stop below:

| | tokens | decode | allocated | share of page |
| --- | --- | --- | --- | --- |
| two runaway `text` blocks | 16,384 | 1,379 s | 5.1 GiB | 94.9% |
| the other thirteen blocks | 332 | 13 s | 0.26 GiB | 5.1% |

Both runaway blocks stopped only on the 8192-token budget, and every token past the first few
hundred was discarded by `RepetitionTruncator` immediately afterwards.

### Stopping a decode that has fallen into a cycle

`GenerationOptions.StopOnRepetition` (on by default; `--stop-on-repetition false` to disable) ends a
block once its token stream is `RepetitionRepeats` verbatim copies of one period of at most
`RepetitionMaximumPeriod` tokens, and only after `RepetitionMinimumTokens` have been generated.
Greedy decoding makes the test exact rather than statistical: there is no sampling noise to see
through, so a tail that has repeated six times is a cycle, not a coincidence.

The thresholds sit well past what `truncate_repetitive_content` needs to fire — it acts on five
repeats of an eight-character unit — so what is skipped would have been cut from the string anyway.
The 384-token floor is what keeps ordinary blocks out of the check: their whole output is shorter
than that, so they never reach it.

Measured over the nine-page benchmark corpus, at defaults:

| | before | after | |
| --- | --- | --- | --- |
| `equations.docx` p1 | 1,472 s | 123 s | 12.0x |
| that page's tokens | 16,716 | 1,100 | 15.2x |
| that page's allocations | 5.3 GiB | 445 MiB | 12.2x |
| whole corpus (9 pages) | 2,123 s | 595 s | 3.6x |

The page's three figures are one A/B in a single sitting, `--stop-on-repetition false` against the
default; the corpus row is the earlier nine-page sweep, so its page-1 number (1,615 s) is a
different day's measurement of the same run.

Once the runaway blocks are gone that page is fifteen small blocks and 48% vision, which is the
shape `--block-concurrency` was meant for: 123 s at the default 1, 95 s at 2, 84 s at 4, with
byte-identical output. It stays at 1 by default because the win is a property of this shape — a
page of few large blocks has no idle cores to fill — and the cost is holding several blocks'
activations at once.

#### Upstream has the same runaway, and no stop for it

The early stop is a deliberate divergence, so it is worth being precise that what it skips is not
something upstream keeps. Upstream cannot stop either, by construction:

- PaddleX's local backend builds `generate_kwargs` out of **`max_new_tokens` and nothing else**
  (`doc_vlm/predictor.py`); `repetition_penalty`, `temperature` and `top_p` are each warned about
  and dropped. The server backends send a temperature and a token cap. The default budget is
  `PADDLEOCR_VL_MAX_NEW_TOKENS = 8192`, which is where our own default comes from.
- The checkpoint's `generation_config.json` sets only `eos_token_id`, `pad_token_id` and
  `use_cache` — no `no_repeat_ngram_size`, no stopping criteria — and the model is a stock
  `GenerationMixin`. Greedy decoding there ends on the stop token or on the budget.
- `truncate_repetitive_content` exists precisely because of that: upstream generates the runaway
  tail and then throws it away.

Measured rather than inferred, with `tools/reference/probe_runaway_decode.py`, on the crop of the
block that runs away — upstream's own checkpoint through `transformers`, greedy, no penalties:

| | tokens | stopped on EOS | time |
| --- | --- | --- | --- |
| upstream, capped at 400 | 400 | no | 19 s |
| upstream, at its own 8192 default | 8,192 | no | **525 s** |

One block, nearly nine minutes. Running upstream over **every** block of that page the same way
(`--blocks`, the port's layout boxes so both sides see the same fifteen blocks) puts a number on
the whole stage:

| | blocks | tokens | recognition |
| --- | --- | --- | --- |
| upstream, through `transformers` | 15 | 16,741 | **1,479 s** |
| this port, before the stop | 15 | 16,716 | 1,462 s |
| this port, after the stop | 15 | 1,100 | 118 s |

Two of upstream's fifteen blocks never stopped and took 1,428 s of that 1,479 s — 97%. The two
sides agree to within 1.2% on time and 25 tokens before the stop, which is the fidelity check that
makes the third row meaningful: after it, the port does the same page's recognition **12x faster
than the original**, and the difference is entirely tokens not generated.

Layout detection and markdown assembly are outside that comparison; on the port's side they are
about 10 s, so including them would not move it. The port's rows are its profile's vision + prefill
+ decode; upstream's is wall time around `generate` with the model already loaded. The output is
`This is text.` repeated until the budget runs out, which is the same text this port produced
before the stop.

The two implementations then agree on what survives, which is what pins the divergence down to
string length rather than to the port. Upstream's own `truncate_repetitive_content`, at the
`min_count` its pipeline passes for a non-table block (50, and 5000 for a table — the values
`RepetitionTruncator` uses), returns:

| given | returns |
| --- | --- |
| upstream's 8,192-token output (28,671 chars) | `This is text.` ×3 |
| this port's stopped 384-token output (1,343 chars) | `This is text.` ×1 |

Both are upstream's function on the respective strings; the shorter one takes the
shortest-repeating-unit branch instead of the suffix branch. So the port's pre-stop output matched
upstream exactly, and its post-stop output is what upstream's own truncator makes of a shorter
decode of the same cycle. Neither is the page's text — the source repeats that sentence
thirty-one times and no decode of it terminated.

**Eight of the nine pages come out byte-identical**, and the detector fired on exactly two blocks in
the whole corpus — the two runaway ones. The ninth page differs only in how many copies of a
sentence the source repeats thirty-one times survive: three before, one after. Both are the
truncator's arbitrary reduction of a decode that never terminated, so neither is the page's text;
what changed is which arbitrary reduction, because the truncator takes a different branch on a
shorter string. On the corpus's character-accuracy metric that costs 0.4 points of mean, all of it
on that one page.

### Where the allocations go

A stage that allocates a lot is not automatically slow, so this is worth separating from the
timings above. Two of the three big allocators turned out to be fine and the third was the
layout graph's whole problem.

| Stage | Per call, before | After |
| --- | --- | --- |
| layout detection | 4,458 MiB, every time | 1,038 MiB cold, then **145 MiB** |
| vision tower | 411 MiB cold | 4 MiB per steady pass |
| decoder prefill | 154 MiB cold | 3 MiB per steady pass |

The tower and the decoder were already pooled; their cold figures are the pool filling, which is
why the per-block column of `RecognitionProfile` shows one large block at 672 MiB and the next
identical one at 67. Reading those as a leak is a mistake worth naming, because it sends you
looking in the wrong half of the pipeline.

**The layout graph was allocation-bound.** `TensorArena` gives a run's intermediate tensors pooled
storage, and what makes that safe is knowing when a buffer is really dead — a tensor's array is
not its own, since `reshape`, `cast` between same-width dtypes and `share_data_` all return a
tensor over an existing array. The arena counts, per array, how many of the interpreter's value
slots reference it, and returns the array at zero. The interpreter is the only place a run's
tensors live and every operator's results land in slots it names, so the count is exact.
Interleaved against the control at constant load, four detections in one process: **4,458 MiB and
7.5 s per detection become 145 MiB and 4.7 s**.

That result is also the answer to two experiments in the list below that came out neutral. Neither
vectorising the graph's element-wise fallback nor taking the collector off a core did anything,
because neither addressed the allocation.

Pooling a buffer that a fresh allocation would have zeroed is a real hazard:
`GC.AllocateUninitializedArray` of this size arrives zeroed in practice, so an operator that read
its own output before writing it would have been quietly correct and would now produce garbage.
`PADDLEOCR_SHARP_POISON_ARENA=1` fills every rented buffer with `NaN` and `long.MinValue` first;
the corpus is byte-identical with it on, which is what says no operator does that.

The 145 MiB that remained was not scaffolding, which is what it looked like. `PirProfile` now
reports allocation per operator beside the timings, and it put 128 of the 145 against a single
operator — so the guess that it was a shape array and a results array per operator, several
thousand times over, was wrong twice: wrong about the shape, and describing something that adds up
to 2.5 MiB.

`TensorArena` reports its own accounting under `PADDLEOCR_SHARP_ARENA_STATS=1`, which is what
identified it. Per detection, steady state:

```
arena: 1611 rents of 6359 MiB, 3 missed the pool (0 MiB allocated); returned 6231 MiB at
last use and 0 MiB at the end; 128 MiB kept as fetches
```

**The 128 MiB was the fetches.** A run's intermediates all go back at their last use, but its
fetches cannot — they are the caller's. So each run took a bucket's worth of buffers out of
circulation and the next run allocated the shortfall again, and for a `[1, 300, 200, 200]` mask
that is 128 MiB a page in perpetuity. `RunPooled` returns a `PirRunResult` that gives those
buffers back when disposed; all three callers copy what they need out of the fetches and drop
them, so all three use it.

A further 8 MiB was the graph's *input* tensor, built before the run and so never seen by the
arena inside it. Rented and returned by hand.

| | per detection |
| --- | --- |
| before the arena | 4,458 MiB |
| arena | 145 MiB |
| fetches returned to the pool | 17 MiB |
| input tensor rented | **9 MiB** |

What is left is the scaffolding, and it is small: `batch_norm_`'s two per-channel arrays at 12 KiB
a call are the largest line at 1 MiB, and every operator together is 2.5 MiB. The rest is the
`object?[]` of value slots, the gathered mask bytes the polygon extractor copies out, and the
boxes themselves.

Those figures are the bench's, which feeds the same input every iteration. A real document is less
tidy, because the graph has content-dependent shapes — what `top_k` selects and what the gathers
index are functions of the page — so a new page can ask for a bucket the pool has not got yet.
Three pages of `multi_page_scanned.pdf` in one process allocate **1.6 GiB across three
detections**, against 13.4 GiB for the same three before the arena. The effect is self-limiting:
the pool accumulates the buckets a corpus needs and stops growing. A page parsed in a process of
its own still pays the cold ~1 GiB and nothing else, which is what the per-page corpus runs show.

The stats line's first figure is the graph's intermediate volume — **1,611 rents of 6,359 MiB** a
detection, pooled now rather than allocated. The obvious next move from there is to stop holding
every integral dtype as `long`, since a twelve-million-element boolean mask is then 96 MB where it
could be 12. **That is the wrong move, and the same stats line says so once it splits by storage:**

```
arena: by storage — float 5431 MiB, integral 928 MiB
```

Integral storage is 15% of the traffic. Narrowing it to a byte would take 928 MiB to about 116,
which is 13% of the total — and the operators that touch those tensors are no longer where the
time is either. Post-arena the graph was **`conv2d`-bound** at 47.9% of a 4.05 s detection, and
after the work in "Closing the layout gap" it is 60.0% of a 2.69 s one — the mask-head operators
that this question is about are 7.8%.

So the whole boolean-width question is worth at most half of 11.7% of a stage that is 8% of a
page, against a change to the interpreter's storage model and the byte-exactness every parity
check rests on. The convolutions are where the remaining work in this graph is, and they are
honest im2col-plus-GEMM work of the kind the tower's attention turned out to be.

That is the second time this section has claimed to know where the layout graph's time goes and
been wrong — first the element-wise mask kernels, then the allocation, now the convolutions. The
pattern is that each fix moved the bottleneck somewhere the previous profile could not see it, so
**re-profile after every change to this graph rather than working down a stale list.**

#### Choosing the degree of parallelism

One worker per core is the default and not always the right answer: a host that shares the machine
with its own work, or runs under a CPU quota `Environment.ProcessorCount` cannot see, wants fewer.
`DocumentParserOptions.Parallelism` takes a `ParallelOptions` for a parse, `Parallelism.Use` scopes
one around any other entry point, and `Parallelism.Default` sets it for the process;
`parse --max-parallelism <n>` is the same setting from the command line. The
`ParallelOptions.CancellationToken` is honoured by every kernel region, so it is also how a caller
cancels work already inside a GEMM.

It is ambient rather than a parameter because the kernels that spread work are static and sit a
dozen layers of shape algebra below the pipeline; threading an argument to each would put a
parameter on `Gemm.Linear` for the benefit of its callers' callers. It is an `AsyncLocal` rather
than a thread-static because `BlockConcurrency` above one puts a block's tower and decoder on pool
threads, where a thread-static set on the calling thread would not be — `ParallelismScopeTests`
pins both halves. The lookup happens once per parallel region, a few thousand times over a page
against regions that each cost microseconds: interleaved against the `static readonly` field it
replaced, four warm layout detections measured a 3,930 ms median against 3,850, inside a noise
band whose samples span 3,758-4,279 within one configuration.

Note that it is not `BlockConcurrency`. That decides how many blocks are recognised at once, this
how many threads the kernels inside one of them use, and the two multiply. Measured on
`ocr_test_original.png`, interleaved, with both endpoints repeated:

| degree | page | layout | vision |
| --- | --- | --- | --- |
| 4 (one per core) | 13.9 / 13.7 s | 8.3 / 7.9 s | 3.9 / 4.1 s |
| 2 | 18.1 s | 9.1 s | 6.5 s |
| 1 | 27.5 / 29.0 s | 12.9 / 13.6 s | 10.5 / 11.4 s |

Vision spans 2.75x across that, close to the 3x the tower's own scaling measurement gives, while
layout spans 1.6x because much of that graph is serial. The output is byte-identical at every
degree, which it has to be — nothing here changes the arithmetic or its order.

#### `ArrayPool<T>.Shared` is not the pool its reputation says

`TensorPool` existed because "the default shared pool caps buckets at 1 MiB (2^20 bytes)", which
was true of .NET Framework and has not been true for years. The measurement that settles it, and
the rule that follows from it, are in
[`measuring-performance`](.claude/skills/measuring-performance/SKILL.md) §8; what matters here is
the consequence for this port.

Thirty-six live buffers of one size is exactly what `KvCache` holds, two per layer, and one growth
at the decoder's geometry is 576 MiB — so a cache on a created pool would churn 66 MiB per block
where the shared pool churns nothing. `KvCacheTests` pins it. **Before adding a pool of your own
here, measure the shared one.**

### Where the vision tower's time goes

`bench` prints a stage profile for the tower (`StageProfile`, the hand-written towers' answer to
`PirProfile`), and attention now records its own parts from inside — it runs a thread per head, so
nothing outside it can attribute its time. Each thread accumulates its own and they are summed, so
the figures are thread-seconds; over a stage that keeps every core busy that is what makes the
parts comparable with each other.

At the 4900-patch page a scanned block actually produces, before the work below:

| Stage | Share | GFLOP/s |
| --- | --- | --- |
| attention | 65.5% | 73 |
| MLP matrix products | 19.6% | 215 |
| QKV projections | 7.9% | |
| output projection | 2.7% | |
| rotary + head shuffles, GELU, norms, residuals | 4.3% | |

The rate column is the whole argument: attention carried more arithmetic than the MLP products and
ran at a third of their rate, on the same machine, against a measured all-thread FMA ceiling of
285 GFLOP/s.

Four changes took it from 40.8 s to 25.1 s, each A/B'd in one sitting with the untouched stages as
controls:

| | attention | scores | values |
| --- | --- | --- | --- |
| before | 40.8 s | | |
| parallelism capped at the core count | 36.6 s | 150.2 ms | 169.4 ms |
| transposed keys | 36.6 s | 128.4 ms | 169.4 ms |
| chunked value reduction | 29.1 s | 128.4 ms | 96.1 ms |
| token block outermost | **25.1 s** | **98.2 ms** | **96.1 ms** |

- **The degree of parallelism was unbounded.** `Parallel.For`'s default lets the pool inject
  workers while a queue stays non-empty, which is a policy for work that blocks. None of this
  blocks, and attention holds megabytes per worker at this page size, so the extra threads divided
  the cache and added switches. The instrumentation said so directly: the thread-time sum was 6.8x
  the stage's wall time on a four-core machine, and capping it took that to 3.89x. `Parallelism`
  is the one place that policy lives, and every kernel that spreads work reads it.
- **The score product reduces over 72 columns**, so the general kernel finishes every output with
  a lane reduction, and its four accumulators cannot fill two FMA ports at four cycles of latency.
  Transposing each head's keys to `[headDim][tokens]` makes the lanes output columns: the tile
  becomes four query rows by two column vectors — eight chains, six loads per eight multiply-adds.
  One transpose per head per layer against the `tokens / 16` passes the product makes over them.
- **That is the experiment recorded below as neutral**, and it was: with six or seven threads on
  four cores the stage was contention-bound and no kernel change could show through it. Both
  measurements were right, which is worth remembering before dismissing a mechanism twice.
- **The score kernel's loop order.** A tile of transposed keys is 4.6 KB and the block's row groups
  sweep it from L1; row group outermost instead read the head's whole 1.4 MB key matrix once per
  group.
- **The value product was store-bound.** It kept its output rows in memory and streamed the values
  past them, so the loop-carried dependency was a store and a reload of every output element on
  every one of 4900 reduction steps — one store per multiply-add, against a single store port —
  and it swept the values once per group of four output rows. Chunking the reduction at 80 tokens
  puts the output tile in registers for the chunk and keeps the chunk's values in L1 across the
  tiles that sweep it. The chunk is sized for L1 deliberately: sized for L2 the kernel simply
  becomes L2-bound and gains nothing.

End to end on `code_and_formula_scanned.pdf` page 1 — blocks up to 4,840 patches — bracketed by two
control runs in the same sitting: **vision 200.6 s -> 160.0 s (-20%), the page 270.0 s -> 230.1 s
(-15%)**, with layout, prefill and decode flat and the markdown byte-identical.

The win scales with block size, because attention's share does. On pages whose blocks are 600-800
patches the same build measures within noise of the control, and the corpus splits accordingly:
-20% of vision where blocks approach the pixel budget, ~0% where they are a seventh of it.

Two earlier notes in this section are superseded rather than wrong. The isolated kernel rates
(18.4 and 17.8 GFLOP/s) were measured under the uncapped pool, and the reasoning that the score
product's lane reduction is "the cost whatever the tile" holds only while the tile keeps reducing
along the lanes — the point of transposing is to stop.

### The worst page in the corpus, and what it was actually spending

The pages this section was tuned against are all a few blocks of a letter-sized scan. Over
`test_documents`, one page each, the slowest are a different shape:

| Page | Before |
| --- | --- |
| `pdf_scanned/nougat_010_scanned.pdf` p1 | **468.8 s** |
| `pdf_scanned/code_and_formula_scanned.pdf` p1 | 172.8 s |
| `images/balance_sheet_1.png` | 144.4 s |
| `pdf_scanned/docling_scanned.pdf` p1 | 125.3 s |
| `images/ocr_image.jpg` | 73.1 s |

`nougat_010` is forty-three blocks and 51,716 patches, and its profile splits 323.5 s of vision,
85.4 s of decode, 49.4 s of prefill and 5.1 s of layout. Two things about it are worth naming
before any kernel, because both are properties of the page rather than of the code.

**Its MediaBox is a lie.** The page declares 4967x3508 points — 69 inches by 49 — and holds one
6623x4678 JPEG whose aspect ratio is √2 to three decimals, so the thing that was scanned is an
A-series page in landscape. A dpi is a resolution per inch of the page the file claims to be, so
144 dpi renders it to 9934x7017: seventy megapixels, of which thirty-one carry the scan and the
rest are interpolation. Upstream renders it the same way and takes longer still.

**Forty-three small blocks each pay a floor.** `min_pixels = 112_896` is a lower bound on what
`smart_resize` hands the tower, which is 576 patches — so twenty-two one-line headings cost 576
patches each however few pixels they actually occupy. Twenty-four thousand of the page's patches
are that floor, and no change to the page's resolution moves them.

#### Where it went

Six changes, each measured on its own, took the page to **240.1 s at the defaults** — 1.95x — with
the recognised text equal or better (see below). None of them is specific to this page; over the
five slowest pages in the corpus they are worth 984.4 s → 624.4 s, and every one of those pages but
this one comes out byte-identical.

| Page | Before | After |
| --- | --- | --- |
| `nougat_010_scanned.pdf` p1 | 468.8 s | **240.1 s** (218.2 s at `--block-concurrency 2`) |
| `code_and_formula_scanned.pdf` p1 | 172.8 s | 120.6 s |
| `balance_sheet_1.png` | 144.4 s | 108.7 s |
| `docling_scanned.pdf` p1 | 125.3 s | 103.0 s |
| `ocr_image.jpg` | 73.1 s | 52.0 s |
| **total** | **984.4 s** | **624.4 s** |

- **The vision tower's attention had no 512-bit path.** `Vector512.IsHardwareAccelerated` is true
  on this machine — the runtime's preferred width is 512 here, unlike the machine the note under
  "Things that looked like wins" was written on — so `Gemm`'s `Dot4x4` had been running as `zmm`
  all along while `AttentionKernels` stayed at 256 bits, and attention is a third to two thirds of
  the tower. Widening it is **bit-identical**, which is what makes it safe to choose on vector
  width alone: in both the score and the value product the lanes are output columns and the
  reduction steps along them one at a time, so every output accumulates its terms in exactly the
  order the narrow kernel accumulates them. Interleaved over two rounds at 2916 patches, the tower
  went **20.1/19.0 s → 16.9/16.0 s**.
- **`softmax(scale · x)` is one pass cheaper than `scale` then `softmax`.** Scaling by a positive
  float is monotonic, so the scaled row's maximum is the scaled maximum of the raw row — the same
  element and the same single multiply, hence the same float32 value — and the scaled score can be
  produced in a register where the separate pass wrote it to memory and read it back. Also
  bit-identical, and attention is where it is worth removing: the row is the whole key sequence, so
  a 27-layer tower over a 5000-patch block writes and re-reads eleven billion floats for it.
- **The head shuffles, the GELU and the SiLU ran on one core.** Between the matrix products the
  tower moves `patches x 1152` floats four times a layer — 24 MB at a full-budget block — and
  applies an `exp` and a divide to `patches x 4304` more. Both are passes over independent
  elements and both now use the same worker count as the products beside them.
- **The panel counts did not divide across the workers.** `ChoosePanelWidth` sized a panel for L2
  and stopped there, which at the tower's own 1152-column shape gives ten panels for four workers:
  two of them finish a third early. That is the whole gap between the qkv and output projections at
  190 GFLOP/s and the MLP products at 244 — the MLP's 4304 columns happen to divide into
  thirty-eight, close enough that the tail costs nothing. Rounding the count up rather than the
  width down keeps every panel inside L2 and gives each worker the same number: qkv 192 → 210
  GFLOP/s, output 188 → 223, and the tower 16.4 → 15.8 s.
- **Blocks are decoded as a batch.** A decode step reads all 255M decoder parameters and the
  106M-parameter output head to produce one token — 646 MB of bfloat16 — so it is bound by how fast
  the weights arrive. Blocks on a page are independent, and stepping them together reads those
  weights once for the whole batch, which turns a page's decode into as many weight sweeps as its
  longest block needs rather than as many as all of them put together need. On this page, with the
  same 1,740 tokens generated either way:

  | decode batch | decode |
  | --- | --- |
  | 1 (before) | 85.4 s |
  | 8 | 39.0 s |
  | 16 | 28.5 s |
  | all 43 | **18.2 s** |

  What bounds a batch is memory, not a count: every member holds its own cache for as long as the
  batch runs, and a block's cache is proportional to its prompt. `DecodeBatch` therefore defaults
  to 0 — as many blocks as `DecodeBatchBytes` allows — so a page of forty headings decodes in one
  pass while a page of full-budget blocks still splits. The batched step is the one place the port
  now diverges numerically from itself: four or more rows reach `Gemm.Linear`'s widened-panel
  kernel where one row reaches its fused one, and the two group the same products differently.
  `BatchedDecodeTests` pins a batched row against the same sequence stepped on its own.
- **The decode kernel was waiting for memory it could have asked for earlier.** A decode step
  reads all 646 MB and reuses none of it, so `Gemm.Linear`'s single-row kernel is bound by line
  arrival and nothing else. The fix is a software prefetch, and *where* it points is the whole
  thing: a row of this model is two to six kilobytes and the loop crosses one in far less than the
  memory latency, so a prefetch that stays inside the current row arrives too late to be worth
  anything — measured, exactly neutral over two rounds. Pointing it at `i + 4·Cols` instead, the
  four rows the next call will read, covers one line per row per two steps, which is the rate the
  loop consumes them at. On `balance_sheet_1.png`, whose 1,503 tokens come from two blocks and so
  have nothing to batch with, **decode went 91.5/91.1 s → 63.9/64.5 s** and the page 137.9/135.3 s
  → 108.7/109.3 s, with identical output. It does nothing for a batched decode, which reaches four
  or more rows and so takes the widened-panel kernel, whose weight read is one long sequential
  stream the hardware already handles.
- **A page raster has a pixel budget.** `PdfRasterizer.DefaultMaxPagePixels` is twelve megapixels;
  a page that would exceed it is rendered at the resolution that fits. The budget is set to leave
  every ordinary page alone — most of `pdf_scanned` declares 1275x1650 points, which is a letter
  page written in 150-dpi pixels and renders to 8.4 megapixels, and all of those come out
  byte-identical — while catching the two files that are not mild: `nougat_010` at seventy
  megapixels and `nougat_009_scanned.pdf`, which claims to be 87 inches tall, at twenty-seven. On
  `nougat_010` it takes the patch count from 51,716 to 36,588. **Eight megapixels is the wrong
  budget** and was tried first: it clips the mis-declared letter scans by 3%, which moves their
  layout boxes and so their figure filenames, for nothing.

`--block-concurrency` now applies to the vision tower inside a batch — the token loop is already
shared — and narrows the kernels inside a block to match, so the two settings no longer multiply
into more threads than cores. It is worth 240.1 → 218.2 s here and is left off by default, because
the win is a property of blocks small enough to leave a core idle: four is worse than two even on
this page (222.5 s).

#### What it produces

Six further pages parsed with both builds, one page each: five byte-identical, and `nougat_004`
identical once the budget was raised off eight megapixels. That corpus went 396.9 s → 308.1 s, and
`nougat_006_scanned.pdf` — eighteen blocks, the shape the batch is for — 241.4 s → 173.4 s with
byte-identical markdown.

On `nougat_010` itself the text is not identical, because the page is one of the two the raster
budget resizes. It is not worse either. Against the 468.8 s run the differences are a hyphen, two
trailing dots, a trailing space, and one phrase the *new* output recovers that the old one dropped
(`Tap the … to return to the regular keyboard`, where the old output had `Tap the regular keyboard`). Sixty-two of the seventy megapixels were cost.

#### How much further the resolution knobs go, and what they cost

240 s is where this page lands with upstream's own per-block preprocessing, and it is close to the
floor for that preprocessing on four cores: 36,588 patches through 27 layers is 30 TFLOP of matrix
product before attention, and the tower sustains 210-230 GFLOP/s against an isolated GEMM ceiling
of 270-285. Under a minute means fewer patches, and the only thing left to take them from is the
`min_pixels` floor that twenty-two headings are sitting on.

Measured, with word similarity against the original 468.8 s output:

| | page | patches | word similarity |
| --- | --- | --- | --- |
| defaults | 240.1 s | 36,588 | — |
| `--block-concurrency 2` | 218.2 s | 36,588 | — |
| `--min-pixels 28224` | 109.3 s | 14,068 | 0.987 |
| `--min-pixels 784` | 92.2 s | 10,992 | 0.987 |
| `--min-pixels 784 --max-page-pixels 2000000 --block-concurrency 2` | **78.1 s** | 9,736 | 0.982 |
| the same at `--max-page-pixels 1200000` | 76.1 s | 9,208 | 0.969 |

The one to three percent the floor buys is real and is not noise: at the lower floor the model
stops distinguishing an en dash from a hyphen and a curly quote from a straight one, and it
lowercased one sentence opening. It also recovers the same dropped phrase. That is a quality
setting, so it is exposed and not defaulted — `min_pixels` stays at upstream's 112,896.

**Under a minute is not reachable on this machine at a resolution worth reading.** The last two
rows are within two seconds of each other because once the floor is off, the patch count tracks the
page area and the page is already down to what its text needs: halving it again means rendering the
real A4-landscape page at about 70 dpi. What is left at 78 s is 45 s of vision over 9,736 patches —
8 TFLOP of matrix product, which four cores at the 210-230 GFLOP/s this tower sustains cannot do in
much less — 12.5 s of prefill, 15.6 s of decode for 1,737 tokens, and 3.5 s of layout. Getting
under sixty would take a materially better GEMM, not another pass over this pipeline.

### Alignment and the worker team: what a four-core page was spending that no kernel profile showed

A synthetic A4 page (title, three paragraphs, a 7x4 table; `tools/dpi_study/`) parses to byte-identical
markdown at every resolution from 40 to 400 dpi and falls apart below 35, so 100 and 150 dpi are the
cheapest settings that still read it. Those two pages were profiled and tuned; what follows is the
record. Machine: 4 cores, Xeon with AVX-512 (`avx512_fp16`, AMX present but unreachable from .NET).
Whole process, cold, interleaved base against final, three runs each, best of three with the spread:

| Page | before | after | | vision | prefill | decode | layout |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 dpi | 28.7 s (4.8%) | **17.7 s** (4.9%) | 1.62x | 14.6 -> 9.1 s | 3.0 -> 1.8 s | 7.8 -> 4.0 s | 3.1 -> 2.5 s |
| 150 dpi | 36.0 s (2.8%) | **23.0 s** (2.3%) | 1.57x | 20.2 -> 13.2 s | 3.9 -> 2.4 s | 8.3 -> 4.5 s | 3.2 -> 2.6 s |

The markdown is byte-identical on all twelve runs (`md5 5cfc8ef5` at both resolutions), so this is
the same work in less time, and so are the GEMM outputs: every one of ten shapes hashes the same
before and after. Allocation per page is unchanged (1.2-1.3 GiB, of which 1.0 GiB is the layout
arena filling cold) and so is peak RSS (2.77 against 2.79 GiB). `--block-concurrency 2`,
`--max-parallelism 2` and `1` all keep identical output and are faster too (29.3 -> 20.5, 44.6 -> 33.1,
77.4 -> 59.5 s at 100 dpi).

**The ceilings first, because the calibration in `bench` is not to be trusted on this machine.** Its
FMA figure came out at 988 GFLOP/s for one thread (and 1397 after a "fix" that was reverted): about
4.4x the 224 GFLOP/s an independent twelve-chain `a = a*m + c` loop sustains, which scales linearly to
874 over four threads. The same loop as `bench` writes it ran at 47 GFLOP/s in a separate process and
at roughly one iteration per cycle in the CLI — twelve FMAs per cycle, which no core does — so the
loop's behaviour depends on how it is compiled, and no absolute figure from it was used here.
Streaming read bandwidth, measured separately, is 45-49 GB/s over four threads and 15.8 over one.
Against 224 and 874, the tower's GEMMs were at 28% of peak, and decode streamed weights at
18.6 GB/s against 45. That is what said there was something to find.

**1. 512-bit loads from a .NET array always straddle two cache lines.** An array's elements start
sixteen bytes past its header, so no pooled buffer was 64-byte aligned, and every `vmovups zmm` from
one touched two lines. The four-by-four GEMM tile issues eight loads per sixteen FMAs, and split
loads doubled that to sixteen per eight cycles — both load ports. It was found by cloning the
library's panel structure in a harness and removing pieces one at a time until the one that moved the
number was the buffer's address: the same loop over `NativeMemory.AlignedAlloc` buffers ran 483
GFLOP/s at four threads against 331, aligned weight panel alone 450. `TensorPool.Rent` and
`Tensor.Rent` now start their usable region on a cache line (`AlignmentOffset`), which fixes the
activations, the attention buffers and the GEMM's widened panel at once; the convolution's im2col
columns, rented as bytes, got the same. Alignment moves data and changes no arithmetic, so it is
bit-identical by construction; `AlignmentTests` pins both the alignment and that a GEMM gives the
same bits either way. Tower, 1368 patches: 6.20 -> 4.74 s; conv GEMM 312 -> 421-468 GFLOP/s; warm
layout detection 2.5-2.76 -> 1.79-2.0 s. The address is read without pinning, so a gen0 array that a
collection relocates can end up misaligned — a speed loss and never a wrong answer, and the arrays
where it matters are in the large-object heap, which does not move; the test asserts alignment only
there.

**2. The same tile on raw pointers is 1.7x the span-indexed one.** `Dot4x4` indexed spans, so each of
its eight loads per step carried a bounds check and its own address arithmetic: 80 GFLOP/s a thread
against 135-142 for the identical tile on pointers (one thread, panel in L2). `Tile4x4Wide` is that
tile, with the same per-lane accumulation order and the same final reduction, writing straight into
`y` instead of through a scratch tile; the span version stays for 256-bit machines and for lengths
that are not a multiple of sixteen. In the full four-thread GEMM it was worth only 8% until the
alignment above, which is the order those two findings came in and why the second looked like a
small win for a while.

**3. `Parallel.For` parks its workers, and waking them costs more than a decode step's regions last.**
The decode step is a hundred and thirty parallel regions of 20-400 us, and a streaming read split
into per-matrix regions ran 29 GB/s against 57 with no region boundary. Per-thread busy time inside
one 12 MB region came out 388 / 395 / 186 / 117 us: two of four workers joined after the other two
had finished most of it, which is one and a half threads of work on four cores. Dispatch itself is
5 us hot, so the cost was never the fork-join; it was the wake-up of a thread that had parked during
the previous region. `WorkTeam` is the answer: three helpers that spin for 300 us after their last
region before they park, dynamic claiming (in blocks when there are many items), the caller as the
fourth worker, `Parallelism.For` as the one entry point, and `Parallel.For` as the fallback whenever
the team is in use — a region nested in a region, or a second block's tower under
`--block-concurrency`. Cancellation and exceptions surface as `Parallel.For` surfaces them, and the
helpers adopt the caller's `Parallelism` scope while they work (`WorkTeamTests`). Single-row GEMM at
DRAM speed went 456-496 -> 267-343 us (up to 47 GB/s, the ceiling), four rows 759-808 -> 402-422;
decode -45%, and the tower, whose regions are milliseconds, still gained 11% (4.54-4.74 -> 4.06 s on
1368 patches), though which of its stages that came from was not separated. Two details that cost a measurement each: the caller's wait for stragglers must not be
`SpinWait.SpinOnce`, which escalates to `Thread.Sleep(1)` — the first version of the team made
multi-row calls *slower* for exactly that reason — and publishing a region and parking a helper both
need a full fence, because each reads what the other just wrote and a store followed by a load may
reorder.

**4. What the attention kernels needed was smaller than it looked.** Padding the transposed keys and
score rows to a multiple of sixteen floats, and the per-head value and output rows to 80 (five cache
lines where 72 floats is four and a half), took the score kernel 5-10% and the value kernel nothing,
bit-identically. Attention is 21% of the tower, so that is 1-2% of it; it stays because it is free.

**5. The outer-product tile, built and left opt-in (`PADDLEOCR_SHARP_GEMM=outer`).** The one change
here that is not bit-identical, so it was built and measured separately. Lanes run along activation
rows instead of along the reduction: activations are packed once per call into `[k][32]` groups
(the last zero-padded), weights stay in the row-major panel and are read by broadcast, and a tile of
thirty-two rows by eight columns keeps sixteen independent chains with nothing to reduce afterwards.
Reductions longer than 1152 are cut into chunks with the accumulators carried through memory, which
is the same sequential chain resumed; without that the two large-reduction shapes gained 2% and 7%
where the others gained 22%, because four 551 KB row groups no longer fit L2. Carrying the resume
as a flag on the hot kernel cost the JIT its register allocation (a 520 GFLOP/s kernel measured
280), so resuming is a separate copy of each kernel.

| | dot form | outer form | |
| --- | --- | --- | --- |
| tower qkv / proj, 1368 rows | 428 | 477 GFLOP/s | 1.11x |
| tower MLP up | 435 | 518 | 1.19x |
| tower MLP down (K 4304) | 349 | 405 | 1.16x |
| projector (K 4608) | 333 | 422 | 1.27x |
| prefill qkv / MLP | 403 / 404 | 474 / 481 | 1.18-1.19x |

Medians of five interleaved rounds, spread 2-5%. What it is worth on a page is much less, because
the GEMMs are about two thirds of the tower and the tower is a bit over half of the page: 150 dpi
21.8 -> 20.7 s (mean of three interleaved runs; best 21.4 -> 20.1 s), 100 dpi within noise, vision
12.3 -> 11.6 s, layout convolutions 957-1103 -> 862-959 ms. A projection of 10% for the whole page,
made from the isolated kernel, came out at about half that.

**Why the page gains about a third of what the kernel does.** The 19% on the tower's MLP shape
becomes 10% of the tower's MLP stage in place (stage medians, dot against outer: MLP 2112 -> 1899
ms, qkv 879 -> 840, out 300 -> 280), for two reasons. The activations are repacked on every call,
and three projections that share one input repack it three times: 4-5% of the GEMM time, 3% of the
tower, measured with a temporary stopwatch. The rest of the gap was not isolated; the harness calls
one matrix repeatedly, so its weights sit in L3, where the tower streams each layer's from memory.
And the tile only reaches half of a page. At 150 dpi, estimated from the tower's and the prefill's
stage profiles (GEMM 67% of the tower, 71% of the prefill, attention 24% and 20%):

| | s | GEMM tile | the rest |
| --- | --- | --- | --- |
| vision | 12.0 | 8.0 | attention 2.9, shuffles / norms / residuals 1.1 |
| prefill | 2.25 | 1.6 | attention 0.45, the rest 0.2 |
| decode (fewer than 32 rows, memory-bound) | 4.1 | 0 | all of it |
| layout | 2.3 | 0.6 | other operators ~0.9, cold start ~0.5, im2col and copies |

About 10 of 20.6 s is GEMM tile; 9% of that is 0.9 s, which is what was measured. The largest
untouched items are decode (20% of the page, 2.3-2.8 s from its streaming floor), attention (16%,
at about 270 GFLOP/s of 860), and a floor no kernel moves: the five-token title block costs 2.0 s
of vision because `min_pixels` hands the tower 672 patches however little it says.

**Accuracy, which is the reason it is opt-in and not the reason to doubt it.** Per product, a
sequential float32 chain has 2.3-2.5x the rounding error of the dot form against a float64
reference (mean relative error 1.7e-6 against 7.5e-7 at a reduction of 1152, 3.7e-6 against 1.5e-6
at 4304); the two orders' results differ in the low bits of 95-98% of outputs. After 27 layers and
the projector the vision embeddings differ by a relative L2 of 4.0e-6 and 4.7e-6 (a text and a title
block) and 5.1e-5 (a table block), worst per-token cosine similarity 0.9999992. And on the synthetic
A4 page at fifteen resolutions from 20 to 400 dpi, **the markdown is byte-identical to the dot
form's at every one**, including 20-35 dpi where the recognised text is wrong (CER 73.5% down to 2.8%)
and greedy decoding has the least margin to spare: text CER, word recall and table cells match to the
digit at all fifteen (`tools/dpi_study/REPORT.md`). That is one document, a few hundred tokens a
resolution, and decode still runs the dot form (it has fewer than 32 rows); the corpus is what
would move this from opt-in to default, and `GemmOuterTests` pins the accuracy per product against a
float64 reference in the meantime.

**6. Projections that share an input are computed together, and nothing else about the weights is
worth doing ahead of time.** `Gemm.LinearMany` takes a layer's query, key and value (and the decoder's
gate and up) as one call: the activations are packed once instead of three times, and all the
projections' column panels run in one parallel region, so a decode step has four region boundaries a
layer where it had seven. It reads the weights where they lie rather than concatenating them, which
would have copied 215 MB out of the memory-mapped checkpoint. Each output is bit-identical to a
call of its own in both accumulation orders (`GemmManyTests`). Interleaved on a page, best of three,
before -> after: vision 12.8 -> 11.9 s and prefill 2.31 -> 2.13 s with the outer tile on, 13.3 ->
12.9 and 2.50 -> 2.32 without; markdown byte-identical in all four configurations. A single-row
decode step went 18.2 -> 16.4 ms at best of three, but the other two rounds were flat (19.6 against
18.8, 20.9 against 21.2), so a decode gain is not established.

What else was checked as work to do once, at load, instead of per call:

| Candidate | Measured | Verdict |
| --- | --- | --- |
| Align the weights in memory | 77% of the checkpoint's bytes sit at 32 mod 64, the rest at 0. Cold single-row decode streams at 44.8-46.4 GB/s from offset 0 and 45.1-47.4 from offset 32; five rows 20.6-22.9 against 20.3-23.6 | No effect. The weight loaders use 32-byte loads, which cannot straddle a line at 32-byte alignment; the 512-bit loads that did are on activations and panels |
| Widen bf16 to float32 once | Widening is 7.7 Gelem/s a thread: about 11 ms a tower pass, 0.25% of it | Nothing to save in the tower; float32 weights would double the bytes a decode step streams, which is what bounds it |
| Pre-fault the memory-mapped weights | A first tower pass costs 0-1.25 s more than the fourth (0.1 s in two of three runs); a page takes 325k minor faults for 0.8-0.9 s of system time spread over four cores | At most a couple of tenths of a second, and moving the faults would not remove them |
| Concatenate q/k/v and gate/up | `LinearMany` gets the same sharing without the copy | Not needed |
| Fold the norm scale and shift, or the attention scale, into the next weights | Would re-round bf16 weights | Not bit-identical and not a float32 rounding difference but a change to the model; rejected |
| Fold batch norm into the layout convolutions | `batch_norm_` is 25 ms of a 1.8 s detection | 1.4%, and not bit-identical |
| Keep an aligned copy of the convolution filters | +12% of the conv GEMM (about 0.07 s), 125 MB resident | Possible, small |

The one place the weights are re-processed on every call and it shows is the batched decode step: a
five-row step streams at about 21 GB/s where a one-row step streams at 45, because rows 4 and up
widen each panel into a float32 scratch before using it. That cannot be precomputed away (see the
float32 row above); it is the case for the skinny widen-on-the-fly kernel, and the measured gap is
larger than the earlier estimate under "Things that looked like wins": 66 batched steps a page at
16 ms each is about 0.8 s, 4% of it.

What is left, at 150 dpi: the tower's GEMMs are at 410-420 GFLOP/s (48% of peak, 86% of what the
bare tile does on four threads), decode is at 20 ms a step against a 14 ms streaming floor, and a
cold layout detection is 0.7 s slower than a warm one (JIT with tiering off, and 811 MiB of arena
buffers faulted in). The dot-form tile is capped at 137 GFLOP/s a thread by something a counter-less
VM cannot name: removing its sixteen lane reductions gives 155, a 64-bit loop index and a 2x unroll
give 138, taller tiles (6x4, 8x2) the same.

### Quantization: what int8 buys, what it costs, and the one flag

Behind `--quantization none|int8-compute` (`WeightQuantization` in the library; the environment
variable `PADDLEOCR_SHARP_QUANTIZATION` sets the default). Nothing is calibrated: plain
round-to-nearest, symmetric, one scale per output row. Off by default; the default build's output is
unchanged. `int8-compute` makes the vision tower, the projector and the decoder int8, and runs every
product with 32 rows or more in integer arithmetic on quantised activations (see below).

**There was a weight-only `int8` mode and it was removed.** It quantised the decoder and `lm_head` to
int8 with a scale per 128 weights, which cut decode by 28% and a page by 3% (257 s against 266 s over
the ten-page set), because the tower and prefill are bound by arithmetic and a narrower weight does
nothing for them. Not worth a second checkpoint to publish and a second set of recognition results to
explain next to the one that is worth it.

| | what is int8 | speed | disk | peak RSS |
| --- | --- | --- | --- | --- |
| `none` | nothing | | 1828 MiB | 2790 MiB |
| `int8-compute` | tower, projector, decoder; **activations too** on products with 32+ rows | **page -38%** | 1070 MiB | 2125 MiB |

`--quantization int8-compute` also selects what `download` fetches and what a parse downloads when
the cache is empty: `paddleocr-vl/int8-compute/model.safetensors` (1,121,856,264 bytes), with the
tokenizer and configs taken from the original `paddleocr-vl/` directory, since only the weights differ
(`ModelFile.RemotePath`, `ModelCatalog.PaddleOcrVL16Int8Compute`). The mirror answers a missing object
with HTTP 200 and the text `Object Not Found`, so the downloader checks for that body rather than
trusting the status. An explicit `--model-dir` is used as it is, bf16 or int8.

Ten pages (a formula document, a table, a book page, an English document, a text line, two scene
images, the synthetic A4 page at 100 and 40 dpi), one process, 266 s in bf16: **166 s** with
`int8-compute` (vision 169.4 -> 97.3 s, prefill 32.3 -> 18.9 s, decode 43.5 -> 29.2 s). The 150 dpi
synthetic page: 22.7 s -> 16.5 s on load, 17.0 s from a stored checkpoint; loading with on-the-fly
quantisation costs 1.5 s against 0.4 s, so the stored file is better in total. Over the fifteen
resolutions of the synthetic A4 page, from the downloaded checkpoint, the whole run is 492.0 s ->
354.7 s (1.39x), 1.4-1.5x for 45-200 dpi and 1.3x above, with **byte-identical markdown at every
resolution from 45 dpi up** and differing text only where bf16 is itself already wrong
(`tools/dpi_study/REPORT.md`).

**Where each speed-up comes from.** Weight-only int8 halves the bytes a decode step streams, and
decode is bound by exactly that, so it helps decode and nothing else (the mode that was removed): the tower and the prefill are
bound by arithmetic, and a narrower weight widened into the same float32 panel is no faster (int8
widened measured 313-467 GFLOP/s against bf16's 386-464 on the model's shapes). The page-level
gain needed integer *compute*: a 256-bit `vpdpbusd` does thirty-two multiply-adds, and a
four-by-four tile of them ran at 207-211 GMAC/s a thread against 67 for the float32 dot tile
(3.1x). Through `Gemm.Linear` on the model's own shapes it is 1.6-3.0x (735-1170 GFLOP/s
equivalent against 386-464), smallest on the 1152-column projections where quantising the
activations is a larger share of the call. The weights are used in place — a row of int8 is the
contiguous reduction vector the tile wants — so nothing is widened or repacked; activations are
quantised once per call (once for q, k and v together), shifted to unsigned for the instruction,
and the shift is undone with each weight row's sum, computed once per matrix. The scale must be one
per row: a grouped scale needs a conversion and a multiply-add every 128 weights, about half the
gain. AMX is not exposed by .NET, and the 512-bit form of the instruction was not exposed by .NET 10 (`AvxVnni`, the 256-bit form, was), so this
runs at 256 bits; without `AvxVnni` it is weights-only and its output can differ slightly from a
machine that has it. **.NET 11 does expose the 512-bit form** — see "What .NET 11 adds" below.

**What it costs in recognition**, against the bf16 output of the same ten pages:

| | pages identical | words changed | the synthetic A4 page |
| --- | --- | --- | --- |
| float32 control (summation order only) | 10/10 | 0 | exact |
| vision tower int8 only (per row) | 10/10 | 0 | exact |
| decoder only, per row (experiment) | 8/10 | 9 | exact |
| decoder only, per 128 (the removed `int8` mode) | 8/10 | 9 | exact |
| everything int8, weights only (experiment) | 7/10 | 11 | one table cell lost at 40 dpi |
| `int8-compute` (real kernel) | 5/10 | 44 | **a digit changed at 40 dpi** (`53,784` -> `53,794`); exact at 100 dpi |
| plain int4, per 128 | 4/10 | 134 | plain text changed, a table cell lost |

Everything that changes is on the two LaTeX pages except the following, which are the findings to
weigh: with `int8-compute` a trailing period drops (`loan.` -> `loan`), a traditional character
becomes simplified, one book sentence is reworded, and on the 40-dpi stress page — a page at the
edge of what the model can read, where bf16 itself loses a character — a digit in a table cell
changed. The formula differences are mostly spacing inside `$...$`, `\begin{array}` column
specs and `\boxed{}`, with some real misreadings in both directions: bf16 wrote `\kappa^2` for
`\lambda c^2` (int8 was right), int8 dropped a dot accent (`\dot{M}` -> `M`, int8 wrong). The
float32 control is identical on all ten pages, so none of this is summation order. Tower weights in
int8 cost nothing measurable; it is the decoder, and the activations, that cost. The real integer
kernel differs from its own float emulation on four pages (6/10 identical against 5/10, 32 words
against 44): the same arithmetic up to rounding, with long greedy decodes amplifying the
difference, so the two are one distribution and neither is "the" result. Plain int4 is rejected as
it stands; it would need calibrated rounding (GPTQ, AWQ), which is a different project.

**Files.** `tools/quantization/quantize_checkpoint.py <out>` writes the `int8-compute` checkpoint
(1070 MiB; its defaults are `--components all --group row`), which is what the mirror holds;
`--components dec --group 128` writes the weight-only experiment's. An int8 tensor is stored as `I8` with its scales beside it as `<name>.scale` (`F32`,
`[rows, cols / group]`); the loader reads whatever the file holds, and `--quantization` decides
only what an unquantised file is turned into on load and whether the integer path may run. The
Python quantiser and the library's on-the-fly one are the same arithmetic in float32. The
experiment tooling that produced the table is in the same directory: `make_variant.py` (an exact
quantise-then-dequantise float32 checkpoint of any scheme), `score_variants.py`, and the
`PADDLEOCR_SHARP_FAKE_A_BITS` hook, which emulates per-token integer activations without a kernel.

### What .NET 11 adds, and the one thing in it that matters here

The port moved to .NET 11 (see the project layout). Enumerating every `System.Runtime.Intrinsics.X86`
type in both runtimes on this machine (a Xeon with `avx512_vnni`, `avx512_bf16`, `avx512_fp16` and
AMX in `/proc/cpuinfo`) shows **one new class and one new nested class**:

| New in .NET 11 | Supported here | Use |
| --- | --- | --- |
| `AvxVnni.V512` — `vpdpbusd` / `vpdpwssd` on `Vector512` | **yes** | the int8 GEMM tile, at twice its width |
| `Avx512Bmm` — bit matrix multiply | no | none |
| `System.Numerics.BFloat16` | n/a | scalar conversions only; no vector operations |

Still not exposed, though the CPU has them: **AMX** (no type in either runtime), **AVX-512 BF16**
(`vdpbf16ps`) and **AVX-512 FP16** (reachable only through `Avx10v1`, which this CPU lacks).
`AvxVnniInt8` / `AvxVnniInt16` exist in both and are unsupported here; they would remove the
unsigned shift the int8 path applies to its activations, on a CPU that has them.

`tools/vnni` measures what the 512-bit instruction is worth, against the 256-bit one the kernels use
(five rounds, best of three each; rates count int8 multiply-adds):

| | 256-bit | 512-bit | |
| --- | --- | --- | --- |
| pure throughput, twelve chains | 190-215 GMAC/s | 312-367 GMAC/s | 1.6-1.9x |
| a 4x4 tile like `GemmInt8`'s, in L1 | 177-184 GMAC/s | 249-255 GMAC/s | **1.39-1.56x** |

The tile is the number that transfers: 1.4x on the arithmetic of the int8 products, against the 3.1x
the 256-bit tile already gave over the float32 one. What that is worth on a page is bounded by how
much of it is those products: at a 1156-patch block, GEMM is about half of the tower (MLP 952 ms,
qkv 369, output 168 of a pass of roughly 2.9 s) and at a 4900-patch block it is 21%, so **an
estimated 5-10% of vision at ordinary block sizes and 4-6% at the largest, before the work that does
not scale with the vector width** (quantising the activations, the row sums, the final reductions).
That is an estimate from the isolated tile, not a measurement of the kernel, and the isolated tile
over-promised before (item 5 under "Alignment and the worker team": 19% on the tile, 10% in place).
Two constraints on building it: a 512-bit reduction vector is 64 bytes, and the MLP's second
projection reduces over 4304, which is 67.25 of them, so that product needs a 256-bit tail; and the
256-bit kernel stays for machines without `AvxVnni.V512`.

### Attention at large blocks: what a 4900-patch block was spending

With the tower's matrix products in int8, attention is what is left of it. Profiled under
`--quantization int8-compute` at the 4900-patch block a scanned page produces, attention was **18.9 s
of a 26.8 s tower (70%)**, the MLP, qkv and output products 21%. Inside attention the thread time
split 44% scores, 46% values, 11% softmax. That is also why the int8 speed-up fell from 1.5x at
100 dpi to 1.3x at 400: attention stays float32 and grows with the square of the patch count.

The rate said where to look. 4·n²·72 FLOP a head a layer is 2,990 GFLOP for the tower at 4900
patches, so 18.9 s is **158 GFLOP/s, against about 270 at 150 dpi** with the same kernels. A rate
that falls as the problem grows is a cache or bandwidth effect, and the arithmetic of it fits: each
thread owns a head, so its working set is 1.4 MB of keys, 1.6 MB of values and 0.3 MB of scores,
more than a 2 MB L2; and every block of 16 query rows streams all of it once for 22.6 MFLOP, about
7.5 FLOP per byte, which at 158 GFLOP/s is 20 GB/s, the L3 figure measured elsewhere in this file.
The earlier "row-block sizing" and "taller tiles" entries below were rejected when the kernels were
half as fast and the stage a fifth of the tower; both conclusions were true then and are not now.

Measured in `bench`-sized isolation (`Attention.Bidirectional`, 16 heads x 4900 tokens x 72, one
process per configuration, interleaved, the baseline at both ends of every round, best of five
calls a process, eight rounds). **The host's spread here was 15-35% of the best from round to round,
so only the large effects below are established; the small ones are not.** All of these are
bit-identical to the old kernels (checked by hashing the whole output at 1, 37, 129, 1100, 1200,
2500 and 4900 tokens) except the chunk size.

| Change | 4900 tokens, best / median vs baseline |
| --- | --- |
| 8 query rows a block | +17% / +20% (slower) |
| 32 | -3% / -9% |
| 64 | -15..-21% / -12..-14% |
| 128, 256, 512 | -23% / -23..-25% (indistinguishable from each other) |
| 64 + tile-major keys | -20% / -16% |
| 64 + 4x3 score tile | -18% / -14% |
| 64 + tile-major keys + 4x3 tile | -26% / -22% |
| **128 + tile-major keys + 4x3 tile (shipped)** | **-23% / -25%** (-32% / -25% in an earlier round) |

Frozen into the code and run against the old build, interleaved, 8 rounds: **4900 tokens 603 -> 465 ms
a call (-22.9%, median -24.6%), 2500 tokens -15.4% (median -8.1%), 1200 tokens -8.9% (median
-6.8%, inside the noise)**, with the scores kernel's thread time 981 -> 582 ms and the values
kernel's 1070 -> 852 ms at 4900. Softmax and the transpose are flat, as controls.

End to end, the old and the new CLI built from the commits either side of the change, one process per
page, interleaved (the synthetic A4 page, `--quantization int8-compute`, 4 cores). **The markdown is
byte-identical on all twelve runs** (`md5 5cfc8ef5`).

| Page | patches | old | new | vision old -> new |
| --- | --- | --- | --- | --- |
| 150 dpi | 1193 | 21.1 / 19.9 / 20.1 s | 20.4 / 20.6 / 21.1 s | 11.4 / 10.7 / 10.8 -> 10.9 / 11.0 / 11.5 s |
| 400 dpi | 4451 | 109.9 / 106.8 s | 96.7 / 95.8 s (**-11%**) | 87.1 / 83.8 -> 73.7 / 74.4 s (**-13%**) |
| 300 dpi, bf16 (one pair) | 3121 | 90.4 s | 88.2 s | 68.9 -> 65.9 s |

The gain is the large-block gain and nothing else: a page whose blocks are around a thousand patches
does not move (the 150 dpi rows are inside each other's spread), and the 400 dpi page is the one
block of 4451 patches the isolated measurement above was built to look at. The 13% of vision is
what the harness's 23% of attention predicts once attention is 70% of the tower and the
transpose is counted; the single bf16 pair is a hint, not a result.

- **The query block is the effect.** Every block streams the whole of a head's keys and values, so
  the rows per block divide that traffic; 128 rows is 2.5 MB of scores at the largest page we
  accept and still inside L2, and past it nothing more is gained. The block of 16 was chosen at a
  size where this did not bind.
- **Tile-major keys.** The transposed keys are stored `[tile][headDim][128]` instead of
  `[headDim][tokens]`, so one score tile is a single 37 KB sequential read instead of 72 strided
  ones a whole key row (20 KB) apart: more streams than the prefetcher follows and more pages than
  the L1 TLB holds. Alone it is within the noise; with the 4x3 tile the scores kernel is 22%
  faster than with the block change alone (803 -> 629 thread-ms), and the pair is kept.
- **A 4x3 score tile** (four query rows by three 512-bit vectors, twelve accumulators, 48 keys a
  step) against 4x2. The leftover columns of a tile go to the 4x2 kernel, and a machine without
  512-bit vectors never takes it. Every output still accumulates over the head's 72 dimensions in
  the same order, so it is bit-identical.
- **`AttentionTests`** pins `Attention.Bidirectional` against a double-precision reference at
  1, 3, 37, 128, 129, 300, 1000 and 1030 tokens, under the 512-bit path and with
  `PADDLEOCR_SHARP_VECTOR_BITS=256`; there was no direct test of attention before this.

### Things that looked like wins and were not

Each of these is a plausible optimisation that the benchmark rejected. They are recorded because
the argument for them is still convincing on paper, and someone will otherwise try them again.
**Add to this list whenever an experiment is rejected**, with the measurement that rejected it.
Four of the entries below generalise past this port — interleave the configurations, establish
what a stage is bound by before tuning its inner loop, never act on a diagnosis inherited from a
comment, and the two cold-start answers that did not work — and the generalised forms are in
[`measuring-performance`](.claude/skills/measuring-performance/SKILL.md).

- **Banding `Gemm.Linear` over activation rows**, so a band stays cached across all the column
  panels. Consistently slower; the panel loop's traffic is evidently already absorbed by the
  shared cache.
- **Interleaving the weight panel** so the kernel's four weight loads are consecutive rather than
  four streams a row apart. Neutral at best and a 40% loss if the row-major copy is kept alive
  beside it, which doubles what each row-block sweeps through L2. The strides involved are not
  powers of two, so the prefetcher was already handling them.
- **Sizing attention's row-block from the token count**, so a page's keys and values are read
  half as many times. No measurable change, which is what rules key and value bandwidth out as
  attention's limit.
  **Superseded**: re-measured at 4900 patches after the kernels had doubled in speed, the rows per
  block *is* the effect — 128 against 16 is 23% faster. See "Attention at large blocks".
- **Giving attention's score product the four-by-four tile `Gemm.Linear` uses.** The argument is
  that `Dot4` issues five loads per four multiply-adds where `Dot4x4` issues eight per sixteen, so
  the wider tile should be load-bound where the narrow one is not. Exactly neutral, and the reason
  is worth keeping: both finish every output with a lane reduction, and over a 72-long inner
  dimension that reduction is the cost, at an identical 0.111 per multiply-add in both shapes.
- **Hoisting attention's value reduction outermost**, so the head's value matrix is read once per
  block instead of once per group of four rows — a fourfold cut in what looked like 20 GB/s of
  traffic. Neutral, because that traffic is served from L2 and L3, which are nowhere near their
  measured 51 and 22 GB/s.
- **Holding the value product's output tile in registers**, four rows by sixteen columns. A 15%
  loss: covering 64 columns then takes four passes over the value matrix, each touching a quarter
  of every row, which trades one streaming pass for four strided ones.
- **Transposing each head's keys so the score product never reduces along the vector lanes.**
  Measured neutral in the tower — 7,955 ms against 7,948 ms — and reverted. **This one was later
  re-measured and kept**: see the section above. The stage was contention-bound at the time, with
  six or seven pool threads on four cores, and no kernel change could show through that. The
  measurement was correct and the conclusion drawn from it was not, which is the trap worth
  keeping: a stage that is bound by something other than its kernel will report every kernel
  change as neutral, so establish what a stage is bound by before A/B-ing its inner loop.
- **Vectorising the layout graph's broadcast fallback.** Each element-wise kernel has fast paths
  for the contiguous shapes and a per-element delegate through `double` for everything else, and
  the mask head lives in that everything else: `[1, 300, 200, 200]` against `[1, 300, 1, 1]`
  broadcasts over the *leading* dimensions, so twelve million elements each paid two delegate calls
  and two conversions, and the operator moved 410 MB/s where the machine reads at 10 GB/s. A
  general plan — take the longest suffix of dimensions over which every operand is contiguous or
  constant, hand that suffix to the same vector kernels, walk the rest with a counter — covers
  every shape and removes the delegate entirely. Neutral: five warm layout runs measured 6.7-7.7 s
  against the control's 6.3-8.7 s, and a page measured 10.5-11.7 s against 11.7-12.3 s. Per-operator
  timings swung ±25-120% on operators the change never touched, which is what says the graph is not
  where its own profile appears to put it. Reverted, with the vectorised `Widen`, `Narrow` and
  `Truncate` conversions that went with it. **The reason is now known**: the graph was
  allocation-bound, and pooling its intermediates was worth 37% where the arithmetic was worth
  nothing.
- **Turning off background GC**, on the argument that a collector thread is a fourth claimant on
  four cores while the layout graph churns 4.5 GiB of large tensors. First measured as 8.3-9.0 s
  against 5.6-6.1 s over five warm layout runs, which is a large clean-looking separation and was
  entirely an ordering artefact: the configurations ran in sequence, and the whole sequence drifts
  downward as the 125 MB of layout weights settle into the page cache. Interleaved — on, off, on,
  off — the warm medians are 5.12, 5.27, 5.19, 5.32 s. **Interleave the configurations, or the
  first one measured pays for the page cache the others inherit.** The premise was half right and
  aimed at the wrong half: the churn was the problem, but the answer was to stop churning rather
  than to collect it faster.
- **Pooling `RgbImage`'s buffers through a pool sized past the shared one's supposed cap.** The
  premise was the stale cap above, so the change was pointless; worse, routing `Clone` through it
  rounds each image up to a power of two, and on a page's handful of differently-sized crops that
  cold-pool slack costs more than it saves — `stack` measured 35 MiB against the control's 20.
  Reverted. Images are a few tens of MiB a page against a gigabyte for the rest, so there was
  little to win either way.
- **Moving `KvCache` off `ArrayPool<T>.Shared`.** Same stale premise, and the measurement is the
  other way round: 0 MiB a round on the shared pool against 66-130 MiB on a created one. Reverted,
  and a test now pins it. The one thing worth keeping from the exercise is the shape of the mistake
  — a diagnosis inherited from a comment, acted on without measuring the thing the comment claimed.
- **Zeroing the convolution's im2col buffer once per worker instead of once per output row.**
  The buffer is cleared and then scattered into, and whether a position is written depends on the
  horizontal tap alone for every row a worker handles — so the positions the scatter skips are
  never written by any row and the initial zero would stand. That removes 2.1 GB of stores a
  detection and is worth **nothing**: 3.55/3.96, 3.81/3.72, 3.76/3.89 s against a control of
  3.55/3.82, 3.81/3.68, 3.76/3.71. The buffer fits L2, so the stores were never leaving it.
- **Filling that buffer pixel-outermost, so every store is contiguous.** Worse, not better: the
  fill went 1330 -> 1760 ms over three detections, because it trades strided stores for strided
  reads of every input channel. The tap-outermost order stayed.
- **Skipping im2col for 1x1 convolutions.** Their columns are a transpose of the input and nothing
  else — 382 MB of gathering a detection, for 38% of the convolution's arithmetic — and the
  operands in place are exactly the `k × n` form `Gemm.MatMul` takes. Slower: conv2d 1478 -> 1686
  ms, even with the register-blocked tile. The transpose buys the reduction a contiguous axis, and
  `Linear`'s four-by-four tile does sixteen multiply-adds per eight loads where the broadcast
  kernel does eight per six.
- **Widening `Gemm.Linear`'s column panel past 256 KB.** The panel is swept once per four
  activation rows, so a wider one should cut the traffic proportionally. Over layout detections at
  256/512/1024 KB: 3187/3361, 3223/3517, 3455/3147 ms — no separation at all.
- **ReadyToRun.** The usual answer to a cold process, and it measured worse on both the first
  detection (5373/5398 ms against 4581/5014) and the warm ones (2922/2757 against 2557/2562):
  its precompiled code targets a conservative instruction set, and this is SIMD-bound work that
  tiers up regardless. Turning tiered compilation off entirely is what worked — see "Closing the
  layout gap".
- **Using AVX-512 where the runtime's preferred width says 256.** A dependency-free FMA loop is
  60% faster at 512 bits, and the ISA is reachable regardless of the policy — but every 512-bit
  GEMM variant measured slower than the 256-bit kernel, including a narrowed tile chosen to fit
  the register file. See `Core/Simd.cs`; the override is left to whoever measures their own
  machine.
- **An outer-product GEMM tile, as the default.** Built and measured (item 5 above), and shipped
  opt-in rather than on: the GEMMs gain 11-27% but a page only 2-6%, and the project's bar is
  byte-identical output checked over a corpus that was not to hand. Turning it on is the one-line
  change to `GemmOuter.Enabled`.
- **Taller tiles, a 64-bit loop index and a 2x unroll for the dot-form tile.** 6x4 and 8x2 measured
  136-137 against 136-142 for 4x4 (same accumulation order, so they would have been bit-identical);
  a `nint` index with the loop unrolled twice measured 138. The tile is not loop-overhead-bound.
- **Taller score tiles in attention** (6x2, 8x2, 4x3 against 4x2): +10, +14 and +20% in isolation,
  which on a kernel that is 8% of the tower is about 1% of it. Not worth the code.
  **Partly superseded**: scores are about 30% of the tower's thread time at a 4900-patch block, not
  8%, and 4x3 is kept (with tile-major keys and 128-row blocks); 8x2 and 6x2 were re-measured and
  stay rejected — see below.
- **GC settings for the cold layout detection.** Workstation against server GC, concurrent on and
  off, over three interleaved rounds: first detection 2.3-2.9 s in every configuration, no
  separation. The 0.7 s it costs over a warm one is JIT and first-touch page faults, not collection.
- **Aligning the convolution's filters as well as its columns.** Measured +12% on the conv GEMM
  against +35% for the columns, but it needs either a per-call copy of the filters (125 MB over a
  detection, which eats most of the gain) or a second persistent copy. Columns only.
- **Plain round-to-nearest int4 weights.** Per group of 128: 4/10 pages unchanged, 134 words changed
  including plain English and a table cell; the weight RMS error is 13.9% against int8's 0.8-1.3%.
  Not built a kernel for. See "Quantization".
- **A skinny multi-row decode kernel (widen on the fly, no panel).** Estimated before building it:
  with the worker team in, a four-row step costs about 1.5x a one-row step, and the best this could
  do is bring that to about 1.2x, which over a page's 66 batched steps is about 0.2-0.4 s of 23.
  Not built — **and the estimate was low**: a cold-DRAM measurement later put five rows at 21 GB/s
  against 45 for one, which is about 0.8 s a page (item 6 above). It is the next thing to build.

- **8x2 and 6x2 score tiles, again, at 4900 patches.** Against 4x3, in the same interleaved sitting:
  8x2 +2.4% / +1.0%, 6x2 +0.4% / -4.6% (best / median) of the baseline, where 4x3 was -3.7% / -9.9%
  and 4x3 with tile-major keys and 64-row blocks -26% / -22%, 6x2 in that combination -24% / -19%.
  More accumulators per load is not what limits this kernel; 4x3 covers three vectors of keys a step
  and is the one kept.
- **Taking each row's maximum per tile, while the tile is in L1.** The softmax pass's first read of
  the row is what it removes: softmax thread time fell 257 -> 202 ms (-21%) and the scores kernel
  rose by about the same, 894 -> 920 ms, for no change in the total (b64 + max -12% against b64
  -3% in one round and -3% in median; 487 ms against 489 ms with tile-major keys and the 4x3 tile).
  The tile-local `TensorPrimitives.Max` costs what the later pass did, just from a warmer cache.
  Folding the maximum into the accumulators' stores would avoid the loads and was not built: the
  whole pass is 11% of attention.
- **Sweeping the score tile width (64, 96, 256, 512 keys).** Every one inside the noise of the
  baseline. The comment on that constant said the tile was 4.6 KB; at 128 keys it is 37 KB.
- **Sweeping the value chunk (40, 160 tokens against 80).** -4% and -1% in the noise, **and not
  bit-identical**: the chunk is how the reduction over tokens is grouped, so changing it regroups
  the float32 sums (the output hash differs at 1100 tokens). 80 stays, and it is a numerical
  choice as much as a cache one.

### Traps that make a SIMD microbenchmark lie

Three of them — accumulators seeded equal, a result consumed through a `volatile` field, and an
indexed accumulator that is never enregistered — each report roughly a fifth of the truth while
looking entirely reasonable, and all three were hit while building the calibration above. They
generalise past this port, so they live in
[`measuring-performance`](.claude/skills/measuring-performance/SKILL.md) §6 with the rest of the
method. Read that before writing a microbenchmark; `MachineProfile.cs` is the worked example of the
traps, and also of a fourth: its absolute figures were found wrong by 4.4x on the machine of the
latest pass (see "Alignment and the worker team"), so cross-check a ceiling against an independent
loop before dividing anything by it.

## What the pipeline does beyond the models

Roughly half the port is not model code. These are the stages that decide what the models are
asked and what becomes of their answers, all of them checked against the upstream functions
named beside them:

| Stage | Upstream |
| --- | --- |
| Drop overlapping regions, consulting their outlines | `filter_overlap_boxes` |
| Stack a paragraph split across columns into one image, and reorder around it | `merge_blocks`, `merge_images` |
| Stretch a formula crop's contrast and trim its margins, upscale a small spotting crop | `crop_margin`, `pre_process_for_spotting` |
| Cover figures inside a table with `[Fn]` placeholders and put them back afterwards | `tokenize_figure_of_table`, `untokenize_figure_of_table` |
| Cut runaway repetition out of a block's output | `truncate_repetitive_content` |
| OTSL markup to HTML | `convert_otsl_to_html` |
| Spotting's `<|LOC_n|>` coordinates to polygons | `post_process_for_spotting` |
| Number the blocks that belong to the reading flow | `update_order_index` |
| Render the page, HTML decoration and all | `MarkdownConverter`, `build_handle_funcs_dict` |
| Rejoin a table split by a page break | `merge_tables_across_pages` |
| Decide how deep each heading sits | `assign_levels_to_parsing_res` |

Three places diverge from upstream on purpose, and each says so where it is implemented: the
token glyphs painted over a table's figures (SkiaSharp, not OpenCV's Hershey font), the
clustering behind heading levels (an exact one-dimensional k-means, not scikit-learn's seeded
local search), and the shuffle that assigns those token numbers (a stable bijection, not
Python's Mersenne Twister). Two upstream quirks are reproduced rather than repaired, because both
decide what a consumer actually gets: the doubled quote in `untokenize_figure_of_table`'s `alt`
attribute, and `crop_margin` asking OpenCV for a BGR-to-grey conversion of a buffer that is RGB,
which swaps the red and blue weights and so decides which pixels of a formula survive the trim.

## Working agreements for this port

1. Read the upstream Python for a stage **before** writing the C# for it. Quote the file and
   line range in the C# doc-comment so the mapping is auditable.
2. Land work in vertical slices that build and test green. Track progress in
   [`to-do.md`](to-do.md).
3. Where a stage cannot be reproduced exactly — OpenCV's SIMD rounding, OpenCV's Hershey font
   for the table-figure placeholders — say so in the doc comment and pin down what *is*
   guaranteed instead.
4. Upstream CI, pre-commit hooks and agent skill files in this repository are **disabled**
   (renamed to `*.disabled`) for the duration of the port; do not re-enable them.
5. Do not modify the upstream Python packages (`paddleocr/`, `ppocr/`, `ppstructure/`, …)
   — the port is additive and lives entirely under `dotnet/`.
6. Performance work follows
   [`.claude/skills/measuring-performance`](.claude/skills/measuring-performance/SKILL.md). It is
   this port's own skill and is unrelated to the disabled upstream `skills/` in item 4. A timing
   claim that does not satisfy its checklist does not land, and a rejected experiment is written
   up under "Things that looked like wins and were not" rather than forgotten.

## Publishing

`.devops/build-nuget.yml` builds, tests and publishes on `main`: `PaddleOCR`, `PaddleOCR.Pdf`,
and `PaddleOCR.Cli` — the last as a .NET tool, so `dotnet tool install -g PaddleOCR.Cli` puts
`paddleocr-sharp` on the PATH. The package ids drop the `Sharp`; the assemblies and namespaces
keep it, as `HNSW` and `HNSW.Net` do in the sibling repository. Versions are
CalVer (`yy.M.<build id mod 65536>`), stamped by the pipeline; `Directory.Build.props` carries
0.1.0 for local packs. The tool package is the project's publish output — that is how the
SkiaSharp and PDFium native assets get in — so it is the one project `dotnet pack --no-build`
cannot pack.
