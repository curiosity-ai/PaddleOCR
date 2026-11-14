namespace PaddleOcrSharp.Cli;

/// <summary>Usage output.</summary>
public static class Help
{
    /// <summary>Prints the usage banner.</summary>
    public static int Print()
    {
        Console.WriteLine(
            """
            paddleocr-sharp — pure C# PaddleOCR-VL

            Usage:
              paddleocr-sharp download [model ...]        Fetch models into the local cache
              paddleocr-sharp parse <image|pdf ...>       Parse pages to markdown / JSON
              paddleocr-sharp recognize <image>           Recognise one already-cropped block
              paddleocr-sharp bench                       Measure throughput of the model stages

            Common options:
              --model-dir <path>      Use a checkpoint directory instead of the cache
              --cache <path>          Override the cache root (default: ~/.cache/paddleocr-sharp)
              --endpoint <url>        Model mirror base URL (default: PADDLEOCR_SHARP_MODELS_URL)

            parse options:
              --layout-dir <path>     Use a PP-DocLayoutV3 directory instead of the cache
              --no-layout             Send the whole page to the VL model
              --layout-threshold <f>  Detection score threshold (default: 0.3)
              --chart                 Recognise charts instead of keeping them as images
              --seal                  Recognise seals instead of keeping them as images
              --ocr-images            Run OCR over image blocks too
              --doc-orientation       Rotate pages upright before parsing
              --doc-unwarping         Flatten curled pages before parsing
              --prompt-label <name>   Whole-page mode label when --no-layout (e.g. spotting)
              --dpi <n>               PDF rendering resolution (default: 144, as upstream)
              --max-page-pixels <n>   Largest page raster to render (default: 12000000, above ISO
                                      A2 at 144 dpi). A dpi is per inch of the page the PDF claims
                                      to be, and a scan's MediaBox is not always its real size; a
                                      page over the budget is rendered at the resolution that fits.
                                      0 renders at --dpi whatever the page claims.
              --max-pages <n>         Stop after this many PDF pages
              --password <text>       Password for an encrypted PDF
              --output-dir <path>     Write <name>.md, <name>.json and imgs/ per page
              --format markdown|json  Output format when writing to stdout
              --page-separator <text> Text between pages (default: a blank line)
              --block-concurrency <n> Blocks encoded in parallel (default: 1). The tower's kernels
                                      already use every core, so this only helps where blocks are
                                      small enough to leave cores idle; the kernels are narrowed to
                                      match, so the two settings do not multiply.
              --decode-batch <n>      Cap on the blocks whose token loops run as one batch
                                      (default: 0, i.e. as many as the cache budget allows). A
                                      decode step reads 646 MB of weights for one token, so a batch
                                      reads them once for all its blocks. 1 decodes a block at a
                                      time, as the port did before.
              --decode-batch-bytes <n>
                                      Key/value cache one batch may hold (default: 805306368)
              --max-parallelism <n>   Threads the kernels spread work across (default: one per
                                      core). Independent of --block-concurrency, which decides how
                                      many blocks run at once; the two multiply. -1 removes the cap,
                                      which is slower here — see CLAUDE.md.
              --quantization none|int8|int8-compute
                          How the model's weights are held (default: none). Nothing is
                          calibrated; formulas can change in small ways.
                          int8: the decoder's projections and output head are bytes with a
                          scale per 128 weights, which halves what a decode step streams.
                          int8-compute: also the vision tower and projector, one scale per
                          row, and products with 32 rows or more quantise their activations
                          too and use the integer dot-product instruction (AVX-VNNI),
                          about three times the float32 rate; without AVX-VNNI it is
                          weights-only. A checkpoint already quantised on disk is read as
                          it is stored. PADDLEOCR_SHARP_QUANTIZATION sets the default.
              --max-new-tokens <n>    Token budget per block (default: 8192)
              --stop-on-repetition <bool>
                                      Stop a block once its output has fallen into a verbatim
                                      cycle (default: true). The tokens this skips are the ones
                                      the repetition truncator discards anyway.
              --profile               Print per-block tokens, timings and allocations, and the
                                      per-stage cost of everything outside the model call
              --layout-profile        Also print the layout graph's per-operator cost

            recognize options:
              --prompt-label <name>   ocr | table | formula | chart | seal | spotting  (default: ocr)
              --prompt <text>         Raw instruction, overrides --prompt-label
              --max-new-tokens <n>    Token budget (default: 8192)
              --temperature <f>       0 selects greedy decoding (default: 0)
              --top-p <f>             Nucleus mass; ignored when greedy
              --repetition-penalty <f>
              --min-pixels <n>        Lower pixel budget (default: 112896)
              --max-pixels <n>        Upper pixel budget (default: 1003520)
              --<kind>-max-pixels <n> Per-label ceiling, overriding --max-pixels; <kind> is one
                                      of ocr, table, chart, formula, seal
              --format-block-content  Put each block's rendered form in the JSON, not its raw text
              --merge-tables          Rejoin tables split across a page break
              --title-levels          Set heading depth from the whole document
              --output <path>         Write the result to a file instead of stdout
              --profile               Print tokens, timings and allocations for the call

            bench options:
              --width <n> --height <n>   Synthetic page size (default: 1024x1024)
              --image <path>             Use a real image instead of a synthetic page. The tower
                                         does not care, but the decoder does: on noise the model
                                         stops at once and the decode loop never runs.
              --iterations <n>           Repeats per stage (default: 3)
              --tokens <n>               Tokens to decode per iteration (default: 32)
              --no-vl                    Skip the vision tower and decoder
              --no-layout                Skip the layout graph
              --gemm true                Sweep the GEMM shapes the model issues
              --calibrate false          Skip the machine measurement that heads the run
              --calibrate-seconds <f>    Time to spend measuring the machine (default: 4)

            Every run begins by measuring this machine — FMA rate at each vector width, cache and
            memory bandwidth, and the spread across samples — because stage times from a shared
            host are not comparable between runs without it. PADDLEOCR_SHARP_VECTOR_BITS=256|512
            overrides the vector width the kernels choose.
            """);

        return 0;
    }

    /// <summary>Reports an unknown verb.</summary>
    public static int Unknown(string verb)
    {
        Console.Error.WriteLine($"Unknown command '{verb}'.");
        Print();
        return 2;
    }
}
