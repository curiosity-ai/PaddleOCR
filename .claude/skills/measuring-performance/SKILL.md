---
name: measuring-performance
description: >-
  How to measure performance on this port so a result means something: calibrating the machine
  before the code, ensuring the code under test is actually compiled, best-of-N with the spread,
  interleaved A/B runs against an untouched control, pinning the output, and the microbenchmark
  traps that report a fifth of the truth. Load before benchmarking, profiling, optimising a
  kernel, or accepting or rejecting a performance change. Not needed for ordinary correctness
  work.
---

# Measuring performance

This port is developed on shared, virtualised hardware. The core count is whatever the hypervisor
schedules, the clock moves with the host's AVX-512 licence state, and a noisy neighbour halves the
memory bandwidth between one run and the next. On the reference machine the per-sample spread runs
from 10% to 40%, which is wider than most differences worth acting on.

So a stage time in milliseconds is not by itself evidence. **Absolute milliseconds from different
days are not comparable, and neither are two configurations measured in sequence on the same day.**
Everything below exists to make a number mean something anyway.

The project's own measurements — what each stage costs, what has already been tried, where the
time and the allocations actually go — live in [`CLAUDE.md`](../../../CLAUDE.md). This file is the
method; that file is the record.

---

## 1. Measure the machine before you measure the code

`paddleocr-sharp bench` starts by measuring the machine, before it loads anything: the FMA rate
the hardware sustains at each vector width, at one thread and at every thread, and the read
bandwidth at each level of the hierarchy. Stage times are then reported as a fraction of those
ceilings as well as in milliseconds, and `--gemm true` reports each GEMM shape the same way.

The reference implementation is
[`dotnet/src/PaddleOcrSharp.Cli/MachineProfile.cs`](../../../dotnet/src/PaddleOcrSharp.Cli/MachineProfile.cs).
Read it before writing any new calibration; every detail in it was learned by getting it wrong
(see §6).

Three things the profile reports are worth acting on:

- **`runtime prefers 256`** — the ISA is present but `Vector512.IsHardwareAccelerated` is false by
  default. A kernel gated on that property is running narrow while the one beside it, gated on
  `Avx512F.IsSupported`, is running wide. That is how the vision tower's attention and its GEMM
  silently diverged.
- **FMA ports**, inferred from the ratio between the two widths, which is what makes the clock
  estimate possible at all.
- **Jitter**, the honest error bar on everything else. Above a few percent the machine is busy
  with something and every stage time in the run is an upper bound. The profile says so in words;
  read it.

A fraction-of-ceiling figure still means the same thing tomorrow. A millisecond figure does not.

## 2. Make sure you are measuring compiled code

Three separate problems hide under "warm up first".

- **Tiering.** A method starts at tier 0 and is recompiled with optimisations only after enough
  calls. Take untimed passes until the numbers stop moving.
- **On-stack replacement.** A method entered once and then spinning for a long loop is escaped
  through OSR, and OSR code keeps the frame's locals where tier 0 put them — on the stack. The
  loop head then reloads and re-spills every accumulator each iteration and reports
  store-forwarding throughput, six times under the real rate. Put
  `[MethodImpl(MethodImplOptions.AggressiveOptimization)]` on any long-running measurement body.
- **The production path may be cold.** The CLI parses one document and exits, so nothing ever
  reaches the steady state tiering is designed for. `PaddleOcrSharp.Cli.csproj` therefore sets
  `<TieredCompilation>false</TieredCompilation>`; the library is untouched, so a long-lived host
  keeps tiering and `DOTNET_TieredCompilation=1` overrides it. If you are measuring the tool,
  measure it the way it runs.

ReadyToRun is the usual answer to a cold process and measured **worse** here, on both the first
detection and the warm ones: its precompiled code targets a conservative instruction set and this
is SIMD-bound work that tiers up regardless.

## 3. Best of N, and report the spread beside it

Never report a single sample. The best sample is the one least contaminated by whatever else the
host was doing, and on shared hardware the noise is one-sided, so the best is also the least
biased estimator available. Three samples is the floor; `GemmBenchmark` uses nine because each is
cheap.

**The spread is not optional.** It is what says whether to believe the comparison at all: if the
spread is wider than the effect, the effect is not established. `GemmBenchmark.Time` is the shape
to copy, including the guard it prints when a shape measures above the calibrated ceiling — that
means the machine was busier during calibration than during the sweep, so the whole efficiency
column is a lower bound and should be read as ordering rather than as absolute.

## 4. Interleave the A/B, and keep an untouched control

An A/B where every A run happens before every B run is not an A/B.

Turning off background GC first measured 8.3-9.0 s against 5.6-6.1 s over five warm layout runs: a
large, clean-looking separation, and entirely an ordering artefact. The configurations had run in
sequence and the whole sequence drifts downward as 125 MB of weights settle into the page cache.
Interleaved on-off-on-off, the medians are 5.12, 5.27, 5.19, 5.32 s. No effect at all.

**Interleave the configurations, or the first one measured pays for the page cache the others
inherit.** Repeat both endpoints. And bracket the run with a stage the change does not touch: if
the control moved, the machine moved and the run says nothing. That is why every result in
`CLAUDE.md` is quoted with its control flat beside it.

## 5. Establish what a stage is bound by before A/B-ing its inner loop

Transposing the vision tower's attention keys measured 7,955 ms against a 7,948 ms control and was
reverted as a dead end. It was later re-measured and kept: it is worth about 20% of that stage.
The first measurement was correct and the conclusion drawn from it was not. The stage was running
six or seven pool threads on four cores at the time, so it was contention-bound and *no* kernel
change could have shown through.

**A stage bound by something other than its kernel reports every kernel change as neutral.** Two
consequences:

- Find the binding constraint first — threads, allocation, memory, or arithmetic — and only then
  tune the inner loop. The layout graph was believed to be bound by its element-wise kernels, then
  by its allocation, then by its convolutions, and the first two beliefs were both wrong at the
  time they were acted on.
- **Re-profile after every change**, rather than working down a list made before the last one.
  Each fix moves the bottleneck somewhere the previous profile could not see it.

A neutral result on a contended stage is evidence about the stage, not about the change.

## 6. Two traps that make a SIMD microbenchmark lie

Both were hit while building the calibration in §1, and both report roughly a fifth of the truth
while looking entirely reasonable.

- **Accumulators that start equal.** Eight chains seeded to zero and updated identically are
  provably the same value, so the JIT emits one `vfmadd231ps` reusing a single register. The loop
  then measures FMA latency rather than throughput. Seed them differently.
- **Consuming the result through a `volatile` field.** The release barrier makes the JIT keep
  every SIMD local in the frame across the loop, so the body becomes load-operate-store and
  reports store-forwarding throughput. On the reference machine that turned 159 GFLOP/s into 27.
  `GC.KeepAlive` prevents dead-code elimination without touching register allocation.

A third, related: an indexed accumulator — an array or a `stackalloc` span — is never
enregistered, so the chains have to be named locals.

A calibration that is wrong by 5x is worse than none, because every kernel measured against it
then appears to exceed the hardware.

## 7. Pin the output, not just the number

The fastest implementation of any function is the one that returns the wrong answer, so a timing
claim without a correctness claim beside it does not land.

- **Byte-identical output** is the bar here, across the test corpus, and it is reachable for
  almost every change: nothing in a blocking, tiling or pooling change alters the arithmetic or
  its order. Say so explicitly when quoting a win.
- **Where bit-identity is genuinely unavailable** — a batched decode reaching a different GEMM
  kernel than a single row, a quantised path — state the tolerance and pin it with a test
  (`BatchedDecodeTests` is the example).
- **Pooling is a correctness change.** A buffer that a fresh allocation would have zeroed is a
  real hazard. `PADDLEOCR_SHARP_POISON_ARENA=1` fills every rented buffer with `NaN` and
  `long.MinValue`; a corpus that stays byte-identical with it on is what says no operator reads
  its own output before writing it.

## 8. Allocation is a separate axis, and an easier one

A stage that allocates a lot is not automatically slow, and a stage that is slow is not
necessarily allocating — but the layout graph was allocation-bound while every timing profile
pointed at arithmetic, so measure both. Allocation counts are exact and do not care how busy the
host is, which makes them the one performance figure safe to assert in a test.

- `GC.GetTotalAllocatedBytes(precise: true)` around a stage; `PirProfile` reports per operator and
  `TensorArena` reports its own accounting under `PADDLEOCR_SHARP_ARENA_STATS=1`.
- **Read a cold figure as the pool filling, not as a leak.** A pooled stage allocates its buffers
  once; the per-block column showing 672 MiB then 67 MiB for identical blocks is the pool warming.
  Misreading that sends you looking in the wrong half of the pipeline.
- **`ArrayPool<T>.Shared` is not the pool its reputation says.** "The shared pool caps buckets at
  1 MiB" was true of .NET Framework and has not been true for years; measured, it round-trips a
  1 GiB array with zero allocation at every size from 1 MiB up. `ArrayPool.Create(_,
  maxArraysPerBucket)` keeps only that many buffers of a size and drops the rest, so a created
  pool churns where the shared one does not — 0 MiB against 128 MiB over a round of 64 live 4 MiB
  buffers. **Before adding a pool of your own, measure the shared one.**

## 9. Write down what did not work

`CLAUDE.md` has a section titled *Things that looked like wins and were not*, with the argument for
each rejected change left intact. That is deliberate: the argument is convincing on paper and
someone will otherwise try it again, at the cost of a full measurement cycle.

**Add to it whenever an experiment is rejected**, including the measurement that rejected it. Two
of its entries are there because the same idea was tried twice, and one is there because a
measurement was right and the conclusion wrong.

---

## 10. Lessons from tuning the 100 and 150 dpi pages

Four things that were found by measuring and would not have been found by profiling the kernels.

- **Check a ceiling against an independent loop before believing it.** The calibration in `bench`
  reported 988 GFLOP/s for one thread; a separate twelve-chain `a = a*m + c` loop measured 224, and
  the GEMM's own best tile is 155. A fraction of a wrong ceiling is worse than no fraction, so the
  first thing to do on a new machine is run `dotnet run -c Release --project dotnet/tools/ceilings`,
  which has no reference to the library and so cannot share a bug with it, and compare. On the
  machine of that pass it said 226 / 459 / 863 GFLOP/s at one, two and four threads and 14 / 23 /
  42 GB/s of streaming read.
- **A kernel at 30% of peak is not necessarily a bad kernel.** The GEMM tile measured 137 GFLOP/s on
  its own and 331 over four threads in the library, 483 over four threads in a harness. The
  difference was found by cloning the library's panel structure in the harness and removing one
  piece at a time — widening, rent, scheduling — until the piece that moved the number was the
  buffer's *address*. 512-bit loads from a .NET array straddle two cache lines; align the data.
- **Measure what each thread does inside a parallel region, not only how long the region takes.** A
  region that streamed 12 MB in 430 us had per-thread busy times of 388 / 395 / 186 / 117 us: the
  other two workers had been parked since the last region and joined late. Wall time said "29 GB/s";
  the per-thread split said "two cores". The fix was a team that spins, and the test that it worked
  was the same split.
- **`SpinWait.SpinOnce` is not a spin.** It escalates to `Thread.Sleep(1)`, and a sleep of a
  millisecond is longer than the work being waited for. A hand-written wait loop that made things
  *slower* than `Parallel.For` was this, and nothing in the code said so.

Also: a microbenchmark that cycles through *distinct* weight matrices larger than the last-level
cache (`--cold`-style, 32 matrices of 12 MB) measures decode; one that repeats a single matrix
measures L3 and reports 66 GB/s for something that streams at 45.

## Checklist before quoting a performance result

- [ ] The machine was calibrated in the same run, and the jitter is stated.
- [ ] The code under test was warm, or deliberately measured cold, and it is said which.
- [ ] Best of at least three, with the spread reported beside the figure.
- [ ] The configurations were interleaved, both endpoints repeated.
- [ ] An untouched stage is quoted beside the changed one as a control.
- [ ] The output is byte-identical, or the tolerance is stated and pinned by a test.
- [ ] Allocation was checked as well as time.
- [ ] If the change was rejected, it is written down in `CLAUDE.md` with its measurement.
