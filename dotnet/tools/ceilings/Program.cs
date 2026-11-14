using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

// Independent ceilings: 512-bit FMA throughput at 1, 2 and every thread, and streaming read
// bandwidth over a buffer far larger than the last-level cache. Run it before trusting any
// "fraction of peak": `dotnet run -c Release --project dotnet/tools/ceilings`.
//
// The FMA loop is `a = a * m + c` over twelve named, differently seeded accumulators. Named
// because the JIT will not enregister an indexed accumulator; differently seeded because equal
// chains are provably equal and collapse to one; twelve because two ports at four cycles of
// latency need eight in flight and a loop with exactly eight has no slack for its own counter;
// and consumed through GC.KeepAlive rather than a volatile field, whose release barrier makes the
// JIT spill every vector register on every iteration.
internal static unsafe class Program
{
    private static void Main()
    {
        Console.WriteLine($"cpus {Environment.ProcessorCount}, Vector512.IsHardwareAccelerated {Vector512.IsHardwareAccelerated}");

        foreach (int threads in new[] { 1, 2, Environment.ProcessorCount })
        {
            const long Iterations = 40_000_000;
            float sink = 0;
            double best = double.MaxValue;
            for (int rep = 0; rep < 5; rep++)
            {
                var clock = Stopwatch.StartNew();
                Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, t =>
                {
                    float r = Fma512(Iterations, t);
                    lock (typeof(Program))
                    {
                        sink += r;
                    }
                });
                best = Math.Min(best, clock.Elapsed.TotalSeconds);
            }

            GC.KeepAlive(sink);
            double flops = (double)threads * Iterations * 12 * 16 * 2;
            Console.WriteLine($"FMA 512-bit, {threads} thread(s): {flops / best / 1e9,7:F1} GFLOP/s");
        }

        const long Bytes = 1L << 30;
        byte* buffer = (byte*)NativeMemory.AlignedAlloc((nuint)Bytes, 64);
        for (long offset = 0; offset < Bytes; offset += 1 << 28)
        {
            new Span<byte>(buffer + offset, 1 << 28).Fill(2);
        }


        foreach (int threads in new[] { 1, 2, Environment.ProcessorCount })
        {
            float sink = 0;
            double best = double.MaxValue;
            for (int rep = 0; rep < 4; rep++)
            {
                long chunk = Bytes / threads;
                var clock = Stopwatch.StartNew();
                byte* start = buffer;
                Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, t =>
                {
                    float r = Read(start + (t * chunk), chunk);
                    lock (typeof(Program))
                    {
                        sink += r;
                    }
                });
                best = Math.Min(best, clock.Elapsed.TotalSeconds);
            }

            GC.KeepAlive(sink);
            Console.WriteLine($"DRAM read 1 GiB, {threads} thread(s): {Bytes / best / 1e9,6:F1} GB/s");
        }

        NativeMemory.AlignedFree(buffer);
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static float Fma512(long iterations, int seed)
    {
        Vector512<float> a0 = Vector512.Create(1f + (seed * 1e-3f));
        Vector512<float> a1 = a0 * 1.01f, a2 = a0 * 1.02f, a3 = a0 * 1.03f, a4 = a0 * 1.04f;
        Vector512<float> a5 = a0 * 1.05f, a6 = a0 * 1.06f, a7 = a0 * 1.07f, a8 = a0 * 1.08f;
        Vector512<float> a9 = a0 * 1.09f, a10 = a0 * 1.1f, a11 = a0 * 1.11f;
        Vector512<float> m = Vector512.Create(0.9999f);
        Vector512<float> c = Vector512.Create(1e-7f);

        for (long i = 0; i < iterations; i++)
        {
            a0 = Vector512.FusedMultiplyAdd(a0, m, c);
            a1 = Vector512.FusedMultiplyAdd(a1, m, c);
            a2 = Vector512.FusedMultiplyAdd(a2, m, c);
            a3 = Vector512.FusedMultiplyAdd(a3, m, c);
            a4 = Vector512.FusedMultiplyAdd(a4, m, c);
            a5 = Vector512.FusedMultiplyAdd(a5, m, c);
            a6 = Vector512.FusedMultiplyAdd(a6, m, c);
            a7 = Vector512.FusedMultiplyAdd(a7, m, c);
            a8 = Vector512.FusedMultiplyAdd(a8, m, c);
            a9 = Vector512.FusedMultiplyAdd(a9, m, c);
            a10 = Vector512.FusedMultiplyAdd(a10, m, c);
            a11 = Vector512.FusedMultiplyAdd(a11, m, c);
        }

        return Vector512.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static float Read(byte* p, long bytes)
    {
        Vector512<uint> s0 = Vector512<uint>.Zero, s1 = s0, s2 = s0, s3 = s0;
        for (long i = 0; i < bytes; i += 256)
        {
            s0 += Vector512.Load((uint*)(p + i));
            s1 += Vector512.Load((uint*)(p + i + 64));
            s2 += Vector512.Load((uint*)(p + i + 128));
            s3 += Vector512.Load((uint*)(p + i + 192));
        }

        return Vector512.Sum(s0 + s1 + s2 + s3);
    }
}
