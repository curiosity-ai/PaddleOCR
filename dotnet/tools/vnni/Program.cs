// Measures what the 512-bit form of the int8 dot-product instruction (`vpdpbusd`, .NET 11's
// `AvxVnni.V512`) is worth against the 256-bit one the int8 GEMM kernels use, on the machine it
// runs on. Like `tools/ceilings`, it has no reference to the library, so it cannot share a bug with
// it. Two measurements, each the best of three, repeated for five rounds:
//
//   pure  twelve independent accumulator chains with both operands in registers: what the
//         instruction can do. Accumulators are seeded differently so the JIT cannot merge them.
//   tile  a four-by-four tile of dot products shaped like `GemmInt8`'s — eight vector loads and
//         sixteen instructions a step, every operand an aligned, L1-resident reduction vector.
//
// Rates count int8 multiply-adds: 32 per 256-bit instruction, 64 per 512-bit one.
//
//   dotnet run -c Release --project dotnet/tools/vnni
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// (a) pure throughput: 12 independent chains, operands in registers, accumulators seeded differently.
[MethodImpl(MethodImplOptions.AggressiveOptimization)]
static long Pure256(long iters)
{
    var a = Vector256.Create((byte)3); var b = Vector256.Create((sbyte)5);
    Vector256<int> c0 = Vector256.Create(1), c1 = Vector256.Create(2), c2 = Vector256.Create(3), c3 = Vector256.Create(4), c4 = Vector256.Create(5), c5 = Vector256.Create(6),
                   c6 = Vector256.Create(7), c7 = Vector256.Create(8), c8 = Vector256.Create(9), c9 = Vector256.Create(10), c10 = Vector256.Create(11), c11 = Vector256.Create(12);
    for (long i = 0; i < iters; i++)
    {
        c0 = AvxVnni.MultiplyWideningAndAdd(c0, a, b); c1 = AvxVnni.MultiplyWideningAndAdd(c1, a, b); c2 = AvxVnni.MultiplyWideningAndAdd(c2, a, b); c3 = AvxVnni.MultiplyWideningAndAdd(c3, a, b);
        c4 = AvxVnni.MultiplyWideningAndAdd(c4, a, b); c5 = AvxVnni.MultiplyWideningAndAdd(c5, a, b); c6 = AvxVnni.MultiplyWideningAndAdd(c6, a, b); c7 = AvxVnni.MultiplyWideningAndAdd(c7, a, b);
        c8 = AvxVnni.MultiplyWideningAndAdd(c8, a, b); c9 = AvxVnni.MultiplyWideningAndAdd(c9, a, b); c10 = AvxVnni.MultiplyWideningAndAdd(c10, a, b); c11 = AvxVnni.MultiplyWideningAndAdd(c11, a, b);
    }
    var s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11;
    return Vector256.Sum(s);
}
[MethodImpl(MethodImplOptions.AggressiveOptimization)]
static long Pure512(long iters)
{
    var a = Vector512.Create((byte)3); var b = Vector512.Create((sbyte)5);
    Vector512<int> c0 = Vector512.Create(1), c1 = Vector512.Create(2), c2 = Vector512.Create(3), c3 = Vector512.Create(4), c4 = Vector512.Create(5), c5 = Vector512.Create(6),
                   c6 = Vector512.Create(7), c7 = Vector512.Create(8), c8 = Vector512.Create(9), c9 = Vector512.Create(10), c10 = Vector512.Create(11), c11 = Vector512.Create(12);
    for (long i = 0; i < iters; i++)
    {
        c0 = AvxVnni.V512.MultiplyWideningAndAdd(c0, a, b); c1 = AvxVnni.V512.MultiplyWideningAndAdd(c1, a, b); c2 = AvxVnni.V512.MultiplyWideningAndAdd(c2, a, b); c3 = AvxVnni.V512.MultiplyWideningAndAdd(c3, a, b);
        c4 = AvxVnni.V512.MultiplyWideningAndAdd(c4, a, b); c5 = AvxVnni.V512.MultiplyWideningAndAdd(c5, a, b); c6 = AvxVnni.V512.MultiplyWideningAndAdd(c6, a, b); c7 = AvxVnni.V512.MultiplyWideningAndAdd(c7, a, b);
        c8 = AvxVnni.V512.MultiplyWideningAndAdd(c8, a, b); c9 = AvxVnni.V512.MultiplyWideningAndAdd(c9, a, b); c10 = AvxVnni.V512.MultiplyWideningAndAdd(c10, a, b); c11 = AvxVnni.V512.MultiplyWideningAndAdd(c11, a, b);
    }
    var s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11;
    return Vector512.Sum(s);
}

// (b) a 4x4 dot-product tile shaped like the GEMM's: four activation rows against four weight rows, each a
// contiguous reduction vector, 16 accumulators, 8 loads per 16 instructions. Data is 64-byte aligned, in L1.
[MethodImpl(MethodImplOptions.AggressiveOptimization)]
static unsafe long Tile256(byte* act, sbyte* w, int stride, int steps, int reps)
{
    Vector256<int> c00 = Vector256.Create(1), c01 = Vector256.Create(2), c02 = Vector256.Create(3), c03 = Vector256.Create(4),
                   c10 = Vector256.Create(5), c11 = Vector256.Create(6), c12 = Vector256.Create(7), c13 = Vector256.Create(8),
                   c20 = Vector256.Create(9), c21 = Vector256.Create(10), c22 = Vector256.Create(11), c23 = Vector256.Create(12),
                   c30 = Vector256.Create(13), c31 = Vector256.Create(14), c32 = Vector256.Create(15), c33 = Vector256.Create(16);
    for (int r = 0; r < reps; r++)
    {
        for (int k = 0; k < steps; k++)
        {
            int o = k * 32;
            var a0 = Vector256.Load(act + o); var a1 = Vector256.Load(act + stride + o); var a2 = Vector256.Load(act + (2 * stride) + o); var a3 = Vector256.Load(act + (3 * stride) + o);
            var w0 = Vector256.Load(w + o); var w1 = Vector256.Load(w + stride + o); var w2 = Vector256.Load(w + (2 * stride) + o); var w3 = Vector256.Load(w + (3 * stride) + o);
            c00 = AvxVnni.MultiplyWideningAndAdd(c00, a0, w0); c01 = AvxVnni.MultiplyWideningAndAdd(c01, a0, w1); c02 = AvxVnni.MultiplyWideningAndAdd(c02, a0, w2); c03 = AvxVnni.MultiplyWideningAndAdd(c03, a0, w3);
            c10 = AvxVnni.MultiplyWideningAndAdd(c10, a1, w0); c11 = AvxVnni.MultiplyWideningAndAdd(c11, a1, w1); c12 = AvxVnni.MultiplyWideningAndAdd(c12, a1, w2); c13 = AvxVnni.MultiplyWideningAndAdd(c13, a1, w3);
            c20 = AvxVnni.MultiplyWideningAndAdd(c20, a2, w0); c21 = AvxVnni.MultiplyWideningAndAdd(c21, a2, w1); c22 = AvxVnni.MultiplyWideningAndAdd(c22, a2, w2); c23 = AvxVnni.MultiplyWideningAndAdd(c23, a2, w3);
            c30 = AvxVnni.MultiplyWideningAndAdd(c30, a3, w0); c31 = AvxVnni.MultiplyWideningAndAdd(c31, a3, w1); c32 = AvxVnni.MultiplyWideningAndAdd(c32, a3, w2); c33 = AvxVnni.MultiplyWideningAndAdd(c33, a3, w3);
        }
    }
    var s = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 + c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33;
    return Vector256.Sum(s);
}
[MethodImpl(MethodImplOptions.AggressiveOptimization)]
static unsafe long Tile512(byte* act, sbyte* w, int stride, int steps, int reps)
{
    Vector512<int> c00 = Vector512.Create(1), c01 = Vector512.Create(2), c02 = Vector512.Create(3), c03 = Vector512.Create(4),
                   c10 = Vector512.Create(5), c11 = Vector512.Create(6), c12 = Vector512.Create(7), c13 = Vector512.Create(8),
                   c20 = Vector512.Create(9), c21 = Vector512.Create(10), c22 = Vector512.Create(11), c23 = Vector512.Create(12),
                   c30 = Vector512.Create(13), c31 = Vector512.Create(14), c32 = Vector512.Create(15), c33 = Vector512.Create(16);
    for (int r = 0; r < reps; r++)
    {
        for (int k = 0; k < steps; k++)
        {
            int o = k * 64;
            var a0 = Vector512.Load(act + o); var a1 = Vector512.Load(act + stride + o); var a2 = Vector512.Load(act + (2 * stride) + o); var a3 = Vector512.Load(act + (3 * stride) + o);
            var w0 = Vector512.Load(w + o); var w1 = Vector512.Load(w + stride + o); var w2 = Vector512.Load(w + (2 * stride) + o); var w3 = Vector512.Load(w + (3 * stride) + o);
            c00 = AvxVnni.V512.MultiplyWideningAndAdd(c00, a0, w0); c01 = AvxVnni.V512.MultiplyWideningAndAdd(c01, a0, w1); c02 = AvxVnni.V512.MultiplyWideningAndAdd(c02, a0, w2); c03 = AvxVnni.V512.MultiplyWideningAndAdd(c03, a0, w3);
            c10 = AvxVnni.V512.MultiplyWideningAndAdd(c10, a1, w0); c11 = AvxVnni.V512.MultiplyWideningAndAdd(c11, a1, w1); c12 = AvxVnni.V512.MultiplyWideningAndAdd(c12, a1, w2); c13 = AvxVnni.V512.MultiplyWideningAndAdd(c13, a1, w3);
            c20 = AvxVnni.V512.MultiplyWideningAndAdd(c20, a2, w0); c21 = AvxVnni.V512.MultiplyWideningAndAdd(c21, a2, w1); c22 = AvxVnni.V512.MultiplyWideningAndAdd(c22, a2, w2); c23 = AvxVnni.V512.MultiplyWideningAndAdd(c23, a2, w3);
            c30 = AvxVnni.V512.MultiplyWideningAndAdd(c30, a3, w0); c31 = AvxVnni.V512.MultiplyWideningAndAdd(c31, a3, w1); c32 = AvxVnni.V512.MultiplyWideningAndAdd(c32, a3, w2); c33 = AvxVnni.V512.MultiplyWideningAndAdd(c33, a3, w3);
        }
    }
    var s = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 + c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33;
    return Vector512.Sum(s);
}

static double Best(Func<long> f, int n) { double b = double.MaxValue; long sink = 0; for (int i = 0; i < n; i++) { long t = Stopwatch.GetTimestamp(); sink += f(); b = Math.Min(b, Stopwatch.GetElapsedTime(t).TotalSeconds); } GC.KeepAlive(sink); return b; }
// "GMAC/s" counts int8 multiply-adds: 32 per ymm instruction, 64 per zmm.
long iters = 40_000_000;
Pure256(1000); Pure512(1000);
Console.WriteLine($"{"round",5} {"pure256 GMAC/s",15} {"pure512 GMAC/s",15} {"tile256 GMAC/s",15} {"tile512 GMAC/s",15}");
unsafe
{
    // 4 rows x 4 rows of 2048 bytes each (L1-resident: 16 KB per operand set at stride 2048 — 8 rows x 2 KB)
    int stride = 2048, steps256 = stride / 32, steps512 = stride / 64;
    byte* act = (byte*)NativeMemory.AlignedAlloc((nuint)(4 * stride), 64);
    sbyte* w = (sbyte*)NativeMemory.AlignedAlloc((nuint)(4 * stride), 64);
    for (int i = 0; i < 4 * stride; i++) { act[i] = (byte)(i * 7 + 3); w[i] = (sbyte)(i * 5 - 9); }
    int reps256 = 20_000, reps512 = 20_000;
    Tile256(act, w, stride, steps256, 10); Tile512(act, w, stride, steps512, 10);
    for (int round = 0; round < 5; round++)
    {
        double p256 = Best(() => Pure256(iters), 3), p512 = Best(() => Pure512(iters), 3);
        double t256 = Best(() => Tile256(act, w, stride, steps256, reps256), 3), t512 = Best(() => Tile512(act, w, stride, steps512, reps512), 3);
        double gp256 = iters * 12 * 32 / p256 / 1e9, gp512 = iters * 12 * 64 / p512 / 1e9;
        double gt256 = (double)reps256 * steps256 * 16 * 32 / t256 / 1e9, gt512 = (double)reps512 * steps512 * 16 * 64 / t512 / 1e9;
        Console.WriteLine($"{round,5} {gp256,15:F0} {gp512,15:F0} {gt256,15:F0} {gt512,15:F0}  (512/256: pure {gp512 / gp256:F2}x, tile {gt512 / gt256:F2}x)");
    }
    NativeMemory.AlignedFree(act); NativeMemory.AlignedFree(w);
}
