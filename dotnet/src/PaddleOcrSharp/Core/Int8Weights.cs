using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace PaddleOcrSharp.Core;

/// <summary>
/// Symmetric int8 weights with one float32 scale per group of consecutive reduction elements:
/// the quantiser, and the kernels that read the result.
/// </summary>
/// <remarks>
/// <para>
/// <c>w ≈ q · scale</c>, with <c>q</c> in <c>[-127, 127]</c> and <c>scale = max|w| / 127</c> over the
/// group, rounded to nearest even. It halves the bytes a decode step streams — 646 MB of bfloat16
/// per token — and decode is bound by exactly that. Nothing is calibrated: the scheme is plain
/// round-to-nearest, and what it costs in recognition is in <c>CLAUDE.md</c>.
/// </para>
/// <para>
/// The scale is applied to a group's partial sum, not to each weight: a decode kernel accumulates
/// the group's sixteen-wide products, folds the sum into the row's accumulator with one fused
/// multiply-add by the broadcast scale, and reduces once per row, at the end. The arithmetic is
/// float32 throughout; only the weights are narrow.
/// </para>
/// </remarks>
internal static unsafe class Int8Weights
{
    /// <summary>The group size the quantiser uses where the row length allows it.</summary>
    internal const int DefaultGroup = 128;

    /// <summary>The group size to use for a row of <paramref name="cols"/> elements.</summary>
    internal static int GroupFor(int cols) => cols % DefaultGroup == 0 ? DefaultGroup : cols;

    /// <summary>Quantises <paramref name="rows"/> rows of float32 weights, <paramref name="group"/> elements to a scale.</summary>
    /// <param name="source">Reads row <c>r</c> into the span it is given, as float32.</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Row length.</param>
    /// <param name="group">Elements per scale; divides <paramref name="cols"/>.</param>
    /// <returns>The int8 values, row-major, and the scales, <c>[rows, cols / group]</c>.</returns>
    internal static (byte[] Values, float[] Scales) Quantize(Action<int, Span<float>> source, int rows, int cols, int group)
    {
        int groups = cols / group;
        byte[] values = GC.AllocateUninitializedArray<byte>(checked(rows * cols));
        float[] scales = new float[rows * groups];

        Parallelism.For((rows + 63) / 64, block =>
        {
            float[] row = new float[cols];
            Span<sbyte> q = MemoryMarshal.Cast<byte, sbyte>(values.AsSpan());
            for (int r = block * 64; r < Math.Min(rows, (block + 1) * 64); r++)
            {
                source(r, row);
                for (int g = 0; g < groups; g++)
                {
                    ReadOnlySpan<float> w = row.AsSpan(g * group, group);
                    float peak = 0f;
                    foreach (float value in w)
                    {
                        peak = Math.Max(peak, Math.Abs(value));
                    }

                    float scale = peak / 127f;
                    if (scale == 0f)
                    {
                        scale = 1f;
                    }

                    scales[(r * groups) + g] = scale;
                    Span<sbyte> target = q.Slice((r * cols) + (g * group), group);
                    for (int i = 0; i < group; i++)
                    {
                        target[i] = (sbyte)Math.Clamp(MathF.Round(w[i] / scale), -127f, 127f);
                    }
                }
            }
        });

        return (values, scales);
    }

    /// <summary>Dequantises one row: <c>destination[i] = q[i] · scale[i / group]</c>.</summary>
    internal static void DequantizeRow(ReadOnlySpan<byte> values, ReadOnlySpan<float> scales, int group, Span<float> destination)
    {
        int length = destination.Length;
        fixed (byte* pq = values)
        fixed (float* ps = scales, pd = destination)
        {
            sbyte* q = (sbyte*)pq;
            int i = 0;

            if (Simd.Use512 && group % 16 == 0)
            {
                for (int g = 0; g * group < length; g++)
                {
                    Vector512<float> scale = Vector512.Create(ps[g]);
                    int end = Math.Min(length, (g + 1) * group);
                    for (i = g * group; i + 16 <= end; i += 16)
                    {
                        (Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(q + i))) * scale).Store(pd + i);
                    }
                }
            }
            else if (Simd.Use256 && group % 8 == 0)
            {
                for (int g = 0; g * group < length; g++)
                {
                    Vector256<float> scale = Vector256.Create(ps[g]);
                    int end = Math.Min(length, (g + 1) * group);
                    for (i = g * group; i + 8 <= end; i += 8)
                    {
                        (Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(long*)(q + i)).AsSByte())) * scale).Store(pd + i);
                    }
                }
            }

            for (; i < length; i++)
            {
                pd[i] = q[i] * ps[i / group];
            }
        }
    }

    /// <summary>Dot product of <paramref name="x"/> with one quantised row.</summary>
    internal static float Dot(ReadOnlySpan<float> x, ReadOnlySpan<byte> values, ReadOnlySpan<float> scales, int group)
    {
        Dot4Core(x, values, 0, 0, 1, scales, group, 0, out float a0, out _, out _, out _);
        return a0;
    }

    /// <summary>Dot products of <paramref name="x"/> with four consecutive quantised rows.</summary>
    /// <param name="x">The activation row.</param>
    /// <param name="values">Every row's int8 values, row-major.</param>
    /// <param name="row">First of the four rows.</param>
    /// <param name="cols">Row length.</param>
    /// <param name="scales">Every row's group scales.</param>
    /// <param name="group">Elements per scale.</param>
    /// <param name="lookahead">Bytes ahead to prefetch in each row, or 0 for none.</param>
    /// <param name="a0">Dot product with the first row.</param>
    /// <param name="a1">Dot product with the second row.</param>
    /// <param name="a2">Dot product with the third row.</param>
    /// <param name="a3">Dot product with the fourth row.</param>
    internal static void Dot4(
        ReadOnlySpan<float> x,
        ReadOnlySpan<byte> values,
        int row,
        int cols,
        ReadOnlySpan<float> scales,
        int group,
        int lookahead,
        out float a0,
        out float a1,
        out float a2,
        out float a3) =>
        Dot4Core(x, values, row, cols, 4, scales, group, lookahead, out a0, out a1, out a2, out a3);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Dot4Core(
        ReadOnlySpan<float> x,
        ReadOnlySpan<byte> values,
        int row,
        int cols,
        int count,
        ReadOnlySpan<float> scales,
        int group,
        int lookahead,
        out float a0,
        out float a1,
        out float a2,
        out float a3)
    {
        int length = x.Length;
        int groups = length / group;
        float r0 = 0f, r1 = 0f, r2 = 0f, r3 = 0f;

        fixed (float* px = x, ps = scales)
        fixed (byte* pv = values)
        {
            sbyte* q0 = (sbyte*)pv + ((long)row * length);
            sbyte* q1 = q0 + length;
            sbyte* q2 = q1 + length;
            sbyte* q3 = q2 + length;
            float* s0 = ps + ((long)row * groups);
            float* s1 = s0 + groups;
            float* s2 = s1 + groups;
            float* s3 = s2 + groups;

            if (Simd.Use512 && group % 16 == 0 && length % group == 0)
            {
                Vector512<float> t0 = default, t1 = default, t2 = default, t3 = default;
                for (int g = 0; g < groups; g++)
                {
                    int b = g * group;
                    if (lookahead != 0 && (g & 1) == 0)
                    {
                        for (int line = 0; line < group; line += 64)
                        {
                            Sse.Prefetch0(q0 + b + lookahead + line);
                            if (count == 4)
                            {
                                Sse.Prefetch0(q1 + b + lookahead + line);
                                Sse.Prefetch0(q2 + b + lookahead + line);
                                Sse.Prefetch0(q3 + b + lookahead + line);
                            }
                        }
                    }

                    Vector512<float> c0 = default, c1 = default, c2 = default, c3 = default;
                    if (count == 4)
                    {
                        for (int i = b; i < b + group; i += 16)
                        {
                            Vector512<float> xv = Vector512.Load(px + i);
                            c0 = Vector512.FusedMultiplyAdd(xv, Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(q0 + i))), c0);
                            c1 = Vector512.FusedMultiplyAdd(xv, Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(q1 + i))), c1);
                            c2 = Vector512.FusedMultiplyAdd(xv, Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(q2 + i))), c2);
                            c3 = Vector512.FusedMultiplyAdd(xv, Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(q3 + i))), c3);
                        }

                        t1 = Vector512.FusedMultiplyAdd(c1, Vector512.Create(s1[g]), t1);
                        t2 = Vector512.FusedMultiplyAdd(c2, Vector512.Create(s2[g]), t2);
                        t3 = Vector512.FusedMultiplyAdd(c3, Vector512.Create(s3[g]), t3);
                    }
                    else
                    {
                        for (int i = b; i < b + group; i += 16)
                        {
                            c0 = Vector512.FusedMultiplyAdd(
                                Vector512.Load(px + i),
                                Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(q0 + i))),
                                c0);
                        }
                    }

                    t0 = Vector512.FusedMultiplyAdd(c0, Vector512.Create(s0[g]), t0);
                }

                r0 = Vector512.Sum(t0);
                r1 = Vector512.Sum(t1);
                r2 = Vector512.Sum(t2);
                r3 = Vector512.Sum(t3);
            }
            else if (Simd.Use256 && group % 8 == 0 && length % group == 0)
            {
                Vector256<float> t0 = default, t1 = default, t2 = default, t3 = default;
                for (int g = 0; g < groups; g++)
                {
                    int b = g * group;
                    Vector256<float> c0 = default, c1 = default, c2 = default, c3 = default;
                    for (int i = b; i < b + group; i += 8)
                    {
                        Vector256<float> xv = Vector256.Load(px + i);
                        c0 = Vector256.FusedMultiplyAdd(xv, Widen(q0 + i), c0);
                        if (count == 4)
                        {
                            c1 = Vector256.FusedMultiplyAdd(xv, Widen(q1 + i), c1);
                            c2 = Vector256.FusedMultiplyAdd(xv, Widen(q2 + i), c2);
                            c3 = Vector256.FusedMultiplyAdd(xv, Widen(q3 + i), c3);
                        }
                    }

                    t0 = Vector256.FusedMultiplyAdd(c0, Vector256.Create(s0[g]), t0);
                    if (count == 4)
                    {
                        t1 = Vector256.FusedMultiplyAdd(c1, Vector256.Create(s1[g]), t1);
                        t2 = Vector256.FusedMultiplyAdd(c2, Vector256.Create(s2[g]), t2);
                        t3 = Vector256.FusedMultiplyAdd(c3, Vector256.Create(s3[g]), t3);
                    }
                }

                r0 = Vector256.Sum(t0);
                r1 = Vector256.Sum(t1);
                r2 = Vector256.Sum(t2);
                r3 = Vector256.Sum(t3);
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    float xv = px[i];
                    int g = i / group;
                    r0 += xv * q0[i] * s0[g];
                    if (count == 4)
                    {
                        r1 += xv * q1[i] * s1[g];
                        r2 += xv * q2[i] * s2[g];
                        r3 += xv * q3[i] * s3[g];
                    }
                }
            }
        }

        a0 = r0;
        a1 = r1;
        a2 = r2;
        a3 = r3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Widen(sbyte* p) =>
        Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(*(long*)p).AsSByte()));
}
