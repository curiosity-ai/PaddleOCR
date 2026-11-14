using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats.Paddle;

namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>Reductions, sorting and selection.</summary>
internal static class ReduceOps
{
    /// <summary>How a reduction combines values.</summary>
    public enum Kind
    {
        /// <summary>Sum of the reduced elements.</summary>
        Sum,

        /// <summary>Largest of the reduced elements.</summary>
        Max,

        /// <summary>Smallest of the reduced elements.</summary>
        Min,

        /// <summary>Whether any reduced element is non-zero.</summary>
        Any,
    }

    /// <summary>Reduces <paramref name="input"/> along <paramref name="axes"/>.</summary>
    /// <param name="input">Tensor to reduce.</param>
    /// <param name="axes">Axes to reduce; empty means all of them.</param>
    /// <param name="keepDimension">Whether reduced axes stay as size-1 dimensions.</param>
    /// <param name="kind">Reduction to apply.</param>
    public static PaddleTensor Reduce(PaddleTensor input, long[] axes, bool keepDimension, Kind kind)
    {
        int rank = input.Rank;
        var reduced = new HashSet<int>(
            axes.Length == 0
                ? Enumerable.Range(0, rank)
                : axes.Select(axis => (int)(axis < 0 ? axis + rank : axis)));

        var shape = new List<int>();
        for (int i = 0; i < rank; i++)
        {
            if (!reduced.Contains(i))
            {
                shape.Add(input.Shape[i]);
            }
            else if (keepDimension)
            {
                shape.Add(1);
            }
        }

        PaddleDType dtype = kind == Kind.Any ? PaddleDType.Bool : input.Dtype;
        PaddleTensor result = PaddleTensor.Allocate([.. shape], dtype);

        // When the reduced axes are a contiguous suffix of the shape — which is what
        // `min(x, axis=[-2, -1])` over a mask stack looks like — every output element owns one
        // contiguous run of the input, so the whole reduction is a walk of vectorised segments
        // instead of a per-element index computation.
        if (IsSuffix(reduced, rank) && input.Count > 0)
        {
            int inner = 1;
            for (int axis = rank - reduced.Count; axis < rank; axis++)
            {
                inner *= input.Shape[axis];
            }

            if (inner > 0 && input.IsFloat && kind != Kind.Any)
            {
                ReduceSegments(input.FloatSpan, inner, kind, result.FloatSpan);
                return result;
            }

            // `any` over a mask stack is this shape and nothing else: [1, 300, 200, 200] reduced
            // to [1, 300], twelve million elements. Through the general walk below it cost 68 ms
            // of a 3.2 s detection — an index recomputed per element from the shape, and every
            // element through `double`.
            if (inner > 0 && kind == Kind.Any && !input.IsFloat)
            {
                AnySegments(input.Ints!, input.Count, inner, result.Ints!);
                return result;
            }
        }

        double seed = kind switch
        {
            Kind.Sum or Kind.Any => 0,
            Kind.Max => double.NegativeInfinity,
            _ => double.PositiveInfinity,
        };

        double[] accumulator = new double[result.Count];
        Array.Fill(accumulator, seed);

        int[] strides = PaddleTensor.Strides(input.Shape);
        int[] outputStrides = new int[rank];
        int stride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            if (reduced.Contains(i))
            {
                outputStrides[i] = 0;
            }
            else
            {
                outputStrides[i] = stride;
                stride *= input.Shape[i];
            }
        }

        int[] counters = new int[rank];
        for (int i = 0; i < input.Count; i++)
        {
            int target = 0;
            for (int axis = 0; axis < rank; axis++)
            {
                target += counters[axis] * outputStrides[axis];
            }

            double value = input.GetDouble(i);
            accumulator[target] = kind switch
            {
                Kind.Sum => accumulator[target] + value,
                Kind.Max => Math.Max(accumulator[target], value),
                Kind.Min => Math.Min(accumulator[target], value),
                _ => (accumulator[target] != 0 || value != 0) ? 1 : 0,
            };

            for (int axis = rank - 1; axis >= 0; axis--)
            {
                if (++counters[axis] < input.Shape[axis])
                {
                    break;
                }

                counters[axis] = 0;
            }
        }

        _ = strides;

        for (int i = 0; i < result.Count; i++)
        {
            if (result.IsFloat)
            {
                result.Floats![i] = (float)accumulator[i];
            }
            else
            {
                result.Ints![i] = (long)accumulator[i];
            }
        }

        return result;
    }


    /// <summary>Whether each contiguous segment holds a non-zero, one output element apiece.</summary>
    /// <param name="source">Integral storage, segments back to back.</param>
    /// <param name="count">Elements in <paramref name="source"/>.</param>
    /// <param name="inner">Elements per segment.</param>
    /// <param name="destination">One element per segment.</param>
    private static void AnySegments(long[] source, int count, int inner, long[] destination)
    {
        int segments = count / inner;

        Parallelism.For(segments, i =>
        {
            ReadOnlySpan<long> segment = source.AsSpan(i * inner, inner);
            destination[i] = segment.ContainsAnyExcept(0L) ? 1 : 0;
        });
    }

    /// <summary>Whether the reduced axes form a contiguous suffix of a rank-<paramref name="rank"/> shape.</summary>
    /// <param name="reduced">The axes being reduced.</param>
    /// <param name="rank">Rank of the tensor.</param>
    private static bool IsSuffix(HashSet<int> reduced, int rank)
    {
        for (int axis = rank - reduced.Count; axis < rank; axis++)
        {
            if (!reduced.Contains(axis))
            {
                return false;
            }
        }

        return reduced.Count > 0;
    }

    /// <summary>Reduces each contiguous run of <paramref name="inner"/> elements to one output.</summary>
    /// <remarks>
    /// The accumulation happens in float32, which is both faster and closer to Paddle than the
    /// double accumulation the general path uses.
    /// </remarks>
    private static void ReduceSegments(
        ReadOnlySpan<float> source,
        int inner,
        Kind kind,
        Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            ReadOnlySpan<float> segment = source.Slice(i * inner, inner);
            destination[i] = kind switch
            {
                Kind.Sum => System.Numerics.Tensors.TensorPrimitives.Sum(segment),
                Kind.Max => System.Numerics.Tensors.TensorPrimitives.Max(segment),
                _ => System.Numerics.Tensors.TensorPrimitives.Min(segment),
            };
        }
    }

    /// <summary>
    /// Top-<paramref name="k"/> values and their indices along <paramref name="axis"/>.
    /// </summary>
    public static (PaddleTensor Values, PaddleTensor Indices) TopK(
        PaddleTensor input,
        int k,
        int axis,
        bool largest,
        bool sorted)
    {
        int rank = input.Rank;
        axis = axis < 0 ? axis + rank : axis;

        int[] shape = [.. input.Shape];
        shape[axis] = k;

        PaddleTensor values = PaddleTensor.Allocate(shape, input.Dtype);
        PaddleTensor indices = PaddleTensor.Int(shape);

        int width = input.Shape[axis];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++)
        {
            inner *= input.Shape[i];
        }

        int outer = width == 0 ? 0 : input.Count / (width * inner);
        int[] order = new int[width];

        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                int sourceBase = (o * width * inner) + i;
                for (int w = 0; w < width; w++)
                {
                    order[w] = w;
                }

                // Paddle's top-k is stable: ties keep their original order.
                int[] ranked = [.. order];
                Array.Sort(ranked, (left, right) =>
                {
                    double a = input.GetDouble(sourceBase + (left * inner));
                    double b = input.GetDouble(sourceBase + (right * inner));
                    int comparison = largest ? b.CompareTo(a) : a.CompareTo(b);
                    return comparison != 0 ? comparison : left.CompareTo(right);
                });

                int targetBase = (o * k * inner) + i;
                for (int j = 0; j < k; j++)
                {
                    int source = sourceBase + (ranked[j] * inner);
                    int target = targetBase + (j * inner);
                    if (values.IsFloat)
                    {
                        values.Floats![target] = input.Floats![source];
                    }
                    else
                    {
                        values.Ints![target] = input.Ints![source];
                    }

                    indices.Ints![target] = ranked[j];
                }
            }
        }

        _ = sorted;
        return (values, indices);
    }

    /// <summary>Sorted values and their indices along <paramref name="axis"/>.</summary>
    public static (PaddleTensor Values, PaddleTensor Indices) ArgSort(
        PaddleTensor input,
        int axis,
        bool descending,
        bool stable)
    {
        int rank = input.Rank;
        axis = axis < 0 ? axis + rank : axis;
        int width = input.Shape[axis];
        (PaddleTensor values, PaddleTensor indices) = TopK(input, width, axis, descending, sorted: true);
        _ = stable;
        return (values, indices);
    }

    /// <summary>Element-wise selection: <c>condition ? x : y</c>, with broadcasting.</summary>
    public static PaddleTensor Where(PaddleTensor condition, PaddleTensor x, PaddleTensor y)
    {
        int[] shape = Broadcast.ResultShape(Broadcast.ResultShape(condition.Shape, x.Shape), y.Shape);
        PaddleTensor result = PaddleTensor.Allocate(shape, x.Dtype);

        int[] conditionStrides = Broadcast.StridesFor(condition.Shape, shape);
        int[] xStrides = Broadcast.StridesFor(x.Shape, shape);
        int[] yStrides = Broadcast.StridesFor(y.Shape, shape);

        // Three calls a detection over [1, 300, 200, 200], which is twelve million selections
        // apiece; each chunk seeds its own counters and walks from there.
        Parallelism.Chunked(result.Count, (start, end) =>
        {
            int[] counters = new int[shape.Length];
            Broadcast.Seed(start, shape, counters);
            int conditionOffset = Broadcast.Offset(counters, conditionStrides);
            int xOffset = Broadcast.Offset(counters, xStrides);
            int yOffset = Broadcast.Offset(counters, yStrides);

            for (int i = start; i < end; i++)
            {
                bool take = condition.GetLong(conditionOffset) != 0;
                if (result.IsFloat)
                {
                    result.Floats![i] = take ? x.Floats![xOffset] : y.Floats![yOffset];
                }
                else
                {
                    result.Ints![i] = take ? x.GetLong(xOffset) : y.GetLong(yOffset);
                }

                for (int axis = shape.Length - 1; axis >= 0; axis--)
                {
                    counters[axis]++;
                    conditionOffset += conditionStrides[axis];
                    xOffset += xStrides[axis];
                    yOffset += yStrides[axis];

                    if (counters[axis] < shape[axis])
                    {
                        break;
                    }

                    conditionOffset -= conditionStrides[axis] * shape[axis];
                    xOffset -= xStrides[axis] * shape[axis];
                    yOffset -= yStrides[axis] * shape[axis];
                    counters[axis] = 0;
                }
            }
        });

        return result;
    }
}
