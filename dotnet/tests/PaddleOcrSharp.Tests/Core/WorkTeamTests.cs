using System.Collections.Concurrent;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Tests.Core;

/// <summary>Tests for <see cref="Parallelism.For"/> and the persistent team behind it.</summary>
public class WorkTeamTests
{
    /// <summary>Every index runs exactly once, whatever the count and however often the team is reused.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(48)]
    [InlineData(1000)]
    public void EveryIndexRunsExactlyOnce(int count)
    {
        for (int round = 0; round < 200; round++)
        {
            int[] seen = new int[count];
            Parallelism.For(count, index => Interlocked.Increment(ref seen[index]));
            Assert.All(seen, hits => Assert.Equal(1, hits));
        }
    }

    /// <summary>An empty or negative range is a no-op.</summary>
    [Fact]
    public void AnEmptyRangeRunsNothing()
    {
        Parallelism.For(0, _ => throw new InvalidOperationException());
        Parallelism.For(-3, _ => throw new InvalidOperationException());
    }

    /// <summary>A region inside a region completes: the team is busy, so the inner one falls back.</summary>
    [Fact]
    public void ARegionInsideARegionCompletes()
    {
        int total = 0;
        Parallelism.For(8, _ => Parallelism.For(8, _ => Interlocked.Increment(ref total)));
        Assert.Equal(64, total);
    }

    /// <summary>
    /// Callers on several threads at once all get their work done — one holds the team and the
    /// others take the fallback, which is what <c>--block-concurrency</c> above one does.
    /// </summary>
    [Fact]
    public void ConcurrentCallersAllComplete()
    {
        var totals = new ConcurrentBag<int>();
        Task[] callers = Enumerable.Range(0, 4).Select(_ => Task.Run(() =>
        {
            for (int round = 0; round < 50; round++)
            {
                int sum = 0;
                Parallelism.For(256, index => Interlocked.Add(ref sum, index));
                totals.Add(sum);
            }
        })).ToArray();

        Task.WaitAll(callers);
        Assert.Equal(200, totals.Count);
        Assert.All(totals, sum => Assert.Equal(255 * 256 / 2, sum));
    }

    /// <summary>What a body throws surfaces as an <see cref="AggregateException"/>, as from <c>Parallel.For</c>.</summary>
    [Fact]
    public void ABodyExceptionSurfacesAsAnAggregate()
    {
        AggregateException thrown = Assert.Throws<AggregateException>(() =>
            Parallelism.For(64, index =>
            {
                if (index == 3)
                {
                    throw new InvalidOperationException("boom");
                }
            }));

        Assert.IsType<InvalidOperationException>(Assert.Single(thrown.InnerExceptions));

        // And the team is still usable afterwards.
        int hits = 0;
        Parallelism.For(16, _ => Interlocked.Increment(ref hits));
        Assert.Equal(16, hits);
    }

    /// <summary>A cancelled token surfaces as <see cref="OperationCanceledException"/>.</summary>
    [Fact]
    public void CancellationSurfaces()
    {
        using var source = new CancellationTokenSource();
        source.Cancel();

        using (Parallelism.Use(new ParallelOptions { CancellationToken = source.Token, MaxDegreeOfParallelism = 4 }))
        {
            Assert.ThrowsAny<OperationCanceledException>(() => Parallelism.For(64, _ => { }));
        }
    }

    /// <summary>The caller's options reach the helper threads, so nested kernels see the same scope.</summary>
    [Fact]
    public void TheCallersScopeReachesTheHelpers()
    {
        var mine = new ParallelOptions { MaxDegreeOfParallelism = 3 };
        var seen = new ConcurrentBag<ParallelOptions>();

        using (Parallelism.Use(mine))
        {
            for (int round = 0; round < 50; round++)
            {
                Parallelism.For(32, _ => seen.Add(Parallelism.Options));
            }
        }

        Assert.Equal(50 * 32, seen.Count);
        Assert.All(seen, options => Assert.Same(mine, options));
    }

    /// <summary>A single worker runs the region inline and still covers every index.</summary>
    [Fact]
    public void OneWorkerStillCoversTheRange()
    {
        int[] seen = new int[100];
        using (Parallelism.Use(new ParallelOptions { MaxDegreeOfParallelism = 1 }))
        {
            Parallelism.For(100, index => seen[index]++);
        }

        Assert.All(seen, hits => Assert.Equal(1, hits));
    }
}
