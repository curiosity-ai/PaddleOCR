using System.Diagnostics;

namespace PaddleOcrSharp.Core;

/// <summary>
/// A persistent team of helper threads that run the kernels' parallel regions, spinning briefly
/// between regions so that the next one starts with every worker already awake.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Parallel.For(int, int, ParallelOptions, Action{int})"/> hands a region to thread-pool workers that park as soon as they run
/// out of work, and waking a parked thread costs tens to hundreds of microseconds on a virtualised
/// machine. A matrix product over a few megabytes of weights lasts about as long as that, so on
/// a decode step — a hundred and thirty regions of 20-400 µs back to back — the late workers
/// were measured joining a region after the early ones had finished most of it: per-thread busy
/// times inside one 12 MB streaming region came out 388 / 395 / 186 / 117 µs, which is one
/// and a half threads' worth of work on four cores. Streaming the same bytes without a region
/// boundary every few hundred microseconds ran 57 GB/s against 29.
/// </para>
/// <para>
/// A helper that finishes spins for <see cref="SpinTicks"/> before it parks, which covers the gap
/// between consecutive regions of a step and the sampling between steps, and costs nothing once
/// the work stops. Items are claimed dynamically, in blocks when there are many, so a straggler
/// is absorbed by the others.
/// </para>
/// <para>
/// There is one team, and one caller at a time uses it. A caller that finds it busy — a region
/// nested inside a region, or a second block's tower running beside the first under
/// <c>--block-concurrency</c> — gets <see langword="false"/> from <see cref="TryRun"/> and falls
/// back to <see cref="Parallel.For(int, int, ParallelOptions, Action{int})"/>, so nothing waits on anything and nothing deadlocks.
/// </para>
/// </remarks>
internal static class WorkTeam
{
    /// <summary>How long an idle helper spins before parking: about 300 µs.</summary>
    private static readonly long SpinTicks = Stopwatch.Frequency * 3 / 10_000;

    private static readonly object Gate = new();
    private static readonly Helper[] Helpers = CreateHelpers();
    private static Region? _current;
    private static int _busy;

    private static Helper[] CreateHelpers()
    {
        int count = Math.Max(0, Environment.ProcessorCount - 1);
        var helpers = new Helper[count];
        for (int i = 0; i < count; i++)
        {
            helpers[i] = new Helper(i);
        }

        return helpers;
    }

    /// <summary>
    /// Runs <paramref name="body"/> for every index in <c>[0, count)</c> on up to
    /// <paramref name="workers"/> threads, the caller being one of them.
    /// </summary>
    /// <returns>
    /// <see langword="false"/>, having run nothing, when the team is already in use or has no
    /// helpers to offer.
    /// </returns>
    /// <param name="count">Number of items.</param>
    /// <param name="workers">Threads to use, including the caller; at least two.</param>
    /// <param name="options">The caller's options, which the helpers adopt while they run.</param>
    /// <param name="body">The work for one item.</param>
    internal static bool TryRun(int count, int workers, ParallelOptions options, Action<int> body)
    {
        int helpersWanted = Math.Min(workers - 1, Helpers.Length);
        if (helpersWanted <= 0 || Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
        {
            return false;
        }

        var region = new Region(count, helpersWanted, options, body);
        try
        {
            // Publish, then wake whoever parked. A helper that is spinning sees the new region on
            // its next poll; one that is parked is signalled. The publication is a full fence
            // because the wake below reads `_parked`, and a store followed by a load may be
            // reordered: a helper that parks concurrently re-reads `_current` after raising
            // `_parked` for the mirror-image reason, so one of the two always sees the other.
            Interlocked.Exchange(ref _current, region);
            for (int i = 0; i < helpersWanted; i++)
            {
                Helpers[i].Wake();
            }

            region.Work();

            // Every item has been claimed; wait for the ones still running on helpers. A plain
            // spin, not `SpinWait.SpinOnce`: that escalates to `Thread.Sleep(1)`, and a sleep of a
            // millisecond or more is far longer than the item being waited for.
            int spins = 0;
            while (Volatile.Read(ref region.Completed) < count)
            {
                Thread.SpinWait(4);
                if (++spins > 50_000)
                {
                    Thread.Yield();
                }
            }
        }
        finally
        {
            Volatile.Write(ref _current, null);
            Volatile.Write(ref _busy, 0);
        }

        region.Rethrow();
        return true;
    }

    /// <summary>One parallel region's shared state.</summary>
    private sealed class Region
    {
        private readonly int _count;
        private readonly int _grain;
        private readonly Action<int> _body;
        private readonly CancellationToken _token;
        private int _next;
        private int _failed;
        private Exception? _exception;

        internal Region(int count, int helpers, ParallelOptions options, Action<int> body)
        {
            _count = count;
            _body = body;
            _token = options.CancellationToken;
            Options = options;
            HelpersWanted = helpers;

            // One at a time while the region is small enough for that to be cheap, so a straggler
            // never holds a block of items the others could have taken; blocks beyond that.
            _grain = Math.Max(1, count / ((helpers + 1) * 16));
        }

        internal ParallelOptions Options { get; }

        internal int HelpersWanted { get; }

        internal int Completed;

        /// <summary>Claims and runs items until none are left.</summary>
        internal void Work()
        {
            while (true)
            {
                int start = Interlocked.Add(ref _next, _grain) - _grain;
                if (start >= _count)
                {
                    return;
                }

                int end = Math.Min(_count, start + _grain);
                for (int i = start; i < end; i++)
                {
                    if (Volatile.Read(ref _failed) == 0 && !_token.IsCancellationRequested)
                    {
                        try
                        {
                            _body(i);
                        }
                        catch (Exception exception)
                        {
                            if (Interlocked.CompareExchange(ref _failed, 1, 0) == 0)
                            {
                                _exception = exception;
                            }
                        }
                    }

                    Interlocked.Increment(ref Completed);
                }
            }
        }

        /// <summary>
        /// Raises what <see cref="Parallel.For(int, int, ParallelOptions, Action{int})"/> would:
        /// the token's cancellation as is, anything the body threw wrapped in an
        /// <see cref="AggregateException"/>.
        /// </summary>
        internal void Rethrow()
        {
            if (_exception is { } exception)
            {
                throw new AggregateException(exception);
            }

            _token.ThrowIfCancellationRequested();
        }
    }

    private sealed class Helper
    {
        private readonly int _index;
        private readonly ManualResetEventSlim _wake = new(false);
        private int _parked;

        internal Helper(int index)
        {
            _index = index;
            var thread = new Thread(Run)
            {
                IsBackground = true,
                Name = $"paddleocr-team-{index}",
            };
            thread.Start();
        }

        /// <summary>Wakes the helper if it is parked; a spinning helper needs no help.</summary>
        internal void Wake()
        {
            if (Volatile.Read(ref _parked) != 0)
            {
                _wake.Set();
            }
        }

        private void Run()
        {
            Region? last = null;
            long idleSince = Stopwatch.GetTimestamp();

            while (true)
            {
                Region? region = Volatile.Read(ref _current);
                if (region is not null && !ReferenceEquals(region, last))
                {
                    // Seen, whether or not this helper is wanted: a helper the region does not
                    // need must still be able to park rather than poll a region it ignores.
                    last = region;
                    if (_index < region.HelpersWanted)
                    {
                        ParallelOptions? previous = Parallelism.SetScoped(region.Options);
                        try
                        {
                            region.Work();
                        }
                        finally
                        {
                            Parallelism.SetScoped(previous);
                        }

                        idleSince = Stopwatch.GetTimestamp();
                        continue;
                    }
                }

                if (Stopwatch.GetTimestamp() - idleSince < SpinTicks)
                {
                    Thread.SpinWait(32);
                    continue;
                }

                // Park. Announce it, look once more so a region published in between is not
                // missed, then wait to be signalled.
                _wake.Reset();
                Interlocked.Exchange(ref _parked, 1);
                Region? again = Volatile.Read(ref _current);
                if (again is null || ReferenceEquals(again, last))
                {
                    _wake.Wait();
                }

                Volatile.Write(ref _parked, 0);
                idleSince = Stopwatch.GetTimestamp();
            }
        }
    }
}
