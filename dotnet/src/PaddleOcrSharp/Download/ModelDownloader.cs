using System.Net;
using System.Net.Http.Headers;

namespace PaddleOcrSharp.Download;

/// <summary>
/// Fetches model files from a static HTTP mirror into a local cache.
/// </summary>
/// <remarks>
/// <para>
/// A file's URL is <c>{endpoint}/{remote path}/{file path}</c> — the mirror is a plain directory
/// tree, so <c>https://models.curiosity.ai/paddleocr-vl/config.json</c> is the whole address of
/// that file. Nothing here is specific to a model host's API.
/// </para>
/// <para>
/// Transfers resume from a <c>.part</c> file, are verified against the length the server reports,
/// and are published by an atomic rename so a killed process never leaves a truncated file that
/// later looks complete. A cross-process lock file makes concurrent CLI invocations safe.
/// </para>
/// <para>
/// The endpoint can be redirected with <c>PADDLEOCR_SHARP_MODELS_URL</c> for a private or local
/// mirror, and the cache root with <c>PADDLEOCR_SHARP_CACHE</c>.
/// </para>
/// </remarks>
public sealed class ModelDownloader : IDisposable
{
    private const string DefaultEndpoint = "https://models.curiosity.ai";
    private const string EndpointVariable = "PADDLEOCR_SHARP_MODELS_URL";
    private const string TokenVariable = "PADDLEOCR_SHARP_MODELS_TOKEN";

    private readonly HttpClient _client;
    private readonly bool _ownsClient;

    /// <summary>Creates a downloader with its own <see cref="HttpClient"/>.</summary>
    /// <param name="cacheRoot">Cache directory; defaults to the user cache.</param>
    /// <param name="endpoint">
    /// Base URL; defaults to <c>PADDLEOCR_SHARP_MODELS_URL</c> or models.curiosity.ai.
    /// </param>
    /// <param name="token">
    /// Optional bearer token, for a mirror that is not public. The default endpoint needs none.
    /// </param>
    public ModelDownloader(string? cacheRoot = null, string? endpoint = null, string? token = null)
        : this(CreateClient(token), ownsClient: true, cacheRoot, endpoint)
    {
    }

    /// <summary>Creates a downloader over a caller-owned <see cref="HttpClient"/>.</summary>
    public ModelDownloader(HttpClient client, bool ownsClient, string? cacheRoot = null, string? endpoint = null)
    {
        _client = client;
        _ownsClient = ownsClient;
        CacheRoot = cacheRoot ?? DefaultCacheRoot();
        Endpoint = (endpoint
            ?? Environment.GetEnvironmentVariable(EndpointVariable)
            ?? DefaultEndpoint).TrimEnd('/');
    }

    /// <summary>Root of the local model cache.</summary>
    public string CacheRoot { get; }

    /// <summary>Base URL files are fetched from.</summary>
    public string Endpoint { get; }

    /// <summary>Directory a model resolves to, whether or not it has been downloaded.</summary>
    public string DirectoryFor(ModelDescriptor model) => Path.Combine(CacheRoot, model.Name);

    /// <summary>Whether every required file of <paramref name="model"/> is already present.</summary>
    public bool IsComplete(ModelDescriptor model)
    {
        string directory = DirectoryFor(model);
        return model.Files
            .Where(file => file.Required)
            .All(file => File.Exists(Path.Combine(directory, file.Path)));
    }

    /// <summary>
    /// Ensures every file of <paramref name="model"/> is present and returns its directory.
    /// </summary>
    public async Task<string> EnsureAsync(
        ModelDescriptor model,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        string directory = DirectoryFor(model);
        Directory.CreateDirectory(directory);

        using FileStream guard = await AcquireLockAsync(directory, cancellationToken).ConfigureAwait(false);

        foreach (ModelFile file in model.Files)
        {
            try
            {
                await FetchAsync(model, file, directory, progress, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception exception) when (!file.Required && exception is not OperationCanceledException)
            {
                // Optional files (chat templates, generation configs) are absent from some
                // revisions; the model still loads without them.
            }
        }

        return directory;
    }

    /// <summary>The address <paramref name="file"/> of <paramref name="model"/> is fetched from.</summary>
    public string UrlFor(ModelDescriptor model, ModelFile file) =>
        $"{Endpoint}/{file.RemotePath ?? model.RemotePath}/{file.Path}";

    private async Task FetchAsync(
        ModelDescriptor model,
        ModelFile file,
        string directory,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        string target = Path.Combine(directory, file.Path);
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);

        string url = UrlFor(model, file);
        RemoteFileInfo? remote = await HeadAsync(url, cancellationToken).ConfigureAwait(false);

        if (File.Exists(target))
        {
            var existing = new FileInfo(target);
            bool sizeMatches = remote?.Length is null || remote.Value.Length == existing.Length;
            if (sizeMatches)
            {
                progress?.Report(new DownloadProgress(
                    model.Name, file.Path, existing.Length, existing.Length, Cached: true));
                return;
            }

            File.Delete(target);
        }

        if (remote is null)
        {
            throw new HttpRequestException($"'{url}' is not reachable and no cached copy exists.");
        }

        string partial = target + ".part";
        long offset = File.Exists(partial) ? new FileInfo(partial).Length : 0;
        if (offset > 0 && remote.Value.Length is { } total && (offset > total || !remote.Value.SupportsRange))
        {
            File.Delete(partial);
            offset = 0;
        }

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (offset > 0)
        {
            request.Headers.Range = new RangeHeaderValue(offset, null);

            // If the mirror's copy changed since the .part was written, appending to it would
            // splice two different files into one of exactly the right length — which the size
            // check below cannot see. If-Range makes the server answer 200 with the whole body
            // instead, and the stale partial content is discarded a few lines down.
            if (remote.Value.Validator is { } validator)
            {
                request.Headers.IfRange = new RangeConditionHeaderValue(validator);
            }
        }

        using HttpResponseMessage response = await _client
            .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);

        if (offset > 0 && response.StatusCode != HttpStatusCode.PartialContent)
        {
            // The server ignored the range; start over rather than concatenating a full body
            // onto the partial file.
            offset = 0;
        }

        response.EnsureSuccessStatusCode();

        await using (FileStream output = new(
            partial,
            offset > 0 ? FileMode.Append : FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 1 << 20,
            useAsync: true))
        {
            await using Stream input = await response.Content
                .ReadAsStreamAsync(cancellationToken)
                .ConfigureAwait(false);

            byte[] buffer = new byte[1 << 20];
            long received = offset;
            int read;
            while ((read = await input.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                received += read;
                progress?.Report(new DownloadProgress(
                    model.Name, file.Path, received, remote.Value.Length, Cached: false));
            }
        }

        if (remote.Value.Length is { } expected && new FileInfo(partial).Length != expected)
        {
            long actual = new FileInfo(partial).Length;
            File.Delete(partial);
            throw new IOException($"'{file.Path}' downloaded {actual} bytes but {expected} were expected.");
        }

        if (IsMissingObjectMarker(partial))
        {
            File.Delete(partial);
            throw new HttpRequestException($"'{url}' does not exist on the mirror.");
        }

        File.Move(partial, target, overwrite: true);
    }

    /// <summary>
    /// Whether <paramref name="path"/> holds the mirror's answer for an object it does not have.
    /// The mirror's front end replies <c>200</c> with the text <c>Object Not Found</c> rather than
    /// <c>404</c>, so a status code cannot tell a missing file from a present one, and without this
    /// an absent optional file would be cached as a sixteen-byte text file the loader then fails
    /// to parse.
    /// </summary>
    private static bool IsMissingObjectMarker(string path)
    {
        ReadOnlySpan<byte> marker = "Object Not Found"u8;
        if (new FileInfo(path).Length != marker.Length)
        {
            return false;
        }

        Span<byte> head = stackalloc byte[16];
        using FileStream stream = File.OpenRead(path);
        return stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false) == head.Length
            && head.SequenceEqual(marker);
    }

    private async Task<RemoteFileInfo?> HeadAsync(string url, CancellationToken cancellationToken)
    {
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Head, url);
            using HttpResponseMessage response = await _client
                .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                .ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                // Not every static mirror answers HEAD: an R2 or S3 bucket published through a
                // worker often allows only GET, PUT and DELETE and replies 405 here. A one-byte
                // ranged GET carries the same three facts in its Content-Range, so the probe
                // falls back to that rather than reporting the file unreachable.
                return response.StatusCode is HttpStatusCode.MethodNotAllowed or HttpStatusCode.Forbidden
                    ? await ProbeByRangeAsync(url, cancellationToken).ConfigureAwait(false)
                    : null;
            }

            // A static mirror publishes no content digest: an S3-compatible ETag is an MD5
            // only for a single-part upload, and a CDN in front of it may rewrite the header
            // altogether. So the ETag is kept as what it reliably is — an opaque validator for
            // a conditional request — and length is what the finished file is checked against.
            // A weak tag cannot be used with If-Range, so it is dropped.
            EntityTagHeaderValue? etag = response.Headers.ETag is { IsWeak: false } strong ? strong : null;

            return new RemoteFileInfo(
                response.Content.Headers.ContentLength,
                etag,
                response.Headers.AcceptRanges.Contains("bytes"));
        }
        catch (HttpRequestException)
        {
            return null;
        }
        catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return null;
        }
    }

    /// <summary>
    /// Asks for the first byte of <paramref name="url"/> and reads the file's facts out of the
    /// <c>Content-Range</c> the server answers with. Used when the mirror rejects <c>HEAD</c>.
    /// </summary>
    private async Task<RemoteFileInfo?> ProbeByRangeAsync(string url, CancellationToken cancellationToken)
    {
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Get, url);
            request.Headers.Range = new RangeHeaderValue(0, 0);

            using HttpResponseMessage response = await _client
                .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                .ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                return null;
            }

            EntityTagHeaderValue? etag = response.Headers.ETag is { IsWeak: false } strong ? strong : null;

            // 206 means the range was honoured, so the total length is in Content-Range and the
            // server does support ranges whether or not it also advertises Accept-Ranges. A 200
            // means it ignored the range and sent the whole body, which Content-Length then
            // describes; resuming against such a server is not safe.
            if (response.StatusCode == HttpStatusCode.PartialContent)
            {
                return new RemoteFileInfo(response.Content.Headers.ContentRange?.Length, etag, SupportsRange: true);
            }

            return new RemoteFileInfo(response.Content.Headers.ContentLength, etag, SupportsRange: false);
        }
        catch (HttpRequestException)
        {
            return null;
        }
        catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return null;
        }
    }

    private static async Task<FileStream> AcquireLockAsync(string directory, CancellationToken cancellationToken)
    {
        string path = Path.Combine(directory, ".download.lock");
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                return new FileStream(
                    path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose);
            }
            catch (IOException)
            {
                await Task.Delay(200, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private static HttpClient CreateClient(string? token)
    {
        var handler = new SocketsHttpHandler
        {
            AutomaticDecompression = DecompressionMethods.All,
            PooledConnectionLifetime = TimeSpan.FromMinutes(5),
        };

        var client = new HttpClient(handler)
        {
            // Large weight files stream for minutes; the per-read cancellation token is what
            // actually bounds a stalled transfer.
            Timeout = Timeout.InfiniteTimeSpan,
        };

        client.DefaultRequestHeaders.UserAgent.ParseAdd("PaddleOcrSharp/0.1");

        string? bearer = token ?? Environment.GetEnvironmentVariable(TokenVariable);
        if (!string.IsNullOrEmpty(bearer))
        {
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", bearer);
        }

        return client;
    }

    private static string DefaultCacheRoot()
    {
        string? configured = Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_CACHE");
        if (!string.IsNullOrEmpty(configured))
        {
            return configured;
        }

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string cache = Environment.GetEnvironmentVariable("XDG_CACHE_HOME")
            ?? Path.Combine(home, ".cache");
        return Path.Combine(cache, "paddleocr-sharp");
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsClient)
        {
            _client.Dispose();
        }
    }

    private readonly record struct RemoteFileInfo(
        long? Length,
        EntityTagHeaderValue? Validator,
        bool SupportsRange);
}
