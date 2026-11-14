namespace PaddleOcrSharp.Core;

/// <summary>
/// Switches for the attention experiments. Read once from the environment so one binary can be
/// A/B'd against itself with an interleaved driver; frozen to constants when the experiments end.
/// </summary>
internal static class AttentionTuning
{
    private static int Read(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out int value) ? value : fallback;

    /// <summary>Query rows per block.</summary>
    internal static readonly int Block = Read("ATT_BLOCK", 16);

    /// <summary>Key columns per score tile.</summary>
    internal static readonly int ScoreColumns = Read("ATT_SCOLS", 128);

    /// <summary>Tokens per value chunk.</summary>
    internal static readonly int ChunkTokens = Read("ATT_CHUNK", 80);

    /// <summary>Whether the transposed keys are stored tile-major.</summary>
    internal static readonly bool TileMajor = Read("ATT_KT", 0) == 1;

    /// <summary>0 = 4x2, 1 = 8x2, 2 = 4x3, 3 = 6x2.</summary>
    internal static readonly int Tile = Read("ATT_TILE", 0);

    /// <summary>Whether each row's maximum is taken per tile, while the tile is still in L1.</summary>
    internal static readonly bool TileMax = Read("ATT_TILEMAX", 0) == 1;
}
