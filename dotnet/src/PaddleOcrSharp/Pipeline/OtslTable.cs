using System.Net;
using System.Text;
using System.Text.RegularExpressions;

namespace PaddleOcrSharp.Pipeline;

/// <summary>
/// Converts the OTSL v1.0 table markup the VL model emits into HTML.
/// </summary>
/// <remarks>
/// <para>
/// OTSL describes a table as a stream of six tags: <c>&lt;fcel&gt;</c> opens a filled cell whose
/// text follows, <c>&lt;ecel&gt;</c> an empty one, <c>&lt;lcel&gt;</c> continues the cell to its
/// left, <c>&lt;ucel&gt;</c> the cell above, <c>&lt;xcel&gt;</c> both, and <c>&lt;nl&gt;</c> ends
/// a row. Spans are recovered by counting how far the continuation tags run.
/// </para>
/// <para>Port of <c>convert_otsl_to_html</c> and friends in PaddleX's <c>uilts.py</c>.</para>
/// </remarks>
public static partial class OtslTable
{
    private const string NewLine = "<nl>";
    private const string FilledCell = "<fcel>";
    private const string EmptyCell = "<ecel>";
    private const string LeftCell = "<lcel>";
    private const string UpCell = "<ucel>";
    private const string CrossCell = "<xcel>";

    private static readonly string[] CellTags = [FilledCell, EmptyCell, LeftCell, UpCell, CrossCell];
    private static readonly string[] AllTags = [NewLine, FilledCell, EmptyCell, LeftCell, UpCell, CrossCell];

    [GeneratedRegex(@"(<nl>|<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)")]
    private static partial Regex TagPattern();

    [GeneratedRegex(
        @"(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>).*?(?=<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>|$)",
        RegexOptions.Singleline)]
    private static partial Regex CellPattern();

    /// <summary>
    /// Converts <paramref name="otsl"/> to an HTML <c>&lt;table&gt;</c>, or returns an empty
    /// string when the markup holds no cells.
    /// </summary>
    public static string ToHtml(string otsl)
    {
        if (string.IsNullOrWhiteSpace(otsl))
        {
            return string.Empty;
        }

        string padded = PadToRectangle(otsl);
        (List<string> tokens, List<string> parts) = ExtractTokensAndText(padded);
        List<List<string>> rows = SplitRows(tokens);

        if (rows.Count == 0)
        {
            return string.Empty;
        }

        List<Cell> cells = ParseCells(parts, rows);
        if (cells.Count == 0)
        {
            return string.Empty;
        }

        int columns = rows.Max(row => row.Count);
        return Export(cells, rows.Count, columns);
    }

    private static (List<string> Tokens, List<string> Parts) ExtractTokensAndText(string otsl)
    {
        var tokens = new List<string>();
        foreach (Match match in TagPattern().Matches(otsl))
        {
            tokens.Add(match.Value);
        }

        var parts = new List<string>();
        foreach (string piece in TagPattern().Split(otsl))
        {
            if (!string.IsNullOrWhiteSpace(piece))
            {
                parts.Add(piece);
            }
        }

        return (tokens, parts);
    }

    private static List<List<string>> SplitRows(List<string> tokens)
    {
        var rows = new List<List<string>>();
        var current = new List<string>();

        foreach (string token in tokens)
        {
            if (token == NewLine)
            {
                if (current.Count > 0)
                {
                    rows.Add(current);
                    current = [];
                }
            }
            else
            {
                current.Add(token);
            }
        }

        if (current.Count > 0)
        {
            rows.Add(current);
        }

        return rows;
    }

    /// <summary>
    /// Pads every row to a common width, choosing the width that needs the fewest edits.
    /// </summary>
    /// <remarks>
    /// The model sometimes emits ragged rows. Upstream searches widths from the longest prefix
    /// that still ends in a filled cell up to the longest row, and keeps the width with the
    /// smallest total deviation.
    /// </remarks>
    private static string PadToRectangle(string otsl)
    {
        otsl = otsl.Trim();
        if (!otsl.Contains(NewLine, StringComparison.Ordinal))
        {
            return otsl + NewLine;
        }

        var rows = new List<(string[] Cells, int Total, int Minimum)>();
        foreach (string line in otsl.Split(NewLine))
        {
            if (line.Length == 0)
            {
                continue;
            }

            string[] cells = [.. CellPattern().Matches(line).Select(match => match.Value)];
            if (cells.Length == 0)
            {
                continue;
            }

            int minimum = 0;
            for (int i = 0; i < cells.Length; i++)
            {
                if (cells[i].StartsWith(FilledCell, StringComparison.Ordinal))
                {
                    minimum = i + 1;
                }
            }

            rows.Add((cells, cells.Length, minimum));
        }

        if (rows.Count == 0)
        {
            return NewLine;
        }

        int searchStart = rows.Max(row => row.Minimum);
        int searchEnd = Math.Max(searchStart, rows.Max(row => row.Total));

        int best = searchEnd;
        long bestCost = long.MaxValue;
        for (int width = searchStart; width <= searchEnd; width++)
        {
            long cost = rows.Sum(row => (long)Math.Abs(row.Total - width));
            if (cost < bestCost)
            {
                bestCost = cost;
                best = width;
            }
        }

        var builder = new StringBuilder();
        foreach ((string[] cells, _, _) in rows)
        {
            if (cells.Length > best)
            {
                foreach (string cell in cells.Take(best))
                {
                    builder.Append(cell);
                }
            }
            else
            {
                foreach (string cell in cells)
                {
                    builder.Append(cell);
                }

                for (int i = cells.Length; i < best; i++)
                {
                    builder.Append(EmptyCell);
                }
            }

            builder.Append(NewLine);
        }

        return builder.ToString();
    }

    private static List<Cell> ParseCells(List<string> parts, List<List<string>> rows)
    {
        int maxColumns = rows.Max(row => row.Count);
        foreach (List<string> row in rows)
        {
            while (row.Count < maxColumns)
            {
                row.Add(EmptyCell);
            }
        }

        // Re-thread the text fragments through the padded grid so a cell's text still follows its
        // opening tag after padding inserted empty cells.
        var stream = new List<string>();
        int index = 0;
        foreach (List<string> row in rows)
        {
            foreach (string token in row)
            {
                stream.Add(token);
                if (index < parts.Count && parts[index] == token)
                {
                    index++;
                    if (index < parts.Count && !AllTags.Contains(parts[index]))
                    {
                        stream.Add(parts[index]);
                        index++;
                    }
                }
            }

            stream.Add(NewLine);
            if (index < parts.Count && parts[index] == NewLine)
            {
                index++;
            }
        }

        var cells = new List<Cell>();
        int rowIndex = 0;
        int columnIndex = 0;

        for (int i = 0; i < stream.Count; i++)
        {
            string token = stream[i];

            if (token is FilledCell or EmptyCell)
            {
                string text = string.Empty;
                int rightOffset = 1;

                if (token != EmptyCell && i + 1 < stream.Count)
                {
                    text = stream[i + 1];
                    rightOffset = 2;
                }

                int rowSpan = 1;
                int columnSpan = 1;

                string nextRight = i + rightOffset < stream.Count ? stream[i + rightOffset] : string.Empty;
                string nextBottom = rowIndex + 1 < rows.Count && columnIndex < rows[rowIndex + 1].Count
                    ? rows[rowIndex + 1][columnIndex]
                    : string.Empty;

                if (nextRight is LeftCell or CrossCell)
                {
                    columnSpan += CountRight(rows, rowIndex, columnIndex + 1);
                }

                if (nextBottom is UpCell or CrossCell)
                {
                    rowSpan += CountDown(rows, rowIndex + 1, columnIndex);
                }

                cells.Add(new Cell(rowIndex, columnIndex, rowSpan, columnSpan, text.Trim()));
            }

            if (CellTags.Contains(token))
            {
                columnIndex++;
            }
            else if (token == NewLine)
            {
                rowIndex++;
                columnIndex = 0;
            }
        }

        return cells;
    }

    private static int CountRight(List<List<string>> rows, int row, int column)
    {
        int span = 0;
        while (row < rows.Count && column < rows[row].Count && rows[row][column] is LeftCell or CrossCell)
        {
            column++;
            span++;
        }

        return span;
    }

    private static int CountDown(List<List<string>> rows, int row, int column)
    {
        int span = 0;
        while (row < rows.Count && column < rows[row].Count && rows[row][column] is UpCell or CrossCell)
        {
            row++;
            span++;
        }

        return span;
    }

    private static string Export(List<Cell> cells, int rowCount, int columnCount)
    {
        Cell?[,] grid = new Cell?[rowCount, columnCount];
        foreach (Cell cell in cells)
        {
            for (int r = cell.Row; r < Math.Min(cell.Row + cell.RowSpan, rowCount); r++)
            {
                for (int c = cell.Column; c < Math.Min(cell.Column + cell.ColumnSpan, columnCount); c++)
                {
                    grid[r, c] = cell;
                }
            }
        }

        var builder = new StringBuilder("<table>");
        for (int r = 0; r < rowCount; r++)
        {
            builder.Append("<tr>");
            for (int c = 0; c < columnCount; c++)
            {
                Cell? cell = grid[r, c];
                if (cell is null)
                {
                    builder.Append("<td></td>");
                    continue;
                }

                if (cell.Row != r || cell.Column != c)
                {
                    continue;
                }

                builder.Append("<td");
                if (cell.RowSpan > 1)
                {
                    builder.Append(" rowspan=\"").Append(cell.RowSpan).Append('"');
                }

                if (cell.ColumnSpan > 1)
                {
                    builder.Append(" colspan=\"").Append(cell.ColumnSpan).Append('"');
                }

                builder.Append('>').Append(EncodeCellText(cell.Text)).Append("</td>");
            }

            builder.Append("</tr>");
        }

        return builder.Append("</table>").ToString();
    }

    /// <summary>
    /// HTML-encodes a cell's text, writing each line break inside it as <c>&lt;br&gt;</c>.
    /// </summary>
    /// <remarks>
    /// Deliberate deviation from upstream, which leaves the raw line feed in the <c>td</c>. HTML
    /// treats it as a space, and a consumer that turns the table into another format (a markdown
    /// pipe table, a cell grid) loses the break or, worse, has to escape it. <c>&lt;br&gt;</c>
    /// is the one spelling of a line break every HTML reader agrees on.
    /// </remarks>
    private static string EncodeCellText(string text) =>
        WebUtility.HtmlEncode(text.Trim())
            .Replace("\r\n", "<br>", StringComparison.Ordinal)
            .Replace("\n", "<br>", StringComparison.Ordinal)
            .Replace("\r", "<br>", StringComparison.Ordinal);

    private sealed record Cell(int Row, int Column, int RowSpan, int ColumnSpan, string Text);
}
