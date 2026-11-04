using PaddleOcrSharp.Pipeline;

namespace PaddleOcrSharp.Tests.Pipeline;

/// <summary>Checks the OTSL to HTML conversion against hand-worked examples.</summary>
public class OtslTableTests
{
    [Fact]
    public void ConvertsASimpleGrid()
    {
        string html = OtslTable.ToHtml("<fcel>A<fcel>B<nl><fcel>1<fcel>2<nl>");

        Assert.Equal(
            "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>",
            html);
    }

    [Fact]
    public void EmptyCellsBecomeEmptyTableCells()
    {
        string html = OtslTable.ToHtml("<fcel>A<ecel><nl><ecel><fcel>D<nl>");

        Assert.Equal(
            "<table><tr><td>A</td><td></td></tr><tr><td></td><td>D</td></tr></table>",
            html);
    }

    [Fact]
    public void LeftContinuationBecomesColspan()
    {
        string html = OtslTable.ToHtml("<fcel>Wide<lcel><nl><fcel>1<fcel>2<nl>");

        Assert.Contains("colspan=\"2\"", html);
        Assert.Contains(">Wide<", html);
    }

    [Fact]
    public void UpContinuationBecomesRowspan()
    {
        string html = OtslTable.ToHtml("<fcel>Tall<fcel>B<nl><ucel><fcel>C<nl>");

        Assert.Contains("rowspan=\"2\"", html);
        Assert.Contains(">Tall<", html);
    }

    [Fact]
    public void CrossContinuationSpansBothAxes()
    {
        string html = OtslTable.ToHtml("<fcel>Big<lcel><nl><ucel><xcel><nl>");

        Assert.Contains("rowspan=\"2\"", html);
        Assert.Contains("colspan=\"2\"", html);
    }

    [Fact]
    public void RaggedRowsArePaddedToARectangle()
    {
        string html = OtslTable.ToHtml("<fcel>A<fcel>B<fcel>C<nl><fcel>1<nl>");

        // Both rows must end up with the same cell count.
        int firstRowCells = CountCells(html, 0);
        int secondRowCells = CountCells(html, 1);
        Assert.Equal(firstRowCells, secondRowCells);
    }

    [Fact]
    public void ContentIsHtmlEscaped()
    {
        string html = OtslTable.ToHtml("<fcel>a &amp; b<fcel>x < y<nl>");

        Assert.DoesNotContain("<td>a & b</td>", html);
        Assert.Contains("&amp;", html);
    }

    [Fact]
    public void LineBreaksInsideACellBecomeBr()
    {
        string html = OtslTable.ToHtml("<fcel>Name<fcel>Address<nl><fcel>Bob<fcel>1 Main St\nSpringfield\r\nUSA<nl>");

        Assert.Equal(
            "<table><tr><td>Name</td><td>Address</td></tr><tr><td>Bob</td><td>1 Main St<br>Springfield<br>USA</td></tr></table>",
            html);
        Assert.DoesNotContain('\n', html);
    }

    [Fact]
    public void MarkupInACellIsEncodedBeforeBreaksAreWritten()
    {
        string html = OtslTable.ToHtml("<fcel>a<b\nc<nl>");

        Assert.Contains("<td>a&lt;b<br>c</td>", html);
    }

    [Fact]
    public void EmptyInputProducesNoTable()
    {
        Assert.Equal(string.Empty, OtslTable.ToHtml(string.Empty));
        Assert.Equal(string.Empty, OtslTable.ToHtml("   "));
    }

    private static int CountCells(string html, int rowIndex)
    {
        string[] rows = html.Split("<tr>", StringSplitOptions.RemoveEmptyEntries);
        string row = rows[rowIndex + 1];
        return row.Split("<td").Length - 1;
    }
}
