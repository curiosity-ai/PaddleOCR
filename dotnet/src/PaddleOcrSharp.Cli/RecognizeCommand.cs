using System.Diagnostics;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models;
using PaddleOcrSharp.Pipeline;

namespace PaddleOcrSharp.Cli;

/// <summary>Runs the VL model on one already-cropped block.</summary>
public static class RecognizeCommand
{
    /// <summary>Runs the <c>recognize</c> verb.</summary>
    public static async Task<int> RunAsync(CommandLine command)
    {
        if (command.Positional.Count == 0)
        {
            Console.Error.WriteLine("recognize needs an image path.");
            return 2;
        }

        string imagePath = command.Positional[0];
        if (!File.Exists(imagePath))
        {
            Console.Error.WriteLine($"'{imagePath}' does not exist.");
            return 2;
        }

        using var cancellation = new CancellationTokenSource();
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            cancellation.Cancel();
        };

        string label = command.Get("prompt-label", "ocr")!.ToLowerInvariant();
        string instruction = command.Get("prompt") ?? label switch
        {
            "ocr" => BlockPrompt.Ocr,
            "table" => BlockPrompt.Table,
            "formula" => BlockPrompt.Formula,
            "chart" => BlockPrompt.Chart,
            "seal" => BlockPrompt.Seal,
            "spotting" => BlockPrompt.Spotting,
            _ => BlockPrompt.Ocr,
        };

        var preprocessing = VisionPreprocessorOptions.Default with
        {
            MinPixels = command.GetInt("min-pixels", BlockPrompt.DefaultMinPixels),
            MaxPixels = command.GetInt(
                "max-pixels",
                label == "spotting" ? BlockPrompt.SpottingMaxPixels : BlockPrompt.DefaultMaxPixels),
        };

        var generation = GenerationOptions.Default with
        {
            MaxNewTokens = command.GetInt("max-new-tokens", GenerationOptions.Default.MaxNewTokens),
            Temperature = command.GetFloat("temperature", 0f),
            TopP = command.GetFloat("top-p", 0f),
            RepetitionPenalty = command.GetFloat("repetition-penalty", 1f),
            Seed = command.GetInt("seed", 0),
        };

        string directory = await ModelLocator
            .ResolveVLAsync(command, allowDownload: true, cancellation.Token)
            .ConfigureAwait(false);

        var clock = Stopwatch.StartNew();
        using PaddleOcrVLModel model = PaddleOcrVLModel.Load(directory, ModelLocator.QuantizationFrom(command));
        TimeSpan loaded = clock.Elapsed;

        using RgbImage image = ImageIO.Load(imagePath);
        clock.Restart();
        RecognitionProfile? profile = command.GetBool("profile", false) ? new RecognitionProfile() : null;
        string text = model.Recognize(
            image, instruction, preprocessing, generation, profile, null, cancellation.Token);
        TimeSpan inference = clock.Elapsed;

        if (profile is not null)
        {
            Console.Error.WriteLine(profile.ToString());
        }

        string? output = command.Get("output");
        if (output is not null)
        {
            await File.WriteAllTextAsync(output, text, cancellation.Token).ConfigureAwait(false);
            Console.Error.WriteLine($"wrote {output}");
        }
        else
        {
            Console.WriteLine(text);
        }

        Console.Error.WriteLine(
            $"[{image.Width}x{image.Height}] load {loaded.TotalSeconds:F1}s, inference {inference.TotalSeconds:F1}s");

        return 0;
    }
}
