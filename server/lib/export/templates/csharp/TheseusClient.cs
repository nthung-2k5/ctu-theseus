// Theseus-generated inference client.
//
// Wraps the exported ONNX model with the preprocessing/postprocessing Ludwig
// applied at training time, read from preprocessing.json — Ludwig's exported
// ONNX graph is the bare model only (no resize/normalize, no class-index
// decoding), so this client reconstructs that step from the manifest rather
// than assuming a fixed tensor layout: input/output tensor names are matched
// against the graph's actual names at runtime, not hardcoded, since ONNX
// export doesn't guarantee a stable naming convention across Ludwig versions.
//
// Currently implements preprocessing for `image` inputs with `category`
// outputs (image classification) — the modality this export path has been
// verified against. Other input types throw with a pointer to
// preprocessing.json so you can adapt preprocessing yourself; see README.md.

using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace TheseusExport;

internal sealed class ImageNormalization
{
    public float[] Mean { get; set; } = [];
    public float[] Std { get; set; } = [];
}

internal sealed class PreprocessingInput
{
    public string Type { get; set; } = "";
    public string Column { get; set; } = "";
    public Dictionary<string, JsonElement>? LudwigPreprocessing { get; set; }
    public ImageNormalization? ImageNormalization { get; set; }
}

internal sealed class PreprocessingOutput
{
    public string Column { get; set; } = "";
    public List<string>? Classes { get; set; }
}

internal sealed class PreprocessingManifest
{
    public List<PreprocessingInput> Inputs { get; set; } = [];
    public List<PreprocessingOutput> Outputs { get; set; } = [];
}

public sealed class TheseusClient : IDisposable
{
    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    private readonly InferenceSession _session;
    private readonly PreprocessingInput _inputSpec;
    private readonly PreprocessingOutput _outputSpec;
    private readonly string _onnxInputName;
    private readonly string _onnxOutputName;

    public TheseusClient(string? modelPath = null, string? preprocessingPath = null)
    {
        var here = AppContext.BaseDirectory;
        modelPath ??= Path.Combine(here, "model.onnx");
        preprocessingPath ??= Path.Combine(here, "preprocessing.json");

        var manifest = JsonSerializer.Deserialize<PreprocessingManifest>(File.ReadAllText(preprocessingPath), JsonOpts)
            ?? throw new InvalidDataException("preprocessing.json is empty or invalid");
        _inputSpec = manifest.Inputs[0];
        _outputSpec = manifest.Outputs[0];

        _session = new InferenceSession(modelPath);
        _onnxInputName = MatchTensorName(_session.InputMetadata.Keys, _inputSpec.Column);
        _onnxOutputName = MatchTensorName(_session.OutputMetadata.Keys, _outputSpec.Column);
    }

    private static string MatchTensorName(IEnumerable<string> candidates, string preferred)
    {
        var list = candidates.ToList();
        if (list.Count == 1) return list[0];
        var match = list.FirstOrDefault(n => n == preferred || n.StartsWith(preferred + "::") || n.StartsWith(preferred + "_"));
        if (match is null)
        {
            throw new InvalidOperationException(
                $"Could not match an ONNX tensor to feature '{preferred}' among [{string.Join(", ", list)}]. " +
                "Inspect preprocessing.json and adapt this client if the model uses an unrecognized naming scheme.");
        }
        return match;
    }

    private (int Width, int Height) GetImageDims()
    {
        int width = 224, height = 224;
        if (_inputSpec.LudwigPreprocessing is { } pp)
        {
            if (pp.TryGetValue("width", out var w) && w.ValueKind == JsonValueKind.Number) width = w.GetInt32();
            if (pp.TryGetValue("height", out var h) && h.ValueKind == JsonValueKind.Number) height = h.GetInt32();
        }
        return (width, height);
    }

    private DenseTensor<float> PreprocessImage(string path)
    {
        var (width, height) = GetImageDims();
        using var image = Image.Load<Rgb24>(path);
        image.Mutate(x => x.Resize(width, height));

        var norm = _inputSpec.ImageNormalization;
        var data = new float[3 * height * width];
        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (var x = 0; x < width; x++)
                {
                    var px = row[x];
                    float r = px.R / 255f, g = px.G / 255f, b = px.B / 255f;
                    if (norm is not null)
                    {
                        r = (r - norm.Mean[0]) / norm.Std[0];
                        g = (g - norm.Mean[1]) / norm.Std[1];
                        b = (b - norm.Mean[2]) / norm.Std[2];
                    }
                    data[(0 * height * width) + (y * width) + x] = r;
                    data[(1 * height * width) + (y * width) + x] = g;
                    data[(2 * height * width) + (y * width) + x] = b;
                }
            }
        });
        return new DenseTensor<float>(data, [1, 3, height, width]);
    }

    private static float[] Softmax(float[] logits)
    {
        var max = logits.Max();
        var exp = logits.Select(v => MathF.Exp(v - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(v => v / sum).ToArray();
    }

    /// <summary>Runs inference and returns class -&gt; probability (classification, sorted descending) or {"value": x} (regression).</summary>
    public Dictionary<string, double> Predict(string imagePath, int topK = 5)
    {
        if (_inputSpec.Type != "image")
        {
            throw new NotSupportedException(
                $"No preprocessing implemented for input type '{_inputSpec.Type}'. " +
                "See preprocessing.json's inputs[0].ludwigPreprocessing and adapt this client.");
        }

        var tensor = PreprocessImage(imagePath);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_onnxInputName, tensor) };
        using var results = _session.Run(inputs);
        var logits = results.First(r => r.Name == _onnxOutputName).AsEnumerable<float>().ToArray();

        var classes = _outputSpec.Classes;
        if (classes is { Count: > 0 } && classes.Count == logits.Length)
        {
            var probs = Softmax(logits);
            return classes
                .Select((name, i) => (name, prob: probs[i]))
                .OrderByDescending(p => p.prob)
                .Take(topK)
                .ToDictionary(p => p.name, p => (double)MathF.Round(p.prob, 4));
        }

        return new Dictionary<string, double> { ["value"] = logits[0] };
    }

    public void Dispose() => _session.Dispose();
}
