// OmniTrain-generated inference client.
//
// Wraps the exported ONNX model with the preprocessing/postprocessing Ludwig
// applied at training time, read from preprocessing.json — Ludwig's exported
// ONNX graph is the bare model only (no resize/normalize, no class-index
// decoding), so this client reconstructs that step from the manifest rather
// than assuming a fixed tensor layout: input/output tensor names are matched
// against the graph's actual names at runtime, not hardcoded, since ONNX
// export doesn't guarantee a stable naming convention across Ludwig versions.
//
// Implements preprocessing for `image` inputs with `category` outputs (image
// classification) and for a `detections` output (object detection, YOLO): the
// ONNX graph returns every candidate box, so for a detector this client does
// the confidence filter, per-class non-maximum suppression and the undoing of
// the letterbox, and `Detect()` returns the boxes. Other input types throw with
// a pointer to preprocessing.json so you can adapt preprocessing yourself; see
// README.md.

using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace OmniTrainExport;

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
    // Absent: squash the whole image to height x width.
    public Dictionary<string, JsonElement>? ImageResize { get; set; }
    public ImageNormalization? ImageNormalization { get; set; }
}

internal sealed class PreprocessingOutput
{
    public string Type { get; set; } = "";
    public string Column { get; set; } = "";
    public List<string>? Classes { get; set; }
    public Dictionary<string, JsonElement>? YoloOutput { get; set; }
}

/// <summary>A box as fractions (0..1) of the image: the top-left corner and the size.</summary>
public sealed record Box(double X, double Y, double W, double H);

/// <summary>One object found in an image.</summary>
public sealed record Detection(string Label, double Confidence, Box Box);

internal sealed class PreprocessingManifest
{
    public List<PreprocessingInput> Inputs { get; set; } = [];
    public List<PreprocessingOutput> Outputs { get; set; } = [];
}

public sealed class OmniTrainClient : IDisposable
{
    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    private readonly InferenceSession _session;
    private readonly PreprocessingInput _inputSpec;
    private readonly PreprocessingOutput _outputSpec;
    private readonly string _onnxInputName;
    private readonly string _onnxOutputName;

    /// <summary>An object detector: use <see cref="Detect"/>; <see cref="Predict"/> is for classifiers and regressors.</summary>
    public bool IsDetector => _outputSpec.Type == "detections";

    public OmniTrainClient(string? modelPath = null, string? preprocessingPath = null)
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
        string? mode = null;
        if (_inputSpec.ImageResize is { } resize && resize.TryGetValue("mode", out var m) && m.ValueKind == JsonValueKind.String)
            mode = m.GetString();

        if (mode is null)
        {
            image.Mutate(x => x.Resize(width, height)); // the whole image squashed to the target size
        }
        else if (mode == "shorter_edge_center_crop")
        {
            // Scale so the SHORTER edge is `size` (the other edge truncated to a whole pixel), then centre-crop a
            // size x size square: the same arithmetic as torchvision's Resize(size) + CenterCrop(size), which is
            // what the model was evaluated with. Triangle is bilinear.
            var size = Math.Min(width, height);
            var landscape = image.Width > image.Height;
            var newWidth = landscape ? (int)((long)size * image.Width / image.Height) : size;
            var newHeight = landscape ? size : (int)((long)size * image.Height / image.Width);
            // ToEven rounds .5 to the nearest even integer, as Python's round() does.
            var left = (int)Math.Round((newWidth - size) / 2.0, MidpointRounding.ToEven);
            var top = (int)Math.Round((newHeight - size) / 2.0, MidpointRounding.ToEven);
            image.Mutate(x => x.Resize(newWidth, newHeight, KnownResamplers.Triangle).Crop(new Rectangle(left, top, width, height)));
        }
        else
        {
            throw new NotSupportedException(
                $"Unsupported imageResize mode '{mode}'. See preprocessing.json's inputs[0].imageResize and adapt this client.");
        }

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

    private static double ThresholdFrom(Dictionary<string, JsonElement>? info, string key, double fallback) =>
        info is not null && info.TryGetValue(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : fallback;

    /// <summary>
    /// The image as the detector's input tensor, and what is needed to map its boxes back. For
    /// `imageResize.mode == "letterbox"`: scale so the LONGER edge is `size`, pad the shorter one, centred, with
    /// `padValue` grey up to a size x size square (the arithmetic of ultralytics.data.augment.LetterBox). RGB, 0..1,
    /// no mean/std.
    /// </summary>
    private (DenseTensor<float> Tensor, double Scale, int Left, int Top, int OriginalWidth, int OriginalHeight) Letterbox(string path)
    {
        string? mode = null;
        int height = 0, width = 0, pad = 114;
        if (_inputSpec.ImageResize is { } resize)
        {
            if (resize.TryGetValue("mode", out var m) && m.ValueKind == JsonValueKind.String) mode = m.GetString();
            if (resize.TryGetValue("size", out var size) && size.ValueKind == JsonValueKind.Array && size.GetArrayLength() == 2)
            {
                height = size[0].GetInt32();
                width = size[1].GetInt32();
            }
            if (resize.TryGetValue("padValue", out var p) && p.ValueKind == JsonValueKind.Number) pad = p.GetInt32();
        }
        if (mode != "letterbox" || width <= 0 || height <= 0)
        {
            throw new NotSupportedException(
                $"Unsupported imageResize mode '{mode}' for a detector. See preprocessing.json's inputs[0].imageResize and adapt this client.");
        }

        using var image = Image.Load<Rgb24>(path);
        int originalWidth = image.Width, originalHeight = image.Height;
        var scale = Math.Min((double)height / originalHeight, (double)width / originalWidth);
        // ToEven rounds .5 to the nearest even integer, as Python's round() does.
        var newWidth = (int)Math.Round(originalWidth * scale, MidpointRounding.ToEven);
        var newHeight = (int)Math.Round(originalHeight * scale, MidpointRounding.ToEven);
        var left = (int)Math.Round((width - newWidth) / 2.0 - 0.1, MidpointRounding.ToEven);
        var top = (int)Math.Round((height - newHeight) / 2.0 - 0.1, MidpointRounding.ToEven);
        if (newWidth != originalWidth || newHeight != originalHeight)
            image.Mutate(x => x.Resize(newWidth, newHeight, KnownResamplers.Triangle)); // Triangle is bilinear

        var data = new float[3 * height * width];
        Array.Fill(data, pad / 255f);
        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < newHeight; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (var x = 0; x < newWidth; x++)
                {
                    var px = row[x];
                    var at = ((top + y) * width) + left + x;
                    data[(0 * height * width) + at] = px.R / 255f;
                    data[(1 * height * width) + at] = px.G / 255f;
                    data[(2 * height * width) + at] = px.B / 255f;
                }
            }
        });
        return (new DenseTensor<float>(data, [1, 3, height, width]), scale, left, top, originalWidth, originalHeight);
    }

    /// <summary>
    /// Indices of the boxes to keep, most confident first: within each class, a box is dropped when it overlaps a
    /// more confident one by more than <paramref name="iouThreshold"/> (intersection over union). Boxes are [x1, y1, x2, y2].
    /// </summary>
    private static List<int> NonMaxSuppression(List<float[]> boxes, List<float> scores, List<int> classes, double iouThreshold)
    {
        var kept = new List<int>();
        foreach (var label in classes.Distinct())
        {
            var order = Enumerable.Range(0, classes.Count)
                .Where(i => classes[i] == label)
                .OrderByDescending(i => scores[i]).ThenBy(i => i)
                .ToList();
            while (order.Count > 0)
            {
                var first = order[0];
                kept.Add(first);
                var a = boxes[first];
                var areaFirst = (a[2] - a[0]) * (a[3] - a[1]);
                order = order.Skip(1).Where(i =>
                {
                    var b = boxes[i];
                    var width = Math.Min(a[2], b[2]) - Math.Max(a[0], b[0]);
                    var height = Math.Min(a[3], b[3]) - Math.Max(a[1], b[1]);
                    var intersection = Math.Max(width, 0f) * Math.Max(height, 0f);
                    var union = areaFirst + ((b[2] - b[0]) * (b[3] - b[1])) - intersection;
                    return intersection / Math.Max(union, 1e-9f) <= iouThreshold;
                }).ToList();
            }
        }
        return kept.OrderByDescending(i => scores[i]).ToList();
    }

    /// <summary>
    /// Boxes found in an image, most confident first. <paramref name="confidence"/> and <paramref name="iou"/> default
    /// to the thresholds in preprocessing.json (`outputs[0].yoloOutput`).
    /// </summary>
    public List<Detection> Detect(string imagePath, double? confidence = null, double? iou = null, int maxDetections = 300)
    {
        var minConfidence = confidence ?? ThresholdFrom(_outputSpec.YoloOutput, "confidenceThreshold", 0.25);
        var maxOverlap = iou ?? ThresholdFrom(_outputSpec.YoloOutput, "iouThreshold", 0.7);
        var classes = _outputSpec.Classes ?? [];

        var (tensor, scale, left, top, originalWidth, originalHeight) = Letterbox(imagePath);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_onnxInputName, tensor) };
        using var results = _session.Run(inputs);
        var output = results.First(r => r.Name == _onnxOutputName).AsTensor<float>();
        // [1, 4 + classes, candidates]: centre x, centre y, width, height, then a score per class.
        var candidates = output.Dimensions[2];

        var boxes = new List<float[]>();
        var scores = new List<float>();
        var labels = new List<int>();
        for (var i = 0; i < candidates; i++)
        {
            var best = 0;
            var bestScore = float.NegativeInfinity;
            for (var c = 0; c < classes.Count; c++)
            {
                var score = output[0, 4 + c, i];
                if (score > bestScore) { bestScore = score; best = c; }
            }
            if (bestScore < minConfidence) continue;
            float cx = output[0, 0, i], cy = output[0, 1, i], w = output[0, 2, i], h = output[0, 3, i];
            boxes.Add([cx - (w / 2), cy - (h / 2), cx + (w / 2), cy + (h / 2)]);
            scores.Add(bestScore);
            labels.Add(best);
        }

        static double Clamp(double v, double max) => Math.Min(Math.Max(v, 0), max);
        return NonMaxSuppression(boxes, scores, labels, maxOverlap)
            .Take(maxDetections)
            .Select(index =>
            {
                // Undo the letterbox: remove the padding, divide by the scale, clip to the image.
                var b = boxes[index];
                var x1 = Clamp((Math.Max(b[0], 0f) - left) / scale, originalWidth);
                var x2 = Clamp((Math.Max(b[2], 0f) - left) / scale, originalWidth);
                var y1 = Clamp((Math.Max(b[1], 0f) - top) / scale, originalHeight);
                var y2 = Clamp((Math.Max(b[3], 0f) - top) / scale, originalHeight);
                return new Detection(
                    classes[labels[index]],
                    Math.Round(scores[index], 4),
                    new Box(
                        Math.Round(x1 / originalWidth, 6),
                        Math.Round(y1 / originalHeight, 6),
                        Math.Round((x2 - x1) / originalWidth, 6),
                        Math.Round((y2 - y1) / originalHeight, 6)));
            })
            .ToList();
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
        if (IsDetector)
        {
            throw new NotSupportedException("This model is an object detector: call Detect(), which returns boxes.");
        }
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
