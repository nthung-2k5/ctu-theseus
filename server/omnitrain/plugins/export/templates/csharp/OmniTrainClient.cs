// OmniTrain-generated inference client.
//
// Wraps the exported ONNX model with the preprocessing/postprocessing the platform
// applied at training time, read from preprocessing.json — the platform's exported
// ONNX graph is the bare model only (no resize/normalize, no class-index
// decoding), so this client reconstructs that step from the manifest rather
// than assuming a fixed tensor layout: input/output tensor names are matched
// against the graph's actual names at runtime, not hardcoded, since ONNX
// export doesn't guarantee a stable naming convention across exporter versions.
//
// Implements preprocessing for `image` inputs with `category` outputs (image
// classification) and for a `detections` output (object detection, YOLO): the
// ONNX graph returns every candidate box, so for a detector this client does
// the confidence filter, per-class non-maximum suppression and the undoing of
// the letterbox, and `Detect()` returns the boxes. Other input types throw with
// a pointer to preprocessing.json so you can adapt preprocessing yourself; see
// README.md.
//
// An `audio` input that carries an `audioSpectrogram` recipe (audio classification) is
// supported too: `Predict()` takes the path of a WAV file, makes the log-mel spectrograms
// the recipe describes with no audio library, and averages the windows' probabilities.
//
// A `text` input that carries a `textTokens` recipe (text classification) is supported as well:
// `Predict()` takes the text itself (not a file path), splits it into words and looks them up in
// the vocabulary that ships in preprocessing.json.
//
// A `text` input that carries a `tagWords` recipe and a `tags` output (token classification) is supported
// too: `Tag()` takes the text itself and returns a tag for every word.

using System.Buffers.Binary;
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

internal sealed class StftSpec
{
    public int NFft { get; set; }
    public int HopLength { get; set; }
}

internal sealed class MelSpec
{
    public int Bands { get; set; }
    public double FMin { get; set; }
    public double FMax { get; set; }
}

internal sealed class NormalizationSpec
{
    public double Mean { get; set; }
    public double Std { get; set; } = 1;
}

/// <summary>`inputs[0].audioSpectrogram` (recipe "log_mel_v1"): every number the spectrogram needs.</summary>
internal sealed class AudioRecipe
{
    public string Recipe { get; set; } = "";
    public int SampleRate { get; set; }
    public int WindowSamples { get; set; }
    public int WindowsPerAudio { get; set; }
    public int Frames { get; set; }
    public StftSpec Stft { get; set; } = new();
    public MelSpec Mel { get; set; } = new();
    public double LogFloor { get; set; } = 1e-6;
    public NormalizationSpec Normalization { get; set; } = new();
}

internal sealed class LowercaseSpec
{
    public int From { get; set; }
    public int To { get; set; }
    public int Add { get; set; }
}

/// <summary>`inputs[0].textTokens` (recipe "word_ids_v1"): how a text becomes word ids.</summary>
internal sealed class TextRecipe
{
    public string Recipe { get; set; } = "";
    public LowercaseSpec Lowercase { get; set; } = new();
    public List<int[]> SeparatorRanges { get; set; } = [];
    public int MaxTokens { get; set; }
    public int PadId { get; set; }
    public int UnknownId { get; set; } = 1;
    public int FirstWordId { get; set; } = 2;
    public List<string> Vocabulary { get; set; } = [];
}

/// <summary>`inputs[0].tagWords` (recipe "tag_word_ids_v1"): how a text becomes one id per word.</summary>
internal sealed class TagRecipe
{
    public string Recipe { get; set; } = "";
    public List<int[]> WhitespaceRanges { get; set; } = [];
    public LowercaseSpec Lowercase { get; set; } = new();
    public int MaxTokens { get; set; }
    public int PadId { get; set; }
    public int UnknownId { get; set; } = 1;
    public int FirstWordId { get; set; } = 2;
    public string FallbackTag { get; set; } = "O";
    public List<string> Vocabulary { get; set; } = [];
}

internal sealed class PreprocessingInput
{
    public string Type { get; set; } = "";
    public string Column { get; set; } = "";
    public AudioRecipe? AudioSpectrogram { get; set; }
    public TextRecipe? TextTokens { get; set; }
    public TagRecipe? TagWords { get; set; }
    public Dictionary<string, JsonElement>? ImageSize { get; set; }
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

/// <summary>One word of a tagged text, the tag the model gave it and how sure it was (0..1).</summary>
public sealed record TaggedWord(string Token, string Tag, double Confidence);

internal sealed class PreprocessingManifest
{
    public List<PreprocessingInput> Inputs { get; set; } = [];
    public List<PreprocessingOutput> Outputs { get; set; } = [];
}

/// <summary>
/// `inputs[0].audioSpectrogram` spelled out. The platform computes the same thing in numpy; the arithmetic is plain
/// enough (a polyphase resampler, a radix-2 FFT, a mel filterbank) to follow in any language.
/// </summary>
internal static class AudioFrontEnd
{
    /// <summary>A WAV file's samples, one array per channel, in [-1, 1], and its sample rate.</summary>
    public static (float[][] Channels, int Rate) ReadWav(byte[] file)
    {
        static string Tag(byte[] b, int at) => System.Text.Encoding.ASCII.GetString(b, at, 4);
        if (file.Length < 12 || Tag(file, 0) != "RIFF" || Tag(file, 8) != "WAVE")
            throw new InvalidDataException("Not a WAV file (only WAV is supported)");

        int code = 0, channels = 0, rate = 0, bits = 0, dataAt = -1;
        long dataSize = 0;
        for (long at = 12; at + 8 <= file.Length;)
        {
            var size = BinaryPrimitives.ReadUInt32LittleEndian(file.AsSpan((int)at + 4));
            var body = at + 8;
            if (Tag(file, (int)at) == "fmt " && size >= 16)
            {
                code = BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan((int)body));
                if (code == 0xFFFE && size >= 26) code = BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan((int)body + 24));
                channels = BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan((int)body + 2));
                rate = (int)BinaryPrimitives.ReadUInt32LittleEndian(file.AsSpan((int)body + 4));
                bits = BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan((int)body + 14));
            }
            else if (Tag(file, (int)at) == "data")
            {
                dataAt = (int)body;
                dataSize = Math.Min(size, file.Length - body); // a streamed file may claim more than it holds
                break;
            }
            at = body + size + (size & 1);
        }
        if (channels < 1 || dataAt < 0 || dataSize == 0) throw new InvalidDataException("The WAV file has no format or no samples");
        var width = bits >> 3;
        if (width < 1) throw new InvalidDataException("The WAV file has no channels");
        var frames = (int)(dataSize / (width * channels));

        float Read(int at) => (code, bits) switch
        {
            (1, 8) => (file[at] - 128) / 128f,
            (1, 16) => BinaryPrimitives.ReadInt16LittleEndian(file.AsSpan(at)) / 32767f,
            (1, 24) => (((file[at] | (file[at + 1] << 8) | (file[at + 2] << 16)) << 8) >> 8) / 8388607f,
            (1, 32) => (float)(BinaryPrimitives.ReadInt32LittleEndian(file.AsSpan(at)) / 2147483647.0),
            (3, 32) => BinaryPrimitives.ReadSingleLittleEndian(file.AsSpan(at)),
            (3, 64) => (float)BinaryPrimitives.ReadDoubleLittleEndian(file.AsSpan(at)),
            _ => throw new NotSupportedException($"Unsupported WAV encoding (format {code}, {bits} bits)"),
        };
        var result = new float[channels][];
        for (var c = 0; c < channels; c++) result[c] = new float[frames];
        for (var f = 0; f < frames; f++)
            for (var c = 0; c < channels; c++)
                result[c][f] = Read(dataAt + ((f * channels) + c) * width);
        return (result, rate);
    }

    private static long Gcd(long a, long b) => b == 0 ? a : Gcd(b, a % b);

    /// <summary>Modified Bessel function of the first kind, order 0 (the Kaiser window's building block).</summary>
    private static double BesselI0(double x)
    {
        double sum = 1, term = 1;
        for (var k = 1; k < 500; k++)
        {
            term *= Math.Pow(x / (2 * k), 2);
            sum += term;
            if (term < sum * 1e-17) break;
        }
        return sum;
    }

    /// <summary>
    /// `scipy.signal.resample_poly(x, up, down)`: out[i] = sum over k in -H..H of h[H + k] * x[(i * down + k) / up],
    /// counting the k for which i * down + k is a multiple of `up` and the index is inside the recording. `h` is a
    /// Kaiser-windowed (beta 5) sinc low-pass of 2H + 1 taps, H = 10 * max(up, down), scaled to a gain of `up`.
    /// </summary>
    public static float[] Resample(double[] x, int rate, int target)
    {
        var divisor = Gcd(rate, target);
        var up = target / divisor;
        var down = rate / divisor;
        var big = Math.Max(up, down);
        var half = 10 * big;
        static double Sinc(double v) => v == 0 ? 1 : Math.Sin(Math.PI * v) / (Math.PI * v);
        var taps = new double[(2 * half) + 1];
        var i0Beta = BesselI0(5);
        double total = 0;
        for (var n = 0; n <= 2 * half; n++)
        {
            var ratio = (double)(n - half) / half;
            var window = BesselI0(5 * Math.Sqrt(Math.Max(1 - (ratio * ratio), 0))) / i0Beta;
            taps[n] = Sinc((double)(n - half) / big) / big * window;
            total += taps[n];
        }
        for (var n = 0; n <= 2 * half; n++) taps[n] *= up / total;

        var outLength = (int)(((long)x.Length * up + down - 1) / down);
        var result = new float[outLength];
        for (var i = 0; i < outLength; i++)
        {
            var position = (long)i * down;
            var anchor = position / up;
            var phase = (int)(position % up);
            var first = -((half - phase) / up);
            var last = (half + phase) / up;
            double acc = 0;
            for (var d = first; d <= last; d++)
            {
                var index = anchor + d;
                if (index >= 0 && index < x.Length) acc += taps[half - phase + (d * up)] * x[index];
            }
            result[i] = (float)acc;
        }
        return result;
    }

    /// <summary>Where the windows of a `total`-sample recording start (`audioSpectrogram.windowStarts`).</summary>
    public static List<int> WindowStarts(int total, int length, int count)
    {
        if (total <= length) return [0];
        var room = total - length;
        if (count <= 1) return [room / 2];
        var starts = new SortedSet<int>();
        for (var i = 0; i < count; i++) starts.Add((int)Math.Round((double)i * room / (count - 1), MidpointRounding.ToEven));
        return [.. starts];
    }

    /// <summary>In-place radix-2 FFT of `re` + i `im` (length a power of two).</summary>
    private static void Fft(double[] re, double[] im)
    {
        var n = re.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            var bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }
        for (var size = 2; size <= n; size <<= 1)
        {
            var angle = -2 * Math.PI / size;
            for (var start = 0; start < n; start += size)
            {
                for (var k = 0; k < size / 2; k++)
                {
                    double wr = Math.Cos(angle * k), wi = Math.Sin(angle * k);
                    int a = start + k, b = a + (size / 2);
                    var tr = (re[b] * wr) - (im[b] * wi);
                    var ti = (re[b] * wi) + (im[b] * wr);
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }

    /// <summary>`[nFft / 2 + 1][bands]` triangular HTK mel filters, no area normalisation.</summary>
    public static double[][] MelFilterbank(int sampleRate, int nFft, int bands, double fMin, double fMax)
    {
        static double ToMel(double hz) => 2595 * Math.Log10(1 + (hz / 700));
        var edges = new double[bands + 2];
        for (var i = 0; i < edges.Length; i++)
        {
            var mel = ToMel(fMin) + ((ToMel(fMax) - ToMel(fMin)) * i / (bands + 1));
            edges[i] = 700 * (Math.Pow(10, mel / 2595) - 1);
        }
        var bins = (nFft / 2) + 1;
        var filters = new double[bins][];
        for (var bin = 0; bin < bins; bin++)
        {
            filters[bin] = new double[bands];
            var hz = sampleRate / 2.0 * bin / (bins - 1);
            for (var b = 0; b < bands; b++)
            {
                var rising = (hz - edges[b]) / (edges[b + 1] - edges[b]);
                var falling = (edges[b + 2] - hz) / (edges[b + 2] - edges[b + 1]);
                filters[bin][b] = Math.Max(0, Math.Min(rising, falling));
            }
        }
        return filters;
    }

    /// <summary>The standardised log-mel spectrogram of one window, `[bands][frames]` flattened row by row.</summary>
    public static float[] Spectrogram(float[] window, AudioRecipe recipe, double[][] filters)
    {
        int nFft = recipe.Stft.NFft, hop = recipe.Stft.HopLength, bands = recipe.Mel.Bands, frames = recipe.Frames;
        var hann = new double[nFft];
        for (var n = 0; n < nFft; n++) hann[n] = 0.5 - (0.5 * Math.Cos(2 * Math.PI * n / nFft));
        var result = new float[bands * frames];
        var re = new double[nFft];
        var im = new double[nFft];
        var bins = (nFft / 2) + 1;
        var power = new double[bins];
        for (var f = 0; f < frames; f++)
        {
            for (var n = 0; n < nFft; n++)
            {
                re[n] = window[(f * hop) + n] * hann[n];
                im[n] = 0;
            }
            Fft(re, im);
            for (var k = 0; k < bins; k++) power[k] = (re[k] * re[k]) + (im[k] * im[k]);
            for (var b = 0; b < bands; b++)
            {
                double mel = 0;
                for (var k = 0; k < bins; k++) mel += power[k] * filters[k][b];
                var logMel = (float)Math.Log(mel + recipe.LogFloor);
                result[(b * frames) + f] = (float)((logMel - recipe.Normalization.Mean) / recipe.Normalization.Std);
            }
        }
        return result;
    }
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

    /// <summary>An audio classifier: <see cref="Predict"/> takes the path of a WAV file.</summary>
    public bool IsAudio => _inputSpec.Type == "audio" && _inputSpec.AudioSpectrogram is not null;

    /// <summary>A text classifier: <see cref="Predict"/> takes the text itself, not a path.</summary>
    public bool IsText => _inputSpec.Type == "text" && _inputSpec.TextTokens is not null;

    /// <summary>A token classifier: call <see cref="Tag"/> with the text, which returns a tag for every word.</summary>
    public bool IsTagger => _inputSpec.Type == "text" && _inputSpec.TagWords is not null && _outputSpec.Type == "tags";

    private readonly bool[] _separators = new bool[0x10000];
    private readonly Dictionary<string, int> _wordIds = new(StringComparer.Ordinal);
    private readonly bool[] _whitespace = new bool[0x10000];
    private readonly Dictionary<string, int> _tagIds = new(StringComparer.Ordinal);

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
        if (IsAudio && _inputSpec.AudioSpectrogram!.Recipe != "log_mel_v1")
        {
            throw new NotSupportedException(
                $"Unknown audio recipe '{_inputSpec.AudioSpectrogram.Recipe}'. " +
                "See preprocessing.json's inputs[0].audioSpectrogram and adapt this client.");
        }
        if (IsText)
        {
            var recipe = _inputSpec.TextTokens!;
            if (recipe.Recipe != "word_ids_v1")
            {
                throw new NotSupportedException(
                    $"Unknown text recipe '{recipe.Recipe}'. " +
                    "See preprocessing.json's inputs[0].textTokens and adapt this client.");
            }
            foreach (var range in recipe.SeparatorRanges)
            {
                for (var unit = range[0]; unit <= Math.Min(range[1], 0xFFFF); unit++) _separators[unit] = true;
            }
            for (var i = 0; i < recipe.Vocabulary.Count; i++) _wordIds[recipe.Vocabulary[i]] = recipe.FirstWordId + i;
        }
        if (IsTagger)
        {
            var recipe = _inputSpec.TagWords!;
            if (recipe.Recipe != "tag_word_ids_v1")
            {
                throw new NotSupportedException(
                    $"Unknown tagging recipe '{recipe.Recipe}'. " +
                    "See preprocessing.json's inputs[0].tagWords and adapt this client.");
            }
            foreach (var range in recipe.WhitespaceRanges)
            {
                for (var unit = range[0]; unit <= Math.Min(range[1], 0xFFFF); unit++) _whitespace[unit] = true;
            }
            for (var i = 0; i < recipe.Vocabulary.Count; i++) _tagIds[recipe.Vocabulary[i]] = recipe.FirstWordId + i;
        }
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
        if (_inputSpec.ImageSize is { } pp)
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

    /// <summary>The recording as a batch of spectrograms, `[windows, 1, bands, frames]`, by `audioSpectrogram`.</summary>
    private DenseTensor<float> AudioSpectrograms(string path)
    {
        var recipe = _inputSpec.AudioSpectrogram!;
        var (channels, rate) = AudioFrontEnd.ReadWav(File.ReadAllBytes(path));
        var frames = channels[0].Length;
        // The channels' mean, at double precision as the platform does it.
        var mono = new double[frames];
        foreach (var channel in channels)
            for (var i = 0; i < frames; i++) mono[i] += channel[i];
        for (var i = 0; i < frames; i++) mono[i] /= channels.Length;
        var wave = AudioFrontEnd.Resample(mono, rate, recipe.SampleRate);
        if (wave.Length == 0) throw new InvalidDataException("The recording has no samples");

        var length = recipe.WindowSamples;
        var filters = AudioFrontEnd.MelFilterbank(recipe.SampleRate, recipe.Stft.NFft, recipe.Mel.Bands, recipe.Mel.FMin, recipe.Mel.FMax);
        var starts = AudioFrontEnd.WindowStarts(wave.Length, length, recipe.WindowsPerAudio);
        var size = recipe.Mel.Bands * recipe.Frames;
        var batch = new float[starts.Count * size];
        for (var w = 0; w < starts.Count; w++)
        {
            var window = new float[length]; // zero-padded at the end when the recording runs out
            Array.Copy(wave, starts[w], window, 0, Math.Min(length, wave.Length - starts[w]));
            AudioFrontEnd.Spectrogram(window, recipe, filters).CopyTo(batch, w * size);
        }
        return new DenseTensor<float>(batch, [starts.Count, 1, recipe.Mel.Bands, recipe.Frames]);
    }

    /// <summary>Class probabilities of a text: the logits of its one row through a softmax.</summary>
    private Dictionary<string, double> ClassifyText(string text, int topK)
    {
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_onnxInputName, TextTokens(text)) };
        using var results = _session.Run(inputs);
        var logits = results.First(r => r.Name == _onnxOutputName).AsEnumerable<float>().ToArray();
        var classes = _outputSpec.Classes ?? [];
        var probs = Softmax(logits);
        return classes
            .Select((name, i) => (name, prob: probs[i]))
            .OrderByDescending(p => p.prob)
            .Take(topK)
            .ToDictionary(p => p.name, p => (double)MathF.Round(p.prob, 4));
    }

    /// <summary>Class probabilities of a recording: each window's logits through a softmax, averaged over the windows.</summary>
    private Dictionary<string, double> ClassifyAudio(string path, int topK)
    {
        var tensor = AudioSpectrograms(path);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_onnxInputName, tensor) };
        using var results = _session.Run(inputs);
        var logits = results.First(r => r.Name == _onnxOutputName).AsEnumerable<float>().ToArray();
        var classes = _outputSpec.Classes ?? [];
        var windows = tensor.Dimensions[0];
        var mean = new double[classes.Count];
        for (var w = 0; w < windows; w++)
        {
            var probs = Softmax(logits.Skip(w * classes.Count).Take(classes.Count).ToArray());
            for (var c = 0; c < classes.Count; c++) mean[c] += probs[c] / (double)windows;
        }
        return classes
            .Select((name, i) => (name, prob: mean[i]))
            .OrderByDescending(p => p.prob)
            .Take(topK)
            .ToDictionary(p => p.name, p => Math.Round(p.prob, 4));
    }

    /// <summary>
    /// One <see cref="TaggedWord"/> per word of `text` (the words are the runs of characters that are not whitespace,
    /// as `str.split()` cuts them): the highest of the word's row of logits and its softmax probability. The first
    /// maxTokens words are read; a later word was not, and gets the recipe's fallbackTag with confidence 0.
    /// </summary>
    public List<TaggedWord> Tag(string text)
    {
        if (!IsTagger)
        {
            throw new NotSupportedException("This model does not tag words: call Predict() instead.");
        }
        var recipe = _inputSpec.TagWords!;
        var words = new List<string>();
        var start = -1;
        for (var i = 0; i < text.Length; i++)
        {
            if (_whitespace[text[i]])
            {
                if (start >= 0) words.Add(text[start..i]);
                start = -1;
            }
            else if (start < 0)
            {
                start = i;
            }
        }
        if (start >= 0) words.Add(text[start..]);
        if (words.Count == 0) return [];

        var ids = new long[recipe.MaxTokens];
        Array.Fill(ids, (long)recipe.PadId);
        var lower = recipe.Lowercase;
        for (var i = 0; i < Math.Min(words.Count, recipe.MaxTokens); i++)
        {
            var form = new System.Text.StringBuilder(words[i].Length);
            foreach (var unit in words[i]) form.Append(unit >= lower.From && unit <= lower.To ? (char)(unit + lower.Add) : unit);
            ids[i] = _tagIds.TryGetValue(form.ToString(), out var id) ? id : recipe.UnknownId;
        }
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_onnxInputName, new DenseTensor<long>(ids, [1, recipe.MaxTokens])),
        };
        using var results = _session.Run(inputs);
        var logits = results.First(r => r.Name == _onnxOutputName).AsEnumerable<float>().ToArray();
        var classes = _outputSpec.Classes ?? [];

        var tagged = new List<TaggedWord>(words.Count);
        for (var i = 0; i < words.Count; i++)
        {
            if (i >= recipe.MaxTokens)
            {
                tagged.Add(new TaggedWord(words[i], recipe.FallbackTag, 0.0));
                continue;
            }
            var probs = Softmax(logits.Skip(i * classes.Count).Take(classes.Count).ToArray());
            var best = 0;
            for (var k = 1; k < probs.Length; k++)
            {
                if (probs[k] > probs[best]) best = k;
            }
            tagged.Add(new TaggedWord(words[i], classes[best], (double)MathF.Round(probs[best], 4)));
        }
        return tagged;
    }

    /// <summary>The tokens of a text: the ASCII letters A-Z lower-cased, cut at every run of separator code units.</summary>
    private List<string> Tokenize(string text)
    {
        var lower = _inputSpec.TextTokens!.Lowercase;
        var tokens = new List<string>();
        var current = new System.Text.StringBuilder();
        foreach (var unit in text)
        {
            if (_separators[unit])
            {
                if (current.Length > 0) tokens.Add(current.ToString());
                current.Clear();
            }
            else
            {
                current.Append(unit >= lower.From && unit <= lower.To ? (char)(unit + lower.Add) : unit);
            }
        }
        if (current.Length > 0) tokens.Add(current.ToString());
        return tokens;
    }

    /// <summary>`[1, maxTokens]` int64 word ids: the first maxTokens tokens looked up in the vocabulary, padded at the end.</summary>
    private DenseTensor<long> TextTokens(string text)
    {
        var recipe = _inputSpec.TextTokens!;
        var ids = new long[recipe.MaxTokens];
        Array.Fill(ids, (long)recipe.PadId);
        var tokens = Tokenize(text);
        for (var i = 0; i < Math.Min(tokens.Count, recipe.MaxTokens); i++)
        {
            ids[i] = _wordIds.TryGetValue(tokens[i], out var id) ? id : recipe.UnknownId;
        }
        return new DenseTensor<long>(ids, [1, recipe.MaxTokens]);
    }

    private static float[] Softmax(float[] logits)
    {
        var max = logits.Max();
        var exp = logits.Select(v => MathF.Exp(v - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(v => v / sum).ToArray();
    }

    /// <summary>
    /// Runs inference and returns class -&gt; probability (classification, sorted descending) or {"value": x}
    /// (regression). `imagePath` is an image file, for an audio classifier a WAV file, and for a text
    /// classifier the text itself.
    /// </summary>
    public Dictionary<string, double> Predict(string imagePath, int topK = 5)
    {
        if (IsDetector)
        {
            throw new NotSupportedException("This model is an object detector: call Detect(), which returns boxes.");
        }
        if (IsTagger)
        {
            throw new NotSupportedException("This model tags every word of a text: call Tag(), which returns a tag for each.");
        }
        if (IsAudio) return ClassifyAudio(imagePath, topK);
        if (IsText) return ClassifyText(imagePath, topK);
        if (_inputSpec.Type != "image")
        {
            throw new NotSupportedException(
                $"No preprocessing implemented for input type '{_inputSpec.Type}'. " +
                "See preprocessing.json's inputs[0].imageSize and adapt this client.");
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
