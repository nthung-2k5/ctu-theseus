// Usage: dotnet run -- <path-to-image | path-to-wav-file>
//        dotnet run -- verify

using System.Text.Json;
using OmniTrainExport;

if (args.Length == 0)
{
    Console.Error.WriteLine("Usage: dotnet run -- <path-to-image>");
    Console.Error.WriteLine("       dotnet run -- verify");
    return 1;
}

using var client = new OmniTrainClient();

if (args[0] == "verify")
{
    return Verify(client);
}

if (client.IsDetector)
{
    var found = client.Detect(args[0]);
    foreach (var d in found)
    {
        Console.WriteLine($"{d.Label}: {d.Confidence:F4}  x={d.Box.X:F3} y={d.Box.Y:F3} w={d.Box.W:F3} h={d.Box.H:F3}");
    }
    if (found.Count == 0) Console.WriteLine("No objects found.");
    return 0;
}

var predictions = client.Predict(args[0]);
foreach (var (className, probability) in predictions)
{
    Console.WriteLine($"{className}: {probability:F4}");
}
return 0;

static int Verify(OmniTrainClient client)
{
    const double tolerance = 1e-3;
    var here = AppContext.BaseDirectory;
    var expectedPath = Path.Combine(here, "expected.json");
    if (!File.Exists(expectedPath))
    {
        Console.WriteLine("No expected.json in this bundle — nothing to verify against.");
        return 0;
    }

    var opts = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    var expected = JsonSerializer.Deserialize<ExpectedJson>(File.ReadAllText(expectedPath), opts)
        ?? throw new InvalidDataException("expected.json is empty or invalid");
    var samplePath = Path.Combine(here, expected.SampleFile);

    if (client.IsAudio && !samplePath.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine("The bundled sample is not a WAV file, which this client can't read — nothing to verify.");
        return 0;
    }

    if (client.IsDetector)
    {
        if (expected.Detections is null)
        {
            Console.WriteLine("expected.json has no detections to compare against.");
            return 0;
        }
        return VerifyDetections(expected.Detections, client.Detect(samplePath)) ? 0 : 1;
    }

    if (expected.Predictions.Count == 0)
    {
        Console.WriteLine("expected.json has no predictions to compare against.");
        return 0;
    }

    var actual = client.Predict(samplePath, Math.Max(expected.Predictions.Count, 5));

    var expectedTop1 = expected.Predictions.OrderByDescending(p => p.Value).First().Key;
    var actualTop1 = actual.Count > 0 ? actual.OrderByDescending(p => p.Value).First().Key : null;

    var ok = true;
    if (actualTop1 != expectedTop1)
    {
        ok = false;
        Console.WriteLine($"MISMATCH top-1 class: expected '{expectedTop1}', got '{actualTop1}'");
    }

    foreach (var (className, expectedProb) in expected.Predictions)
    {
        if (!actual.TryGetValue(className, out var actualProb))
        {
            ok = false;
            Console.WriteLine($"MISMATCH class '{className}' missing from this client's output");
            continue;
        }
        var diff = Math.Abs(actualProb - expectedProb);
        if (diff > tolerance)
        {
            ok = false;
            Console.WriteLine($"MISMATCH class '{className}': expected {expectedProb:F4}, got {actualProb:F4} (diff {diff:F4})");
        }
    }

    if (ok)
    {
        Console.WriteLine($"OK — top-1 '{actualTop1}' matches platform output within {tolerance}.");
        return 0;
    }
    return 1;
}

// Object detection. The platform's prediction comes from the original weights through Ultralytics, which pads an
// image only up to a multiple of 32 before the network; this client pads to the full square the exported graph was
// built for. The same box therefore comes out a little different, and a borderline one may appear in only one list.
// So only confident boxes are held to account, and by overlap and a loose confidence margin, not exact numbers.
static bool VerifyDetections(List<Detection> expected, List<Detection> actual)
{
    const double confident = 0.35; // a box at least this confident on either side must have a partner on the other
    const double minIou = 0.7;
    const double confidenceMargin = 0.1;

    static double Iou(Box a, Box b)
    {
        var width = Math.Min(a.X + a.W, b.X + b.W) - Math.Max(a.X, b.X);
        var height = Math.Min(a.Y + a.H, b.Y + b.H) - Math.Max(a.Y, b.Y);
        var intersection = Math.Max(width, 0) * Math.Max(height, 0);
        var union = (a.W * a.H) + (b.W * b.H) - intersection;
        return union > 0 ? intersection / union : 0;
    }

    bool HasPartner(Detection box, List<Detection> others)
    {
        var same = others.Where(o => o.Label == box.Label).ToList();
        if (same.Count == 0) return false;
        var best = same.OrderByDescending(o => Iou(box.Box, o.Box)).First();
        return Iou(box.Box, best.Box) >= minIou && Math.Abs(best.Confidence - box.Confidence) <= confidenceMargin;
    }

    var ok = true;
    foreach (var (source, side, other, otherName) in new[]
    {
        (expected, "platform", actual, "this client"),
        (actual, "this client", expected, "the platform"),
    })
    {
        foreach (var box in source.Where(b => b.Confidence >= confident && !HasPartner(b, other)))
        {
            ok = false;
            Console.WriteLine(
                $"MISMATCH {side} found '{box.Label}' ({box.Confidence:F2}) at x={box.Box.X:F3} y={box.Box.Y:F3} " +
                $"w={box.Box.W:F3} h={box.Box.H:F3}; {otherName} has no matching box");
        }
    }
    if (ok)
    {
        Console.WriteLine($"OK — {expected.Count(b => b.Confidence >= confident)} confident box(es) match the platform's output (IoU >= {minIou}).");
    }
    return ok;
}

file sealed class ExpectedJson
{
    public string SampleFile { get; init; } = "";
    public Dictionary<string, double> Predictions { get; init; } = [];
    public List<Detection>? Detections { get; init; }
}
