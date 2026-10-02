// Usage: dotnet run -- <path-to-image>
//        dotnet run -- verify

using System.Text.Json;
using TheseusExport;

if (args.Length == 0)
{
    Console.Error.WriteLine("Usage: dotnet run -- <path-to-image>");
    Console.Error.WriteLine("       dotnet run -- verify");
    return 1;
}

using var client = new TheseusClient();

if (args[0] == "verify")
{
    return Verify(client);
}

var predictions = client.Predict(args[0]);
foreach (var (className, probability) in predictions)
{
    Console.WriteLine($"{className}: {probability:F4}");
}
return 0;

static int Verify(TheseusClient client)
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

file sealed class ExpectedJson
{
    public string SampleFile { get; init; } = "";
    public Dictionary<string, double> Predictions { get; init; } = [];
}
