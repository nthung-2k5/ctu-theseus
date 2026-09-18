// Usage (once compiled alongside TheseusClient.java, or copied into your own project):
//   java Main <path-to-image>
//   java Main verify

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.file.Path;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;

public final class Main {
    private static final double TOLERANCE = 1e-3;

    private Main() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("Usage: gradle run --args=\"<path-to-image>\"");
            System.err.println("       gradle run --args=\"verify\"");
            System.exit(1);
        }

        try (TheseusClient client = new TheseusClient()) {
            if ("verify".equals(args[0])) {
                System.exit(verify(client));
            }
            Map<String, Double> predictions = client.predict(Path.of(args[0]), 5);
            predictions.forEach((className, probability) -> System.out.printf("%s: %.4f%n", className, probability));
        }
    }

    private static int verify(TheseusClient client) throws Exception {
        Path expectedPath = Path.of("expected.json");
        if (!expectedPath.toFile().exists()) {
            System.out.println("No expected.json in this bundle — nothing to verify against.");
            return 0;
        }

        ObjectMapper mapper = new ObjectMapper();
        JsonNode expected = mapper.readTree(expectedPath.toFile());
        JsonNode expectedPredictions = expected.get("predictions");
        Path samplePath = Path.of(expected.get("sampleFile").asText());

        if (expectedPredictions.size() == 0) {
            System.out.println("expected.json has no predictions to compare against.");
            return 0;
        }

        Map<String, Double> actual = client.predict(samplePath, Math.max(expectedPredictions.size(), 5));

        String expectedTop1 = null;
        double expectedTop1Prob = Double.NEGATIVE_INFINITY;
        Iterator<Map.Entry<String, JsonNode>> fields = expectedPredictions.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> e = fields.next();
            double prob = e.getValue().asDouble();
            if (prob > expectedTop1Prob) {
                expectedTop1Prob = prob;
                expectedTop1 = e.getKey();
            }
        }
        String actualTop1 = actual.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(null);

        boolean ok = true;
        if (!Objects.equals(actualTop1, expectedTop1)) {
            ok = false;
            System.out.printf("MISMATCH top-1 class: expected '%s', got '%s'%n", expectedTop1, actualTop1);
        }

        fields = expectedPredictions.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> e = fields.next();
            String className = e.getKey();
            double expectedProb = e.getValue().asDouble();
            Double actualProb = actual.get(className);
            if (actualProb == null) {
                ok = false;
                System.out.printf("MISMATCH class '%s' missing from this client's output%n", className);
                continue;
            }
            double diff = Math.abs(actualProb - expectedProb);
            if (diff > TOLERANCE) {
                ok = false;
                System.out.printf("MISMATCH class '%s': expected %.4f, got %.4f (diff %.4f)%n", className, expectedProb, actualProb, diff);
            }
        }

        if (ok) {
            System.out.printf("OK — top-1 '%s' matches platform output within %s.%n", actualTop1, TOLERANCE);
            return 0;
        }
        return 1;
    }
}
