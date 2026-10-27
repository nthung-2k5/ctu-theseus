// Usage (once compiled alongside OmniTrainClient.java, or copied into your own project):
//   java Main <path-to-image | path-to-wav-file>
//   java Main verify

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class Main {
    private static final double TOLERANCE = 1e-3;

    // Object detection. The platform's prediction comes from the original weights through Ultralytics, which pads an
    // image only up to a multiple of 32 before the network; this client pads to the full square the exported graph
    // was built for. The same box therefore comes out a little different, and a borderline one may appear in only one
    // list. So only confident boxes are held to account, and by overlap and a loose confidence margin, not exact
    // numbers.
    private static final double CONFIDENT = 0.35; // a box at least this confident on either side must have a partner on the other
    private static final double MIN_IOU = 0.7;
    private static final double CONFIDENCE_MARGIN = 0.1;

    private Main() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("Usage: gradle run --args=\"<path-to-image>\"");
            System.err.println("       gradle run --args=\"verify\"");
            System.exit(1);
        }

        try (OmniTrainClient client = new OmniTrainClient()) {
            if ("verify".equals(args[0])) {
                System.exit(verify(client));
            }
            if (client.isDetector()) {
                List<OmniTrainClient.Detection> found = client.detect(Path.of(args[0]));
                for (OmniTrainClient.Detection d : found) {
                    OmniTrainClient.Box b = d.box();
                    System.out.printf("%s: %.4f  x=%.3f y=%.3f w=%.3f h=%.3f%n", d.label(), d.confidence(), b.x(), b.y(), b.w(), b.h());
                }
                if (found.isEmpty()) System.out.println("No objects found.");
                return;
            }
            if (client.isTagger()) {
                // One line per word: the word, its tag and the model's confidence (in the platform's default charset:
                // for anything but plain ASCII, call tag from code with a String).
                for (OmniTrainClient.TaggedWord word : client.tag(args[0])) {
                    System.out.printf("%s\t%s\t%.4f%n", word.token(), word.tag(), word.confidence());
                }
                return;
            }
            // A text model is given the text itself (in the platform's default charset: for anything but plain
            // ASCII, call predictText from code with a String).
            Map<String, Double> predictions = client.isText() ? client.predictText(args[0], 5) : client.predict(Path.of(args[0]), 5);
            predictions.forEach((className, probability) -> System.out.printf("%s: %.4f%n", className, probability));
        }
    }

    private static double iou(OmniTrainClient.Box a, OmniTrainClient.Box b) {
        double width = Math.min(a.x() + a.w(), b.x() + b.w()) - Math.max(a.x(), b.x());
        double height = Math.min(a.y() + a.h(), b.y() + b.h()) - Math.max(a.y(), b.y());
        double intersection = Math.max(width, 0) * Math.max(height, 0);
        double union = a.w() * a.h() + b.w() * b.h() - intersection;
        return union > 0 ? intersection / union : 0;
    }

    /** Whether {@code others} has a box of the same class that overlaps enough and is about as confident. */
    private static boolean hasPartner(OmniTrainClient.Detection box, List<OmniTrainClient.Detection> others) {
        return others.stream()
            .filter(o -> o.label().equals(box.label()))
            .max(Comparator.comparingDouble(o -> iou(box.box(), o.box())))
            .map(best -> iou(box.box(), best.box()) >= MIN_IOU && Math.abs(best.confidence() - box.confidence()) <= CONFIDENCE_MARGIN)
            .orElse(false);
    }

    private static boolean verifyDetections(List<OmniTrainClient.Detection> expected, List<OmniTrainClient.Detection> actual) {
        boolean ok = true;
        for (int side = 0; side < 2; side++) {
            List<OmniTrainClient.Detection> source = side == 0 ? expected : actual;
            List<OmniTrainClient.Detection> other = side == 0 ? actual : expected;
            for (OmniTrainClient.Detection box : source) {
                if (box.confidence() >= CONFIDENT && !hasPartner(box, other)) {
                    ok = false;
                    OmniTrainClient.Box b = box.box();
                    System.out.printf("MISMATCH %s found '%s' (%.2f) at x=%.3f y=%.3f w=%.3f h=%.3f; %s has no matching box%n",
                        side == 0 ? "platform" : "this client", box.label(), box.confidence(), b.x(), b.y(), b.w(), b.h(),
                        side == 0 ? "this client" : "the platform");
                }
            }
        }
        if (ok) {
            long confident = expected.stream().filter(d -> d.confidence() >= CONFIDENT).count();
            System.out.printf("OK — %d confident box(es) match the platform's output (IoU >= %s).%n", confident, MIN_IOU);
        }
        return ok;
    }

    private static int verify(OmniTrainClient client) throws Exception {
        Path expectedPath = Path.of("expected.json");
        if (!expectedPath.toFile().exists()) {
            System.out.println("No expected.json in this bundle — nothing to verify against.");
            return 0;
        }

        ObjectMapper mapper = new ObjectMapper();
        JsonNode expected = mapper.readTree(expectedPath.toFile());
        JsonNode expectedPredictions = expected.get("predictions");
        Path samplePath = Path.of(expected.get("sampleFile").asText());

        if (client.isAudio() && !samplePath.toString().toLowerCase().endsWith(".wav")) {
            System.out.println("The bundled sample is not a WAV file, which this client can't read — nothing to verify.");
            return 0;
        }

        if (client.isDetector()) {
            JsonNode detections = expected.get("detections");
            if (detections == null || detections.isNull()) {
                System.out.println("expected.json has no detections to compare against.");
                return 0;
            }
            List<OmniTrainClient.Detection> platform = new ArrayList<>();
            for (JsonNode d : detections) {
                JsonNode b = d.get("box");
                platform.add(new OmniTrainClient.Detection(
                    d.get("label").asText(), d.get("confidence").asDouble(),
                    new OmniTrainClient.Box(b.get("x").asDouble(), b.get("y").asDouble(), b.get("w").asDouble(), b.get("h").asDouble())));
            }
            return verifyDetections(platform, client.detect(samplePath)) ? 0 : 1;
        }

        if (client.isTagger()) {
            // A token classifier is given the text and returns one tag per word, which must be the platform's, word for word.
            JsonNode tags = expected.get("tags");
            if (tags == null || tags.isNull()) {
                System.out.println("expected.json has no tags to compare against.");
                return 0;
            }
            List<OmniTrainClient.TaggedWord> found =
                client.tag(java.nio.file.Files.readString(samplePath, java.nio.charset.StandardCharsets.UTF_8));
            boolean same = found.size() == tags.size();
            if (!same) System.out.printf("MISMATCH expected %d tags, got %d%n", tags.size(), found.size());
            for (int i = 0; i < Math.min(found.size(), tags.size()); i++) {
                if (!found.get(i).tag().equals(tags.get(i).asText())) {
                    same = false;
                    System.out.printf("MISMATCH word %d: expected '%s', got '%s'%n", i, tags.get(i).asText(), found.get(i).tag());
                }
            }
            if (same) System.out.printf("OK — all %d tags match the platform's output.%n", found.size());
            return same ? 0 : 1;
        }

        if (expectedPredictions.size() == 0) {
            System.out.println("expected.json has no predictions to compare against.");
            return 0;
        }

        // A text model takes the text in the sample file, not the path to it.
        int topK = Math.max(expectedPredictions.size(), 5);
        Map<String, Double> actual = client.isText()
            ? client.predictText(java.nio.file.Files.readString(samplePath, java.nio.charset.StandardCharsets.UTF_8), topK)
            : client.predict(samplePath, topK);

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
