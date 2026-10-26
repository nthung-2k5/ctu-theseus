/**
 * OmniTrain-generated inference client.
 *
 * Wraps the exported ONNX model with the preprocessing/postprocessing Ludwig
 * applied at training time, read from preprocessing.json — Ludwig's exported
 * ONNX graph is the bare model only (no resize/normalize, no class-index
 * decoding), so this client reconstructs that step from the manifest rather
 * than assuming a fixed tensor layout: input/output tensor names are matched
 * against the graph's actual names at runtime, not hardcoded, since ONNX
 * export doesn't guarantee a stable naming convention across Ludwig versions.
 *
 * Implements preprocessing for {@code image} inputs with {@code category}
 * outputs (image classification) and for a {@code detections} output (object
 * detection, YOLO): the ONNX graph returns every candidate box, so for a
 * detector this client does the confidence filter, per-class non-maximum
 * suppression and the undoing of the letterbox, and {@link #detect} returns the
 * boxes. Other input types throw with a pointer to preprocessing.json so you
 * can adapt preprocessing yourself; see README.md.
 */
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public final class OmniTrainClient implements AutoCloseable {
    /** A box as fractions (0..1) of the image: the top-left corner and the size. */
    public record Box(double x, double y, double w, double h) {}

    /** One object found in an image. */
    public record Detection(String label, double confidence, Box box) {}

    private final OrtEnvironment env;
    private final OrtSession session;
    private final JsonNode inputSpec;
    private final JsonNode outputSpec;
    private final String onnxInputName;
    private final String onnxOutputName;

    public OmniTrainClient() throws OrtException, IOException {
        this(Path.of("model.onnx"), Path.of("preprocessing.json"));
    }

    /** Paths are resolved relative to the working directory the JVM was launched from (the project root when run via {@code gradle run}). */
    public OmniTrainClient(Path modelPath, Path preprocessingPath) throws OrtException, IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode manifest = mapper.readTree(preprocessingPath.toFile());
        this.inputSpec = manifest.get("inputs").get(0);
        this.outputSpec = manifest.get("outputs").get(0);

        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath.toString(), new OrtSession.SessionOptions());
        this.onnxInputName = matchTensorName(session.getInputNames(), inputSpec.get("column").asText());
        this.onnxOutputName = matchTensorName(session.getOutputNames(), outputSpec.get("column").asText());
    }

    /** An object detector: use {@link #detect}; {@link #predict} is for classifiers and regressors. */
    public boolean isDetector() {
        return "detections".equals(outputSpec.path("type").asText());
    }

    private static String matchTensorName(Set<String> candidates, String preferred) {
        if (candidates.size() == 1) return candidates.iterator().next();
        for (String name : candidates) {
            if (name.equals(preferred) || name.startsWith(preferred + "::") || name.startsWith(preferred + "_")) {
                return name;
            }
        }
        throw new IllegalStateException(
            "Could not match an ONNX tensor to feature '" + preferred + "' among " + candidates + ". " +
            "Inspect preprocessing.json and adapt this client if the model uses an unrecognized naming scheme.");
    }

    private float[] preprocessImage(Path path, int width, int height) throws IOException {
        BufferedImage original = ImageIO.read(path.toFile());
        if (original == null) throw new IOException("Could not decode image: " + path);
        // Absent: squash the whole image to width x height.
        String mode = inputSpec.path("imageResize").path("mode").asText(null);
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        if (mode == null) {
            g.drawImage(original, 0, 0, width, height, null);
        } else if ("shorter_edge_center_crop".equals(mode)) {
            // Scale so the SHORTER edge is `size` (the other edge truncated to a whole pixel), then centre-crop a
            // size x size square: the same arithmetic as torchvision's Resize(size) + CenterCrop(size), which is
            // what the model was evaluated with. Drawing the scaled image at a negative offset crops it.
            int size = Math.min(width, height);
            int sourceWidth = original.getWidth(), sourceHeight = original.getHeight();
            int newWidth = sourceWidth <= sourceHeight ? size : (int) ((long) size * sourceWidth / sourceHeight);
            int newHeight = sourceWidth <= sourceHeight ? (int) ((long) size * sourceHeight / sourceWidth) : size;
            // rint rounds .5 to the nearest even integer, as Python's round() does.
            int left = (int) Math.rint((newWidth - size) / 2.0);
            int top = (int) Math.rint((newHeight - size) / 2.0);
            g.drawImage(original, -left, -top, newWidth, newHeight, null);
        } else {
            g.dispose();
            throw new UnsupportedOperationException(
                "Unsupported imageResize mode '" + mode + "'. " +
                "See preprocessing.json's inputs[0].imageResize and adapt this client.");
        }
        g.dispose();

        JsonNode norm = inputSpec.get("imageNormalization");
        float[] mean = null, std = null;
        if (norm != null) {
            mean = toFloatArray(norm.get("mean"));
            std = toFloatArray(norm.get("std"));
        }

        float[] chw = new float[3 * height * width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = resized.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF) / 255f;
                float gr = ((rgb >> 8) & 0xFF) / 255f;
                float b = (rgb & 0xFF) / 255f;
                if (mean != null) {
                    r = (r - mean[0]) / std[0];
                    gr = (gr - mean[1]) / std[1];
                    b = (b - mean[2]) / std[2];
                }
                chw[(0 * height * width) + (y * width) + x] = r;
                chw[(1 * height * width) + (y * width) + x] = gr;
                chw[(2 * height * width) + (y * width) + x] = b;
            }
        }
        return chw;
    }

    /** What {@link #letterbox} returns: the input tensor's data and what is needed to map boxes back. */
    private record Letterboxed(float[] chw, int size, double scale, int left, int top, int originalWidth, int originalHeight) {}

    /**
     * The image as the detector's input, and what is needed to map its boxes back. For {@code
     * imageResize.mode == "letterbox"}: scale so the LONGER edge is {@code size}, pad the shorter one, centred, with
     * {@code padValue} grey up to a size x size square (the arithmetic of ultralytics.data.augment.LetterBox). RGB,
     * 0..1, no mean/std.
     */
    private Letterboxed letterbox(Path path) throws IOException {
        JsonNode resize = inputSpec.path("imageResize");
        String mode = resize.path("mode").asText(null);
        JsonNode sizeNode = resize.path("size");
        if (!"letterbox".equals(mode) || !sizeNode.isArray() || sizeNode.size() != 2) {
            throw new UnsupportedOperationException(
                "Unsupported imageResize mode '" + mode + "' for a detector. " +
                "See preprocessing.json's inputs[0].imageResize and adapt this client.");
        }
        int height = sizeNode.get(0).asInt();
        int width = sizeNode.get(1).asInt();
        int pad = resize.path("padValue").asInt(114);

        BufferedImage original = ImageIO.read(path.toFile());
        if (original == null) throw new IOException("Could not decode image: " + path);
        int originalWidth = original.getWidth(), originalHeight = original.getHeight();
        double scale = Math.min((double) height / originalHeight, (double) width / originalWidth);
        // rint rounds .5 to the nearest even integer, as Python's round() does.
        int newWidth = (int) Math.rint(originalWidth * scale);
        int newHeight = (int) Math.rint(originalHeight * scale);
        int left = (int) Math.rint((width - newWidth) / 2.0 - 0.1);
        int top = (int) Math.rint((height - newHeight) / 2.0 - 0.1);

        BufferedImage canvas = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = canvas.createGraphics();
        g.setColor(new Color(pad, pad, pad));
        g.fillRect(0, 0, width, height);
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(original, left, top, newWidth, newHeight, null);
        g.dispose();

        float[] chw = new float[3 * height * width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = canvas.getRGB(x, y);
                chw[(0 * height * width) + (y * width) + x] = ((rgb >> 16) & 0xFF) / 255f;
                chw[(1 * height * width) + (y * width) + x] = ((rgb >> 8) & 0xFF) / 255f;
                chw[(2 * height * width) + (y * width) + x] = (rgb & 0xFF) / 255f;
            }
        }
        return new Letterboxed(chw, width, scale, left, top, originalWidth, originalHeight);
    }

    /**
     * Indices of the boxes to keep, most confident first: within each class, a box is dropped when it overlaps a
     * more confident one by more than {@code iouThreshold} (intersection over union). Boxes are [x1, y1, x2, y2].
     */
    private static List<Integer> nonMaxSuppression(List<float[]> boxes, List<Float> scores, List<Integer> classes, double iouThreshold) {
        List<Integer> kept = new ArrayList<>();
        for (int label : new TreeSet<>(classes)) {
            List<Integer> order = new ArrayList<>();
            for (int i = 0; i < classes.size(); i++) if (classes.get(i) == label) order.add(i);
            order.sort(Comparator.<Integer>comparingDouble(i -> -scores.get(i)).thenComparingInt(i -> i));
            while (!order.isEmpty()) {
                int first = order.remove(0);
                kept.add(first);
                float[] a = boxes.get(first);
                float areaFirst = (a[2] - a[0]) * (a[3] - a[1]);
                List<Integer> next = new ArrayList<>();
                for (int i : order) {
                    float[] b = boxes.get(i);
                    float width = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
                    float height = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
                    float intersection = Math.max(width, 0f) * Math.max(height, 0f);
                    float union = areaFirst + ((b[2] - b[0]) * (b[3] - b[1])) - intersection;
                    if (intersection / Math.max(union, 1e-9f) <= iouThreshold) next.add(i);
                }
                order = next;
            }
        }
        kept.sort(Comparator.<Integer>comparingDouble(i -> -scores.get(i)));
        return kept;
    }

    private static double clamp(double v, double max) {
        return Math.min(Math.max(v, 0), max);
    }

    private static double round(double v, int places) {
        double factor = Math.pow(10, places);
        return Math.rint(v * factor) / factor;
    }

    /** Boxes found in an image, most confident first, using the thresholds in preprocessing.json ({@code outputs[0].yoloOutput}). */
    public List<Detection> detect(Path imagePath) throws OrtException, IOException {
        JsonNode info = outputSpec.path("yoloOutput");
        return detect(imagePath, info.path("confidenceThreshold").asDouble(0.25), info.path("iouThreshold").asDouble(0.7), 300);
    }

    public List<Detection> detect(Path imagePath, double confidence, double iou, int maxDetections) throws OrtException, IOException {
        List<String> classes = new ArrayList<>();
        outputSpec.path("classes").forEach(n -> classes.add(n.asText()));

        Letterboxed input = letterbox(imagePath);
        long[] shape = {1, 3, input.size(), input.size()};
        float[][] table; // [4 + classes][candidates]
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input.chw()), shape);
             OrtSession.Result result = session.run(Collections.singletonMap(onnxInputName, tensor))) {
            OnnxValue outputValue = result.get(onnxOutputName)
                .orElseThrow(() -> new IllegalStateException("Session produced no output named '" + onnxOutputName + "'"));
            table = ((float[][][]) outputValue.getValue())[0];
        }
        int candidates = table[0].length;

        List<float[]> boxes = new ArrayList<>();
        List<Float> scores = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        for (int i = 0; i < candidates; i++) {
            int best = 0;
            float bestScore = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < classes.size(); c++) {
                float score = table[4 + c][i];
                if (score > bestScore) {
                    bestScore = score;
                    best = c;
                }
            }
            if (bestScore < confidence) continue;
            float cx = table[0][i], cy = table[1][i], w = table[2][i], h = table[3][i];
            boxes.add(new float[] {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2});
            scores.add(bestScore);
            labels.add(best);
        }

        List<Detection> detections = new ArrayList<>();
        for (int index : nonMaxSuppression(boxes, scores, labels, iou)) {
            if (detections.size() >= maxDetections) break;
            // Undo the letterbox: remove the padding, divide by the scale, clip to the image.
            float[] b = boxes.get(index);
            double x1 = clamp((Math.max(b[0], 0f) - input.left()) / input.scale(), input.originalWidth());
            double x2 = clamp((Math.max(b[2], 0f) - input.left()) / input.scale(), input.originalWidth());
            double y1 = clamp((Math.max(b[1], 0f) - input.top()) / input.scale(), input.originalHeight());
            double y2 = clamp((Math.max(b[3], 0f) - input.top()) / input.scale(), input.originalHeight());
            detections.add(new Detection(
                classes.get(labels.get(index)),
                round(scores.get(index), 4),
                new Box(
                    round(x1 / input.originalWidth(), 6),
                    round(y1 / input.originalHeight(), 6),
                    round((x2 - x1) / input.originalWidth(), 6),
                    round((y2 - y1) / input.originalHeight(), 6))));
        }
        return detections;
    }

    private static float[] toFloatArray(JsonNode arr) {
        float[] out = new float[arr.size()];
        for (int i = 0; i < arr.size(); i++) out[i] = (float) arr.get(i).asDouble();
        return out;
    }

    /** Flattens an arbitrarily-nested ONNX output array (its shape depends on the graph) into a single {@code float[]}. */
    private static float[] flatten(Object raw) {
        List<Float> out = new ArrayList<>();
        flattenInto(raw, out);
        float[] result = new float[out.size()];
        for (int i = 0; i < result.length; i++) result[i] = out.get(i);
        return result;
    }

    private static void flattenInto(Object obj, List<Float> out) {
        if (obj instanceof float[] arr) {
            for (float v : arr) out.add(v);
        } else if (obj instanceof Object[] arr) {
            for (Object o : arr) flattenInto(o, out);
        } else if (obj instanceof Float f) {
            out.add(f);
        } else {
            throw new IllegalStateException("Unexpected ONNX output shape: " + (obj == null ? "null" : obj.getClass()));
        }
    }

    private static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) max = Math.max(max, v);
        float sum = 0;
        float[] exp = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < exp.length; i++) exp[i] /= sum;
        return exp;
    }

    /** Runs inference and returns {@code {className: probability}} sorted descending (classification), or {@code {"value": x}} (regression). */
    public Map<String, Double> predict(Path imagePath, int topK) throws OrtException, IOException {
        if (isDetector()) {
            throw new UnsupportedOperationException("This model is an object detector: call detect(), which returns boxes.");
        }
        String inputType = inputSpec.get("type").asText();
        if (!"image".equals(inputType)) {
            throw new UnsupportedOperationException(
                "No preprocessing implemented for input type '" + inputType + "'. " +
                "See preprocessing.json's inputs[0].ludwigPreprocessing and adapt this client.");
        }

        JsonNode pp = inputSpec.path("ludwigPreprocessing");
        int width = pp.path("width").asInt(224);
        int height = pp.path("height").asInt(224);

        float[] chw = preprocessImage(imagePath, width, height);
        long[] shape = {1, 3, height, width};
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape)) {
            try (OrtSession.Result result = session.run(Collections.singletonMap(onnxInputName, tensor))) {
                OnnxValue outputValue = result.get(onnxOutputName)
                    .orElseThrow(() -> new IllegalStateException("Session produced no output named '" + onnxOutputName + "'"));
                float[] logits = flatten(outputValue.getValue());
                return decode(logits, topK);
            }
        }
    }

    private Map<String, Double> decode(float[] logits, int topK) {
        JsonNode classesNode = outputSpec.get("classes");
        if (classesNode != null && classesNode.isArray() && classesNode.size() == logits.length) {
            float[] probs = softmax(logits);
            List<String> classes = new ArrayList<>();
            classesNode.forEach(n -> classes.add(n.asText()));

            List<Map.Entry<String, Float>> ranked = new ArrayList<>();
            for (int i = 0; i < classes.size(); i++) ranked.add(Map.entry(classes.get(i), probs[i]));
            ranked.sort((a, b) -> Float.compare(b.getValue(), a.getValue()));

            LinkedHashMap<String, Double> out = new LinkedHashMap<>();
            for (int i = 0; i < Math.min(topK, ranked.size()); i++) {
                Map.Entry<String, Float> e = ranked.get(i);
                out.put(e.getKey(), Math.round(e.getValue() * 10000.0) / 10000.0);
            }
            return out;
        }
        return Map.of("value", (double) logits[0]);
    }

    @Override
    public void close() throws OrtException {
        session.close();
        // OrtEnvironment.getEnvironment() is a process-wide singleton — do not close it here.
    }
}
