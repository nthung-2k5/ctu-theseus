/**
 * Theseus-generated inference client.
 *
 * Wraps the exported ONNX model with the preprocessing/postprocessing Ludwig
 * applied at training time, read from preprocessing.json — Ludwig's exported
 * ONNX graph is the bare model only (no resize/normalize, no class-index
 * decoding), so this client reconstructs that step from the manifest rather
 * than assuming a fixed tensor layout: input/output tensor names are matched
 * against the graph's actual names at runtime, not hardcoded, since ONNX
 * export doesn't guarantee a stable naming convention across Ludwig versions.
 *
 * Currently implements preprocessing for {@code image} inputs with {@code
 * category} outputs (image classification) — the modality this export path
 * has been verified against. Other input types throw with a pointer to
 * preprocessing.json so you can adapt preprocessing yourself; see README.md.
 */
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class TheseusClient implements AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession session;
    private final JsonNode inputSpec;
    private final JsonNode outputSpec;
    private final String onnxInputName;
    private final String onnxOutputName;

    public TheseusClient() throws OrtException, IOException {
        this(Path.of("model.onnx"), Path.of("preprocessing.json"));
    }

    /** Paths are resolved relative to the working directory the JVM was launched from (the project root when run via {@code gradle run}). */
    public TheseusClient(Path modelPath, Path preprocessingPath) throws OrtException, IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode manifest = mapper.readTree(preprocessingPath.toFile());
        this.inputSpec = manifest.get("inputs").get(0);
        this.outputSpec = manifest.get("outputs").get(0);

        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath.toString(), new OrtSession.SessionOptions());
        this.onnxInputName = matchTensorName(session.getInputNames(), inputSpec.get("column").asText());
        this.onnxOutputName = matchTensorName(session.getOutputNames(), outputSpec.get("column").asText());
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
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(original, 0, 0, width, height, null);
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
