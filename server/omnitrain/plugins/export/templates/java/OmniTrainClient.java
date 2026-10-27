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
 *
 * An {@code audio} input that carries an {@code audioSpectrogram} recipe (audio
 * classification) is supported too: {@link #predict} takes the path of a WAV file,
 * makes the log-mel spectrograms the recipe describes with no audio library, and
 * averages the windows' probabilities.
 *
 * A {@code text} input that carries a {@code textTokens} recipe (text classification) is
 * supported as well: {@link #predictText} takes the text itself, splits it into words and
 * looks them up in the vocabulary that ships in preprocessing.json.
 *
 * A {@code text} input that carries a {@code tagWords} recipe and a {@code tags} output (token
 * classification) is supported too: {@link #tag} takes the text itself and returns a tag for every word.
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * {@code inputs[0].textTokens} (recipe "word_ids_v1") spelled out: how a text becomes word ids.
 */
final class TextFrontEnd {
    private TextFrontEnd() {}

    static final class Recipe {
        final String recipe;
        final int lowerFrom;
        final int lowerTo;
        final int lowerAdd;
        final boolean[] separators = new boolean[0x10000];
        final int maxTokens;
        final int padId;
        final int unknownId;
        final java.util.HashMap<String, Integer> ids = new java.util.HashMap<>();

        Recipe(JsonNode node) {
            recipe = node.path("recipe").asText();
            lowerFrom = node.path("lowercase").path("from").asInt();
            lowerTo = node.path("lowercase").path("to").asInt();
            lowerAdd = node.path("lowercase").path("add").asInt();
            for (JsonNode range : node.path("separatorRanges")) {
                for (int unit = range.get(0).asInt(); unit <= Math.min(range.get(1).asInt(), 0xFFFF); unit++) {
                    separators[unit] = true;
                }
            }
            maxTokens = node.path("maxTokens").asInt();
            padId = node.path("padId").asInt();
            unknownId = node.path("unknownId").asInt(1);
            int first = node.path("firstWordId").asInt(2);
            int i = 0;
            for (JsonNode word : node.path("vocabulary")) ids.put(word.asText(), first + i++);
        }
    }

    /** The tokens of a text: the ASCII letters A-Z lower-cased, cut at every run of separator code units. */
    static List<String> tokenize(String text, Recipe r) {
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char unit = text.charAt(i);
            if (r.separators[unit]) {
                if (current.length() > 0) tokens.add(current.toString());
                current.setLength(0);
            } else {
                current.append(unit >= r.lowerFrom && unit <= r.lowerTo ? (char) (unit + r.lowerAdd) : unit);
            }
        }
        if (current.length() > 0) tokens.add(current.toString());
        return tokens;
    }

    /** {@code maxTokens} word ids: the first maxTokens tokens looked up in the vocabulary, padded at the end. */
    static long[] tokenIds(String text, Recipe r) {
        long[] ids = new long[r.maxTokens];
        java.util.Arrays.fill(ids, r.padId);
        List<String> tokens = tokenize(text, r);
        for (int i = 0; i < Math.min(tokens.size(), r.maxTokens); i++) {
            ids[i] = r.ids.getOrDefault(tokens.get(i), r.unknownId);
        }
        return ids;
    }
}

/**
 * {@code inputs[0].tagWords} (recipe "tag_word_ids_v1") spelled out: how a text becomes one id per word.
 */
final class TagFrontEnd {
    private TagFrontEnd() {}

    static final class Recipe {
        final String recipe;
        final int lowerFrom;
        final int lowerTo;
        final int lowerAdd;
        final boolean[] whitespace = new boolean[0x10000];
        final int maxTokens;
        final int padId;
        final int unknownId;
        final String fallbackTag;
        final java.util.HashMap<String, Integer> ids = new java.util.HashMap<>();

        Recipe(JsonNode node) {
            recipe = node.path("recipe").asText();
            lowerFrom = node.path("lowercase").path("from").asInt();
            lowerTo = node.path("lowercase").path("to").asInt();
            lowerAdd = node.path("lowercase").path("add").asInt();
            for (JsonNode range : node.path("whitespaceRanges")) {
                for (int unit = range.get(0).asInt(); unit <= Math.min(range.get(1).asInt(), 0xFFFF); unit++) {
                    whitespace[unit] = true;
                }
            }
            maxTokens = node.path("maxTokens").asInt();
            padId = node.path("padId").asInt();
            unknownId = node.path("unknownId").asInt(1);
            fallbackTag = node.path("fallbackTag").asText("O");
            int first = node.path("firstWordId").asInt(2);
            int i = 0;
            for (JsonNode word : node.path("vocabulary")) ids.put(word.asText(), first + i++);
        }
    }

    /** The words of a text: the runs of code units that are not whitespace (the ranges are the ones {@code str.split()} cuts at). */
    static List<String> words(String text, Recipe r) {
        List<String> words = new ArrayList<>();
        int start = -1;
        for (int i = 0; i < text.length(); i++) {
            if (r.whitespace[text.charAt(i)]) {
                if (start >= 0) words.add(text.substring(start, i));
                start = -1;
            } else if (start < 0) {
                start = i;
            }
        }
        if (start >= 0) words.add(text.substring(start));
        return words;
    }

    /** {@code maxTokens} word ids: the first maxTokens words, ASCII capitals lower-cased, looked up in the vocabulary, padded at the end. */
    static long[] wordIds(List<String> words, Recipe r) {
        long[] ids = new long[r.maxTokens];
        java.util.Arrays.fill(ids, r.padId);
        for (int i = 0; i < Math.min(words.size(), r.maxTokens); i++) {
            String word = words.get(i);
            StringBuilder form = new StringBuilder(word.length());
            for (int k = 0; k < word.length(); k++) {
                char unit = word.charAt(k);
                form.append(unit >= r.lowerFrom && unit <= r.lowerTo ? (char) (unit + r.lowerAdd) : unit);
            }
            ids[i] = r.ids.getOrDefault(form.toString(), r.unknownId);
        }
        return ids;
    }
}

/**
 * {@code inputs[0].audioSpectrogram} spelled out. The platform computes the same thing in numpy; the arithmetic is
 * plain enough (a polyphase resampler, a radix-2 FFT, a mel filterbank) to follow in any language.
 */
final class AudioFrontEnd {
    private AudioFrontEnd() {}

    /** The numbers of {@code audioSpectrogram} (recipe "log_mel_v1"). */
    record Recipe(String recipe, int sampleRate, int windowSamples, int windowsPerAudio, int frames, int nFft, int hop,
                  int bands, double fMin, double fMax, double logFloor, double mean, double std) {
        static Recipe of(JsonNode r) {
            return new Recipe(
                r.path("recipe").asText(), r.path("sampleRate").asInt(), r.path("windowSamples").asInt(),
                r.path("windowsPerAudio").asInt(), r.path("frames").asInt(),
                r.path("stft").path("nFft").asInt(), r.path("stft").path("hopLength").asInt(),
                r.path("mel").path("bands").asInt(), r.path("mel").path("fMin").asDouble(),
                r.path("mel").path("fMax").asDouble(), r.path("logFloor").asDouble(1e-6),
                r.path("normalization").path("mean").asDouble(), r.path("normalization").path("std").asDouble(1));
        }
    }

    /** A WAV file's samples, one array per channel, in [-1, 1]; the sample rate is returned in {@code rate[0]}. */
    static float[][] readWav(byte[] file, int[] rate) throws IOException {
        if (file.length < 12 || !tag(file, 0).equals("RIFF") || !tag(file, 8).equals("WAVE")) {
            throw new IOException("Not a WAV file (only WAV is supported)");
        }
        ByteBuffer buf = ByteBuffer.wrap(file).order(ByteOrder.LITTLE_ENDIAN);
        int code = 0, channels = 0, bits = 0, dataAt = -1;
        long dataSize = 0;
        for (long at = 12; at + 8 <= file.length; ) {
            long size = buf.getInt((int) at + 4) & 0xFFFFFFFFL;
            long body = at + 8;
            String id = tag(file, (int) at);
            if (id.equals("fmt ") && size >= 16) {
                code = buf.getShort((int) body) & 0xFFFF;
                if (code == 0xFFFE && size >= 26) code = buf.getShort((int) body + 24) & 0xFFFF;
                channels = buf.getShort((int) body + 2) & 0xFFFF;
                rate[0] = buf.getInt((int) body + 4);
                bits = buf.getShort((int) body + 14) & 0xFFFF;
            } else if (id.equals("data")) {
                dataAt = (int) body;
                dataSize = Math.min(size, file.length - body); // a streamed file may claim more than it holds
                break;
            }
            at = body + size + (size & 1);
        }
        if (channels < 1 || dataAt < 0 || dataSize == 0) throw new IOException("The WAV file has no format or no samples");
        int width = bits >> 3;
        if (width < 1) throw new IOException("The WAV file has no channels");
        int frames = (int) (dataSize / ((long) width * channels));
        float[][] out = new float[channels][frames];
        for (int f = 0; f < frames; f++) {
            for (int c = 0; c < channels; c++) {
                int at = dataAt + (f * channels + c) * width;
                float v;
                if (code == 1 && bits == 8) v = ((file[at] & 0xFF) - 128) / 128f;
                else if (code == 1 && bits == 16) v = buf.getShort(at) / 32767f;
                else if (code == 1 && bits == 24) {
                    int raw = (file[at] & 0xFF) | ((file[at + 1] & 0xFF) << 8) | ((file[at + 2] & 0xFF) << 16);
                    v = ((raw << 8) >> 8) / 8388607f;
                } else if (code == 1 && bits == 32) v = (float) (buf.getInt(at) / 2147483647.0);
                else if (code == 3 && bits == 32) v = buf.getFloat(at);
                else if (code == 3 && bits == 64) v = (float) buf.getDouble(at);
                else throw new UnsupportedOperationException("Unsupported WAV encoding (format " + code + ", " + bits + " bits)");
                out[c][f] = v;
            }
        }
        return out;
    }

    private static String tag(byte[] b, int at) {
        return new String(b, at, 4, StandardCharsets.US_ASCII);
    }

    private static long gcd(long a, long b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    /** Modified Bessel function of the first kind, order 0 (the Kaiser window's building block). */
    private static double besselI0(double x) {
        double sum = 1, term = 1;
        for (int k = 1; k < 500; k++) {
            term *= Math.pow(x / (2 * k), 2);
            sum += term;
            if (term < sum * 1e-17) break;
        }
        return sum;
    }

    private static double sinc(double v) {
        return v == 0 ? 1 : Math.sin(Math.PI * v) / (Math.PI * v);
    }

    /**
     * {@code scipy.signal.resample_poly(x, up, down)}: out[i] = sum over k in -H..H of h[H + k] * x[(i * down + k) / up],
     * counting the k for which i * down + k is a multiple of {@code up} and the index is inside the recording.
     * {@code h} is a Kaiser-windowed (beta 5) sinc low-pass of 2H + 1 taps, H = 10 * max(up, down), scaled to a gain
     * of {@code up}.
     */
    static float[] resample(double[] x, int rate, int target) {
        long divisor = gcd(rate, target);
        int up = (int) (target / divisor), down = (int) (rate / divisor);
        int big = Math.max(up, down), half = 10 * big;
        double[] taps = new double[2 * half + 1];
        double i0Beta = besselI0(5), total = 0;
        for (int n = 0; n <= 2 * half; n++) {
            double ratio = (double) (n - half) / half;
            double window = besselI0(5 * Math.sqrt(Math.max(1 - ratio * ratio, 0))) / i0Beta;
            taps[n] = sinc((double) (n - half) / big) / big * window;
            total += taps[n];
        }
        for (int n = 0; n <= 2 * half; n++) taps[n] *= up / total;

        int outLength = (int) (((long) x.length * up + down - 1) / down);
        float[] out = new float[outLength];
        for (int i = 0; i < outLength; i++) {
            long position = (long) i * down;
            long anchor = position / up;
            int phase = (int) (position % up);
            int first = -((half - phase) / up), last = (half + phase) / up;
            double acc = 0;
            for (int d = first; d <= last; d++) {
                long index = anchor + d;
                if (index >= 0 && index < x.length) acc += taps[half - phase + d * up] * x[(int) index];
            }
            out[i] = (float) acc;
        }
        return out;
    }

    /** Where the windows of a {@code total}-sample recording start ({@code audioSpectrogram.windowStarts}). */
    static List<Integer> windowStarts(int total, int length, int count) {
        if (total <= length) return List.of(0);
        int room = total - length;
        if (count <= 1) return List.of(room / 2);
        TreeSet<Integer> starts = new TreeSet<>();
        // rint rounds .5 to the nearest even integer, as Python's round() does.
        for (int i = 0; i < count; i++) starts.add((int) Math.rint((double) i * room / (count - 1)));
        return new ArrayList<>(starts);
    }

    /** In-place radix-2 FFT of {@code re} + i {@code im} (length a power of two). */
    private static void fft(double[] re, double[] im) {
        int n = re.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                double t = re[i]; re[i] = re[j]; re[j] = t;
                t = im[i]; im[i] = im[j]; im[j] = t;
            }
        }
        for (int size = 2; size <= n; size <<= 1) {
            double angle = -2 * Math.PI / size;
            for (int start = 0; start < n; start += size) {
                for (int k = 0; k < size / 2; k++) {
                    double wr = Math.cos(angle * k), wi = Math.sin(angle * k);
                    int a = start + k, b = a + size / 2;
                    double tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }

    /** {@code [nFft / 2 + 1][bands]} triangular HTK mel filters, no area normalisation. */
    static double[][] melFilterbank(int sampleRate, int nFft, int bands, double fMin, double fMax) {
        double[] edges = new double[bands + 2];
        for (int i = 0; i < edges.length; i++) {
            double mel = toMel(fMin) + (toMel(fMax) - toMel(fMin)) * i / (bands + 1);
            edges[i] = 700 * (Math.pow(10, mel / 2595) - 1);
        }
        int bins = nFft / 2 + 1;
        double[][] filters = new double[bins][bands];
        for (int bin = 0; bin < bins; bin++) {
            double hz = sampleRate / 2.0 * bin / (bins - 1);
            for (int b = 0; b < bands; b++) {
                double rising = (hz - edges[b]) / (edges[b + 1] - edges[b]);
                double falling = (edges[b + 2] - hz) / (edges[b + 2] - edges[b + 1]);
                filters[bin][b] = Math.max(0, Math.min(rising, falling));
            }
        }
        return filters;
    }

    private static double toMel(double hz) {
        return 2595 * Math.log10(1 + hz / 700);
    }

    /** The standardised log-mel spectrogram of one window, {@code [bands][frames]} flattened row by row. */
    static float[] spectrogram(float[] window, Recipe r, double[][] filters) {
        double[] hann = new double[r.nFft()];
        for (int n = 0; n < r.nFft(); n++) hann[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / r.nFft());
        float[] out = new float[r.bands() * r.frames()];
        double[] re = new double[r.nFft()], im = new double[r.nFft()];
        int bins = r.nFft() / 2 + 1;
        double[] power = new double[bins];
        for (int f = 0; f < r.frames(); f++) {
            for (int n = 0; n < r.nFft(); n++) {
                re[n] = window[f * r.hop() + n] * hann[n];
                im[n] = 0;
            }
            fft(re, im);
            for (int k = 0; k < bins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
            for (int b = 0; b < r.bands(); b++) {
                double mel = 0;
                for (int k = 0; k < bins; k++) mel += power[k] * filters[k][b];
                float logMel = (float) Math.log(mel + r.logFloor());
                out[b * r.frames() + f] = (float) ((logMel - r.mean()) / r.std());
            }
        }
        return out;
    }
}

public final class OmniTrainClient implements AutoCloseable {
    /** A box as fractions (0..1) of the image: the top-left corner and the size. */
    public record Box(double x, double y, double w, double h) {}

    /** One object found in an image. */
    public record Detection(String label, double confidence, Box box) {}

    /** One word of a tagged text, the tag the model gave it and how sure it was (0..1). */
    public record TaggedWord(String token, String tag, double confidence) {}

    private final OrtEnvironment env;
    private final OrtSession session;
    private final JsonNode inputSpec;
    private final JsonNode outputSpec;
    private final String onnxInputName;
    private final String onnxOutputName;
    private final AudioFrontEnd.Recipe audioRecipe;
    private final TextFrontEnd.Recipe textRecipe;
    private final TagFrontEnd.Recipe tagRecipe;

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
        this.audioRecipe = "audio".equals(inputSpec.path("type").asText()) && inputSpec.has("audioSpectrogram")
            ? AudioFrontEnd.Recipe.of(inputSpec.get("audioSpectrogram")) : null;
        if (audioRecipe != null && !"log_mel_v1".equals(audioRecipe.recipe())) {
            throw new UnsupportedOperationException(
                "Unknown audio recipe '" + audioRecipe.recipe() + "'. " +
                "See preprocessing.json's inputs[0].audioSpectrogram and adapt this client.");
        }
        this.textRecipe = "text".equals(inputSpec.path("type").asText()) && inputSpec.has("textTokens")
            ? new TextFrontEnd.Recipe(inputSpec.get("textTokens")) : null;
        if (textRecipe != null && !"word_ids_v1".equals(textRecipe.recipe)) {
            throw new UnsupportedOperationException(
                "Unknown text recipe '" + textRecipe.recipe + "'. " +
                "See preprocessing.json's inputs[0].textTokens and adapt this client.");
        }
        this.tagRecipe = "text".equals(inputSpec.path("type").asText()) && inputSpec.has("tagWords")
                && "tags".equals(outputSpec.path("type").asText())
            ? new TagFrontEnd.Recipe(inputSpec.get("tagWords")) : null;
        if (tagRecipe != null && !"tag_word_ids_v1".equals(tagRecipe.recipe)) {
            throw new UnsupportedOperationException(
                "Unknown tagging recipe '" + tagRecipe.recipe + "'. " +
                "See preprocessing.json's inputs[0].tagWords and adapt this client.");
        }
    }

    /** A token classifier: use {@link #tag}, which takes the text itself and returns a tag for every word. */
    public boolean isTagger() {
        return tagRecipe != null;
    }

    /** A text classifier: use {@link #predictText}, which takes the text itself. */
    public boolean isText() {
        return textRecipe != null;
    }

    /** An audio classifier: {@link #predict} takes the path of a WAV file. */
    public boolean isAudio() {
        return audioRecipe != null;
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

    /**
     * Runs inference and returns {@code {className: probability}} sorted descending (classification), or
     * {@code {"value": x}} (regression). {@code imagePath} is an image file, or for an audio classifier a WAV file.
     */
    public Map<String, Double> predict(Path imagePath, int topK) throws OrtException, IOException {
        if (isDetector()) {
            throw new UnsupportedOperationException("This model is an object detector: call detect(), which returns boxes.");
        }
        if (isAudio()) return classifyAudio(imagePath, topK);
        if (isText()) {
            throw new UnsupportedOperationException("This model takes text: call predictText(text, topK).");
        }
        if (isTagger()) {
            throw new UnsupportedOperationException("This model tags every word of a text: call tag(text), which returns a tag for each.");
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

    /** Class probabilities of a text: the logits of its one row through a softmax, sorted descending. */
    public Map<String, Double> predictText(String text, int topK) throws OrtException {
        if (textRecipe == null) {
            throw new UnsupportedOperationException("This model does not take text: call predict() with a file path.");
        }
        long[] ids = TextFrontEnd.tokenIds(text, textRecipe);
        float[] logits;
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(ids), new long[] {1, ids.length});
             OrtSession.Result result = session.run(Collections.singletonMap(onnxInputName, tensor))) {
            OnnxValue outputValue = result.get(onnxOutputName)
                .orElseThrow(() -> new IllegalStateException("Session produced no output named '" + onnxOutputName + "'"));
            logits = flatten(outputValue.getValue());
        }
        return decode(logits, topK);
    }

    /**
     * One {@link TaggedWord} per word of {@code text} (the runs of characters that are not whitespace, as {@code str.split()}
     * cuts them): the highest of the word's row of logits and its softmax probability. The first maxTokens words are read;
     * a later word was not, and gets the recipe's fallbackTag with confidence 0.
     */
    public List<TaggedWord> tag(String text) throws OrtException {
        if (tagRecipe == null) {
            throw new UnsupportedOperationException("This model does not tag words: call predict() or predictText().");
        }
        List<String> words = TagFrontEnd.words(text, tagRecipe);
        if (words.isEmpty()) return new ArrayList<>();
        long[] ids = TagFrontEnd.wordIds(words, tagRecipe);
        float[] logits;
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(ids), new long[] {1, ids.length});
             OrtSession.Result result = session.run(Collections.singletonMap(onnxInputName, tensor))) {
            OnnxValue outputValue = result.get(onnxOutputName)
                .orElseThrow(() -> new IllegalStateException("Session produced no output named '" + onnxOutputName + "'"));
            logits = flatten(outputValue.getValue());
        }
        List<String> classes = new ArrayList<>();
        outputSpec.path("classes").forEach(n -> classes.add(n.asText()));
        List<TaggedWord> tagged = new ArrayList<>(words.size());
        for (int i = 0; i < words.size(); i++) {
            if (i >= tagRecipe.maxTokens) {
                tagged.add(new TaggedWord(words.get(i), tagRecipe.fallbackTag, 0.0));
                continue;
            }
            float[] row = java.util.Arrays.copyOfRange(logits, i * classes.size(), (i + 1) * classes.size());
            float[] probs = softmax(row);
            int best = 0;
            for (int k = 1; k < probs.length; k++) {
                if (probs[k] > probs[best]) best = k;
            }
            tagged.add(new TaggedWord(words.get(i), classes.get(best), Math.round(probs[best] * 10000.0) / 10000.0));
        }
        return tagged;
    }

    /** Class probabilities of a recording: each window's logits through a softmax, averaged over the windows. */
    private Map<String, Double> classifyAudio(Path path, int topK) throws OrtException, IOException {
        AudioFrontEnd.Recipe r = audioRecipe;
        int[] rate = new int[1];
        float[][] channels = AudioFrontEnd.readWav(Files.readAllBytes(path), rate);
        int frames = channels[0].length;
        // The channels' mean, at double precision as the platform does it.
        double[] mono = new double[frames];
        for (float[] channel : channels) for (int i = 0; i < frames; i++) mono[i] += channel[i];
        for (int i = 0; i < frames; i++) mono[i] /= channels.length;
        float[] wave = AudioFrontEnd.resample(mono, rate[0], r.sampleRate());
        if (wave.length == 0) throw new IOException("The recording has no samples");

        int length = r.windowSamples();
        double[][] filters = AudioFrontEnd.melFilterbank(r.sampleRate(), r.nFft(), r.bands(), r.fMin(), r.fMax());
        List<Integer> starts = AudioFrontEnd.windowStarts(wave.length, length, r.windowsPerAudio());
        int size = r.bands() * r.frames();
        float[] batch = new float[starts.size() * size];
        for (int w = 0; w < starts.size(); w++) {
            float[] window = new float[length]; // zero-padded at the end when the recording runs out
            System.arraycopy(wave, starts.get(w), window, 0, Math.min(length, wave.length - starts.get(w)));
            System.arraycopy(AudioFrontEnd.spectrogram(window, r, filters), 0, batch, w * size, size);
        }

        long[] shape = {starts.size(), 1, r.bands(), r.frames()};
        float[] logits;
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(batch), shape);
             OrtSession.Result result = session.run(Collections.singletonMap(onnxInputName, tensor))) {
            OnnxValue outputValue = result.get(onnxOutputName)
                .orElseThrow(() -> new IllegalStateException("Session produced no output named '" + onnxOutputName + "'"));
            logits = flatten(outputValue.getValue());
        }
        List<String> classes = new ArrayList<>();
        outputSpec.path("classes").forEach(n -> classes.add(n.asText()));
        double[] mean = new double[classes.size()];
        for (int w = 0; w < starts.size(); w++) {
            float[] probs = softmax(java.util.Arrays.copyOfRange(logits, w * classes.size(), (w + 1) * classes.size()));
            for (int c = 0; c < classes.size(); c++) mean[c] += probs[c] / (double) starts.size();
        }
        List<Integer> order = new ArrayList<>();
        for (int c = 0; c < classes.size(); c++) order.add(c);
        order.sort((a, b) -> Double.compare(mean[b], mean[a]));
        LinkedHashMap<String, Double> out = new LinkedHashMap<>();
        for (int i = 0; i < Math.min(topK, order.size()); i++) {
            out.put(classes.get(order.get(i)), Math.round(mean[order.get(i)] * 10000.0) / 10000.0);
        }
        return out;
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
