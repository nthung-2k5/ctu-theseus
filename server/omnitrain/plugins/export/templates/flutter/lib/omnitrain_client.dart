// OmniTrain-generated inference client.
//
// Wraps the exported ONNX model with the preprocessing/postprocessing
// the platform applied at training time, read from preprocessing.json — the platform's
// exported ONNX graph is the bare model only (no resize/normalize, no
// class-index decoding), so this client reconstructs that step from the
// manifest rather than assuming a fixed tensor layout: input/output tensor
// names are matched against the graph's actual names at runtime, not
// hardcoded, since ONNX export doesn't guarantee a stable naming convention
// across exporter versions.
//
// Implements preprocessing for `image` inputs with `category` outputs (image
// classification) and for a `detections` output (object detection, YOLO): the
// ONNX graph returns every candidate box, so for a detector this client does
// the confidence filter, per-class non-maximum suppression and the undoing of
// the letterbox, and `detect()` returns the boxes. Other input types throw with
// a pointer to preprocessing.json so you can adapt preprocessing yourself; see
// README.md.
//
// An `audio` input that carries an `audioSpectrogram` recipe (audio classification)
// is supported too: `predict()` takes the bytes of a WAV file, makes the log-mel
// spectrograms the recipe describes with no audio library, and averages the
// windows' probabilities.
//
// A `text` input that carries a `textTokens` recipe (text classification) is supported as well:
// `predictText()` takes the text itself, splits it into words and looks them up in the
// vocabulary that ships in preprocessing.json.
//
// A `text` input that carries a `tagWords` recipe and a `tags` output (token classification) is
// supported too: `tag()` takes the text itself and returns a tag for every word.
//
// model.onnx / preprocessing.json are loaded as Flutter assets (declared in
// pubspec.yaml under assets/), not filesystem paths — that's why this client
// reads them through rootBundle rather than dart:io.

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

/// `inputs[0].audioSpectrogram` (recipe "log_mel_v1"): every number the spectrogram needs.
class AudioRecipe {
  AudioRecipe({
    required this.recipe,
    required this.sampleRate,
    required this.windowSamples,
    required this.windowsPerAudio,
    required this.frames,
    required this.nFft,
    required this.hop,
    required this.bands,
    required this.fMin,
    required this.fMax,
    required this.logFloor,
    required this.mean,
    required this.std,
  });

  factory AudioRecipe.fromJson(Map<String, dynamic> json) {
    final stft = json['stft'] as Map<String, dynamic>;
    final mel = json['mel'] as Map<String, dynamic>;
    final norm = json['normalization'] as Map<String, dynamic>;
    return AudioRecipe(
      recipe: json['recipe'] as String,
      sampleRate: (json['sampleRate'] as num).toInt(),
      windowSamples: (json['windowSamples'] as num).toInt(),
      windowsPerAudio: (json['windowsPerAudio'] as num).toInt(),
      frames: (json['frames'] as num).toInt(),
      nFft: (stft['nFft'] as num).toInt(),
      hop: (stft['hopLength'] as num).toInt(),
      bands: (mel['bands'] as num).toInt(),
      fMin: (mel['fMin'] as num).toDouble(),
      fMax: (mel['fMax'] as num).toDouble(),
      logFloor: (json['logFloor'] as num?)?.toDouble() ?? 1e-6,
      mean: (norm['mean'] as num).toDouble(),
      std: (norm['std'] as num).toDouble(),
    );
  }

  final String recipe;
  final int sampleRate;
  final int windowSamples;
  final int windowsPerAudio;
  final int frames;
  final int nFft;
  final int hop;
  final int bands;
  final double fMin;
  final double fMax;
  final double logFloor;
  final double mean;
  final double std;
}

/// `inputs[0].textTokens` (recipe "word_ids_v1"): how a text becomes word ids.
class TextRecipe {
  TextRecipe({
    required this.recipe,
    required this.lowerFrom,
    required this.lowerTo,
    required this.lowerAdd,
    required this.separatorRanges,
    required this.maxTokens,
    required this.padId,
    required this.unknownId,
    required this.firstWordId,
    required this.vocabulary,
  }) {
    for (final (lo, hi) in separatorRanges) {
      for (var unit = lo; unit <= math.min(hi, 0xFFFF); unit++) {
        separators[unit] = 1;
      }
    }
    for (var i = 0; i < vocabulary.length; i++) {
      ids[vocabulary[i]] = firstWordId + i;
    }
  }

  factory TextRecipe.fromJson(Map<String, dynamic> json) {
    final lower = json['lowercase'] as Map<String, dynamic>;
    final ranges = [
      for (final r in json['separatorRanges'] as List) (r as List).map((e) => (e as num).toInt()).toList(),
    ];
    return TextRecipe(
      recipe: json['recipe'] as String,
      lowerFrom: (lower['from'] as num).toInt(),
      lowerTo: (lower['to'] as num).toInt(),
      lowerAdd: (lower['add'] as num).toInt(),
      separatorRanges: [for (final r in ranges) (r[0], r[1])],
      maxTokens: (json['maxTokens'] as num).toInt(),
      padId: (json['padId'] as num).toInt(),
      unknownId: (json['unknownId'] as num).toInt(),
      firstWordId: (json['firstWordId'] as num).toInt(),
      vocabulary: [for (final w in json['vocabulary'] as List) w as String],
    );
  }

  final String recipe;
  final int lowerFrom;
  final int lowerTo;
  final int lowerAdd;
  final List<(int, int)> separatorRanges;
  final int maxTokens;
  final int padId;
  final int unknownId;
  final int firstWordId;
  final List<String> vocabulary;

  /// Which UTF-16 code units separate tokens, as a lookup table built from the ranges.
  final Uint8List separators = Uint8List(0x10000);

  /// A token's id.
  final Map<String, int> ids = {};
}

/// The tokens of a text: the ASCII letters A-Z lower-cased, cut at every run of separator code units.
List<String> tokenizeText(String text, TextRecipe recipe) {
  final tokens = <String>[];
  final current = StringBuffer();
  for (var i = 0; i < text.length; i++) {
    final unit = text.codeUnitAt(i);
    if (recipe.separators[unit] == 1) {
      if (current.isNotEmpty) tokens.add(current.toString());
      current.clear();
    } else {
      current.writeCharCode(unit >= recipe.lowerFrom && unit <= recipe.lowerTo ? unit + recipe.lowerAdd : unit);
    }
  }
  if (current.isNotEmpty) tokens.add(current.toString());
  return tokens;
}

/// `maxTokens` word ids: the first maxTokens tokens looked up in the vocabulary, padded at the end.
Int64List textTokenIds(String text, TextRecipe recipe) {
  final ids = Int64List(recipe.maxTokens)..fillRange(0, recipe.maxTokens, recipe.padId);
  final tokens = tokenizeText(text, recipe);
  for (var i = 0; i < math.min(tokens.length, recipe.maxTokens); i++) {
    ids[i] = recipe.ids[tokens[i]] ?? recipe.unknownId;
  }
  return ids;
}

/// `inputs[0].tagWords` (recipe "tag_word_ids_v1"): how a text becomes one id per word.
class TagRecipe {
  TagRecipe({
    required this.recipe,
    required this.lowerFrom,
    required this.lowerTo,
    required this.lowerAdd,
    required this.whitespaceRanges,
    required this.maxTokens,
    required this.padId,
    required this.unknownId,
    required this.firstWordId,
    required this.fallbackTag,
    required this.vocabulary,
  }) {
    for (final (lo, hi) in whitespaceRanges) {
      for (var unit = lo; unit <= math.min(hi, 0xFFFF); unit++) {
        whitespace[unit] = 1;
      }
    }
    for (var i = 0; i < vocabulary.length; i++) {
      ids[vocabulary[i]] = firstWordId + i;
    }
  }

  factory TagRecipe.fromJson(Map<String, dynamic> json) {
    final lower = json['lowercase'] as Map<String, dynamic>;
    final ranges = [
      for (final r in json['whitespaceRanges'] as List) (r as List).map((e) => (e as num).toInt()).toList(),
    ];
    return TagRecipe(
      recipe: json['recipe'] as String,
      lowerFrom: (lower['from'] as num).toInt(),
      lowerTo: (lower['to'] as num).toInt(),
      lowerAdd: (lower['add'] as num).toInt(),
      whitespaceRanges: [for (final r in ranges) (r[0], r[1])],
      maxTokens: (json['maxTokens'] as num).toInt(),
      padId: (json['padId'] as num).toInt(),
      unknownId: (json['unknownId'] as num).toInt(),
      firstWordId: (json['firstWordId'] as num).toInt(),
      fallbackTag: json['fallbackTag'] as String? ?? 'O',
      vocabulary: [for (final w in json['vocabulary'] as List) w as String],
    );
  }

  final String recipe;
  final int lowerFrom;
  final int lowerTo;
  final int lowerAdd;
  final List<(int, int)> whitespaceRanges;
  final int maxTokens;
  final int padId;
  final int unknownId;
  final int firstWordId;
  final String fallbackTag;
  final List<String> vocabulary;

  /// Which UTF-16 code units separate words (what `str.split()` cuts at), as a lookup table built from the ranges.
  final Uint8List whitespace = Uint8List(0x10000);

  /// A word's id.
  final Map<String, int> ids = {};
}

/// One word of a tagged text, the tag the model gave it and how sure it was (0..1).
class TaggedWord {
  const TaggedWord(this.token, this.tag, this.confidence);
  final String token;
  final String tag;
  final double confidence;
}

/// The words of a text: the runs of code units that are not whitespace.
List<String> splitTagWords(String text, TagRecipe recipe) {
  final words = <String>[];
  var start = -1;
  for (var i = 0; i < text.length; i++) {
    if (recipe.whitespace[text.codeUnitAt(i)] == 1) {
      if (start >= 0) words.add(text.substring(start, i));
      start = -1;
    } else if (start < 0) {
      start = i;
    }
  }
  if (start >= 0) words.add(text.substring(start));
  return words;
}

/// `maxTokens` word ids: the first maxTokens words, ASCII capitals lower-cased, looked up in the vocabulary, padded at the end.
Int64List tagWordIds(List<String> words, TagRecipe recipe) {
  final ids = Int64List(recipe.maxTokens)..fillRange(0, recipe.maxTokens, recipe.padId);
  for (var i = 0; i < math.min(words.length, recipe.maxTokens); i++) {
    final form = StringBuffer();
    for (final unit in words[i].codeUnits) {
      form.writeCharCode(unit >= recipe.lowerFrom && unit <= recipe.lowerTo ? unit + recipe.lowerAdd : unit);
    }
    ids[i] = recipe.ids[form.toString()] ?? recipe.unknownId;
  }
  return ids;
}

/// A tag for each word from the model's `[maxTokens, tags]` logits (flattened): the highest of the word's row and its
/// softmax probability. A word past maxTokens was not read and gets the recipe's fallbackTag with confidence 0.
List<TaggedWord> decodeTags(List<double> logits, List<String> words, List<String> classes, TagRecipe recipe) {
  return [
    for (var i = 0; i < words.length; i++)
      if (i >= recipe.maxTokens)
        TaggedWord(words[i], recipe.fallbackTag, 0.0)
      else
        _bestTag(words[i], _softmax(logits.sublist(i * classes.length, (i + 1) * classes.length)), classes),
  ];
}

TaggedWord _bestTag(String word, List<double> probs, List<String> classes) {
  var best = 0;
  for (var k = 1; k < probs.length; k++) {
    if (probs[k] > probs[best]) best = k;
  }
  return TaggedWord(word, classes[best], (probs[best] * 10000).round() / 10000);
}

class PreprocessingInput {
  PreprocessingInput({
    required this.type,
    required this.column,
    this.audio,
    this.text,
    this.tags,
    this.imageSize,
    this.imageResizeMode,
    this.letterboxSize,
    this.padValue = 114,
    this.normMean,
    this.normStd,
  });

  factory PreprocessingInput.fromJson(Map<String, dynamic> json) {
    final norm = json['imageNormalization'] as Map<String, dynamic>?;
    return PreprocessingInput(
      type: json['type'] as String,
      column: json['column'] as String,
      audio: json['audioSpectrogram'] == null
          ? null
          : AudioRecipe.fromJson(json['audioSpectrogram'] as Map<String, dynamic>),
      text: json['textTokens'] == null ? null : TextRecipe.fromJson(json['textTokens'] as Map<String, dynamic>),
      tags: json['tagWords'] == null ? null : TagRecipe.fromJson(json['tagWords'] as Map<String, dynamic>),
      imageSize: json['imageSize'] as Map<String, dynamic>?,
      imageResizeMode: (json['imageResize'] as Map<String, dynamic>?)?['mode'] as String?,
      letterboxSize: ((json['imageResize'] as Map<String, dynamic>?)?['size'] as List?)
          ?.map((e) => (e as num).toInt())
          .toList(),
      padValue: ((json['imageResize'] as Map<String, dynamic>?)?['padValue'] as num?)?.toInt() ?? 114,
      normMean: (norm?['mean'] as List?)?.map((e) => (e as num).toDouble()).toList(),
      normStd: (norm?['std'] as List?)?.map((e) => (e as num).toDouble()).toList(),
    );
  }

  final String type;
  final String column;
  /// `audioSpectrogram`, for an audio classifier.
  final AudioRecipe? audio;
  /// `textTokens`, for a text classifier.
  final TextRecipe? text;
  /// `tagWords`, for a token classifier.
  final TagRecipe? tags;
  final Map<String, dynamic>? imageSize;
  /// `imageResize.mode`. Null: squash the whole image to width x height.
  final String? imageResizeMode;
  /// `imageResize.size` ([height, width]) and `padValue`, for the `letterbox` mode.
  final List<int>? letterboxSize;
  final int padValue;
  final List<double>? normMean;
  final List<double>? normStd;
}

class PreprocessingOutput {
  PreprocessingOutput({
    required this.column,
    this.type = '',
    this.classes,
    this.confidenceThreshold = 0.25,
    this.iouThreshold = 0.7,
  });

  factory PreprocessingOutput.fromJson(Map<String, dynamic> json) {
    final yolo = json['yoloOutput'] as Map<String, dynamic>?;
    return PreprocessingOutput(
      column: json['column'] as String,
      type: json['type'] as String? ?? '',
      classes: (json['classes'] as List?)?.map((e) => e as String).toList(),
      confidenceThreshold: (yolo?['confidenceThreshold'] as num?)?.toDouble() ?? 0.25,
      iouThreshold: (yolo?['iouThreshold'] as num?)?.toDouble() ?? 0.7,
    );
  }

  final String column;
  final String type;
  final List<String>? classes;
  /// `yoloOutput` defaults of a detector.
  final double confidenceThreshold;
  final double iouThreshold;
}

/// A box as fractions (0..1) of the image: the top-left corner and the size.
class Box {
  const Box(this.x, this.y, this.w, this.h);

  final double x;
  final double y;
  final double w;
  final double h;
}

/// One object found in an image.
class Detection {
  const Detection(this.label, this.confidence, this.box);

  final String label;
  final double confidence;
  final Box box;
}

/// The image as a detector's input, and what is needed to map its boxes back.
class Letterboxed {
  const Letterboxed(this.chw, this.height, this.width, this.scale, this.left, this.top, this.originalWidth, this.originalHeight);

  final Float32List chw;
  final int height;
  final int width;
  final double scale;
  final int left;
  final int top;
  final int originalWidth;
  final int originalHeight;
}

int _roundHalfEvenInt(double x) {
  final floor = x.floor();
  final diff = x - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor.isEven ? floor : floor + 1;
}

/// For `imageResize.mode == 'letterbox'`: scale so the LONGER edge is `size`, pad the shorter one, centred, with
/// `pad` grey up to a height x width square (the arithmetic of ultralytics.data.augment.LetterBox). RGB, 0..1,
/// no mean/std.
Letterboxed letterboxImage(img.Image decoded, {required int height, required int width, int pad = 114}) {
  final originalWidth = decoded.width;
  final originalHeight = decoded.height;
  final scale = math.min(height / originalHeight, width / originalWidth);
  final newWidth = _roundHalfEvenInt(originalWidth * scale);
  final newHeight = _roundHalfEvenInt(originalHeight * scale);
  final left = _roundHalfEvenInt((width - newWidth) / 2 - 0.1);
  final top = _roundHalfEvenInt((height - newHeight) / 2 - 0.1);
  final resized = (newWidth == originalWidth && newHeight == originalHeight)
      ? decoded
      : img.copyResize(decoded, width: newWidth, height: newHeight, interpolation: img.Interpolation.linear);

  final chw = Float32List(3 * height * width)..fillRange(0, 3 * height * width, pad / 255.0);
  for (var y = 0; y < newHeight; y++) {
    for (var x = 0; x < newWidth; x++) {
      final px = resized.getPixel(x, y);
      final at = ((top + y) * width) + left + x;
      chw[(0 * height * width) + at] = px.r / 255.0;
      chw[(1 * height * width) + at] = px.g / 255.0;
      chw[(2 * height * width) + at] = px.b / 255.0;
    }
  }
  return Letterboxed(chw, height, width, scale, left, top, originalWidth, originalHeight);
}

/// Indices of the boxes to keep, most confident first: within each class, a box is dropped when it overlaps a
/// more confident one by more than [iouThreshold] (intersection over union). Boxes are [x1, y1, x2, y2].
List<int> nonMaxSuppression(List<List<double>> boxes, List<double> scores, List<int> classes, double iouThreshold) {
  final kept = <int>[];
  for (final label in classes.toSet()) {
    var order = [for (var i = 0; i < classes.length; i++) if (classes[i] == label) i]
      ..sort((a, b) {
        final byScore = scores[b].compareTo(scores[a]);
        return byScore != 0 ? byScore : a.compareTo(b);
      });
    while (order.isNotEmpty) {
      final first = order.first;
      kept.add(first);
      final a = boxes[first];
      final areaFirst = (a[2] - a[0]) * (a[3] - a[1]);
      order = [
        for (final i in order.skip(1))
          if (_overlap(a, areaFirst, boxes[i]) <= iouThreshold) i,
      ];
    }
  }
  return kept..sort((a, b) => scores[b].compareTo(scores[a]));
}

double _overlap(List<double> a, double areaA, List<double> b) {
  final width = math.min(a[2], b[2]) - math.max(a[0], b[0]);
  final height = math.min(a[3], b[3]) - math.max(a[1], b[1]);
  final intersection = math.max(width, 0) * math.max(height, 0);
  final union = areaA + ((b[2] - b[0]) * (b[3] - b[1])) - intersection;
  return intersection / math.max(union, 1e-9);
}

/// Turns a detector's raw output into boxes. [table] is the `[4 + classes][candidates]` matrix (centre x, centre y,
/// width, height, then a score per class, in pixels of the letterboxed input), [input] what [letterboxImage] returned.
List<Detection> decodeDetections(
  List<List<double>> table,
  Letterboxed input,
  List<String> classes, {
  required double confidence,
  required double iou,
  int maxDetections = 300,
}) {
  final candidates = table[0].length;
  final boxes = <List<double>>[];
  final scores = <double>[];
  final labels = <int>[];
  for (var i = 0; i < candidates; i++) {
    var best = 0;
    var bestScore = double.negativeInfinity;
    for (var c = 0; c < classes.length; c++) {
      final score = table[4 + c][i];
      if (score > bestScore) {
        bestScore = score;
        best = c;
      }
    }
    if (bestScore < confidence) continue;
    final cx = table[0][i], cy = table[1][i], w = table[2][i], h = table[3][i];
    boxes.add([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
    scores.add(bestScore);
    labels.add(best);
  }

  double clamp(double v, double max) => math.min(math.max(v, 0), max);
  double round(double v, int places) {
    final factor = math.pow(10, places).toDouble();
    return (v * factor).roundToDouble() / factor;
  }

  return [
    for (final index in nonMaxSuppression(boxes, scores, labels, iou).take(maxDetections))
      () {
        // Undo the letterbox: remove the padding, divide by the scale, clip to the image.
        final b = boxes[index];
        final x1 = clamp((math.max(b[0], 0) - input.left) / input.scale, input.originalWidth.toDouble());
        final x2 = clamp((math.max(b[2], 0) - input.left) / input.scale, input.originalWidth.toDouble());
        final y1 = clamp((math.max(b[1], 0) - input.top) / input.scale, input.originalHeight.toDouble());
        final y2 = clamp((math.max(b[3], 0) - input.top) / input.scale, input.originalHeight.toDouble());
        return Detection(
          classes[labels[index]],
          round(scores[index], 4),
          Box(
            round(x1 / input.originalWidth, 6),
            round(y1 / input.originalHeight, 6),
            round((x2 - x1) / input.originalWidth, 6),
            round((y2 - y1) / input.originalHeight, 6),
          ),
        );
      }(),
  ];
}

String _matchTensorName(List<String> candidates, String preferred) {
  if (candidates.length == 1) return candidates.first;
  final match = candidates.firstWhere(
    (n) => n == preferred || n.startsWith('$preferred::') || n.startsWith('${preferred}_'),
    orElse: () => '',
  );
  if (match.isEmpty) {
    throw StateError(
      "Could not match an ONNX tensor to feature '$preferred' among $candidates. "
      'Inspect preprocessing.json and adapt this client if the model uses an unrecognized naming scheme.',
    );
  }
  return match;
}

/// ONNX tensor output values arrive as arbitrarily-nested Dart lists
/// (shaped per the tensor's dimensions) — flatten to a single list of
/// doubles regardless of rank.
List<double> _flatten(dynamic value) {
  final out = <double>[];
  void go(dynamic v) {
    if (v is List) {
      for (final e in v) {
        go(e);
      }
    } else if (v is num) {
      out.add(v.toDouble());
    }
  }

  go(value);
  return out;
}

// -- Audio -------------------------------------------------------------------------------------------------------
// `inputs[0].audioSpectrogram` spelled out. The platform computes the same thing in numpy; the arithmetic is plain
// enough (a polyphase resampler, a radix-2 FFT, a mel filterbank) to follow in any language.

/// A WAV file's samples, one list per channel, in [-1, 1], and its sample rate.
({List<Float32List> channels, int rate}) readWav(Uint8List file) {
  String tag(int at) => String.fromCharCodes(file.sublist(at, at + 4));
  if (file.length < 12 || tag(0) != 'RIFF' || tag(8) != 'WAVE') {
    throw const FormatException('Not a WAV file (only WAV is supported)');
  }
  final view = ByteData.sublistView(file);
  var code = 0, channels = 0, rate = 0, bits = 0, dataAt = -1, dataSize = 0;
  var at = 12;
  while (at + 8 <= file.length) {
    final size = view.getUint32(at + 4, Endian.little);
    final body = at + 8;
    if (tag(at) == 'fmt ' && size >= 16) {
      code = view.getUint16(body, Endian.little);
      if (code == 0xFFFE && size >= 26) code = view.getUint16(body + 24, Endian.little);
      channels = view.getUint16(body + 2, Endian.little);
      rate = view.getUint32(body + 4, Endian.little);
      bits = view.getUint16(body + 14, Endian.little);
    } else if (tag(at) == 'data') {
      dataAt = body;
      dataSize = math.min(size, file.length - body); // a streamed file may claim more than it holds
      break;
    }
    at = body + size + (size & 1);
  }
  if (channels < 1 || dataAt < 0 || dataSize == 0) {
    throw const FormatException('The WAV file has no format or no samples');
  }
  final width = bits >> 3;
  if (width < 1) throw const FormatException('The WAV file has no channels');
  final frames = dataSize ~/ (width * channels);

  double read(int offset) {
    if (code == 1 && bits == 8) return (view.getUint8(offset) - 128) / 128;
    if (code == 1 && bits == 16) return view.getInt16(offset, Endian.little) / 32767;
    if (code == 1 && bits == 24) {
      final v = view.getUint8(offset) | (view.getUint8(offset + 1) << 8) | (view.getUint8(offset + 2) << 16);
      return (v >= 0x800000 ? v - 0x1000000 : v) / 8388607;
    }
    if (code == 1 && bits == 32) return view.getInt32(offset, Endian.little) / 2147483647;
    if (code == 3 && bits == 32) return view.getFloat32(offset, Endian.little);
    if (code == 3 && bits == 64) return view.getFloat64(offset, Endian.little);
    throw UnsupportedError('Unsupported WAV encoding (format $code, $bits bits)');
  }

  final out = [for (var c = 0; c < channels; c++) Float32List(frames)];
  for (var f = 0; f < frames; f++) {
    for (var c = 0; c < channels; c++) {
      out[c][f] = read(dataAt + (f * channels + c) * width);
    }
  }
  return (channels: out, rate: rate);
}

int _gcd(int a, int b) => b == 0 ? a : _gcd(b, a % b);

/// Modified Bessel function of the first kind, order 0 (the Kaiser window's building block).
double _besselI0(double x) {
  var sum = 1.0, term = 1.0;
  for (var k = 1; k < 500; k++) {
    term *= math.pow(x / (2 * k), 2);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return sum;
}

double _sinc(double v) => v == 0 ? 1 : math.sin(math.pi * v) / (math.pi * v);

/// `scipy.signal.resample_poly(x, up, down)`: out[i] = sum over k in -H..H of h[H + k] * x[(i * down + k) / up],
/// counting the k for which i * down + k is a multiple of `up` and the index is inside the recording. `h` is a
/// Kaiser-windowed (beta 5) sinc low-pass of 2H + 1 taps, H = 10 * max(up, down), scaled to a gain of `up`.
Float32List resampleAudio(Float64List x, int rate, int target) {
  final divisor = _gcd(rate, target);
  final up = target ~/ divisor;
  final down = rate ~/ divisor;
  final big = math.max(up, down);
  final half = 10 * big;
  final taps = Float64List(2 * half + 1);
  final i0Beta = _besselI0(5);
  var total = 0.0;
  for (var n = 0; n <= 2 * half; n++) {
    final ratio = (n - half) / half;
    final window = _besselI0(5 * math.sqrt(math.max(1 - ratio * ratio, 0))) / i0Beta;
    taps[n] = _sinc((n - half) / big) / big * window;
    total += taps[n];
  }
  for (var n = 0; n <= 2 * half; n++) {
    taps[n] *= up / total;
  }

  final outLength = (x.length * up + down - 1) ~/ down;
  final out = Float32List(outLength);
  for (var i = 0; i < outLength; i++) {
    final position = i * down;
    final anchor = position ~/ up;
    final phase = position % up;
    final first = -((half - phase) ~/ up);
    final last = (half + phase) ~/ up;
    var acc = 0.0;
    for (var d = first; d <= last; d++) {
      final index = anchor + d;
      if (index >= 0 && index < x.length) acc += taps[half - phase + d * up] * x[index];
    }
    out[i] = acc;
  }
  return out;
}

/// Where the windows of a [total]-sample recording start (`audioSpectrogram.windowStarts`).
List<int> audioWindowStarts(int total, int length, int count) {
  if (total <= length) return [0];
  final room = total - length;
  if (count <= 1) return [room ~/ 2];
  final starts = <int>{for (var i = 0; i < count; i++) _roundHalfEvenInt(i * room / (count - 1))}.toList()..sort();
  return starts;
}

/// In-place radix-2 FFT of [re] + i [im] (length a power of two).
void _fft(Float64List re, Float64List im) {
  final n = re.length;
  for (var i = 1, j = 0; i < n; i++) {
    var bit = n >> 1;
    for (; (j & bit) != 0; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      final tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      final ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (var size = 2; size <= n; size <<= 1) {
    final angle = -2 * math.pi / size;
    for (var start = 0; start < n; start += size) {
      for (var k = 0; k < size ~/ 2; k++) {
        final wr = math.cos(angle * k), wi = math.sin(angle * k);
        final a = start + k, b = a + size ~/ 2;
        final tr = re[b] * wr - im[b] * wi;
        final ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

/// `[nFft / 2 + 1][bands]` triangular HTK mel filters, no area normalisation.
List<Float64List> melFilterbank(int sampleRate, int nFft, int bands, double fMin, double fMax) {
  double toMel(double hz) => 2595 * math.log(1 + hz / 700) / math.ln10;
  final edges = [
    for (var i = 0; i < bands + 2; i++)
      700 * (math.pow(10, (toMel(fMin) + (toMel(fMax) - toMel(fMin)) * i / (bands + 1)) / 2595) - 1),
  ];
  final bins = nFft ~/ 2 + 1;
  return [
    for (var bin = 0; bin < bins; bin++)
      () {
        final hz = sampleRate / 2 * bin / (bins - 1);
        final row = Float64List(bands);
        for (var b = 0; b < bands; b++) {
          final rising = (hz - edges[b]) / (edges[b + 1] - edges[b]);
          final falling = (edges[b + 2] - hz) / (edges[b + 2] - edges[b + 1]);
          row[b] = math.max(0, math.min(rising, falling));
        }
        return row;
      }(),
  ];
}

/// The standardised log-mel spectrogram of one window, `[bands][frames]` flattened row by row.
Float32List audioSpectrogram(Float32List window, AudioRecipe recipe, List<Float64List> filters) {
  final nFft = recipe.nFft, bands = recipe.bands, frames = recipe.frames;
  final hann = Float64List.fromList([for (var n = 0; n < nFft; n++) 0.5 - 0.5 * math.cos(2 * math.pi * n / nFft)]);
  final out = Float32List(bands * frames);
  final re = Float64List(nFft), im = Float64List(nFft);
  final bins = nFft ~/ 2 + 1;
  final power = Float64List(bins);
  final single = Float32List(1);
  for (var f = 0; f < frames; f++) {
    for (var n = 0; n < nFft; n++) {
      re[n] = window[f * recipe.hop + n] * hann[n];
      im[n] = 0;
    }
    _fft(re, im);
    for (var k = 0; k < bins; k++) {
      power[k] = re[k] * re[k] + im[k] * im[k];
    }
    for (var b = 0; b < bands; b++) {
      var mel = 0.0;
      for (var k = 0; k < bins; k++) {
        mel += power[k] * filters[k][b];
      }
      single[0] = math.log(mel + recipe.logFloor); // rounded to float32 as the platform's float32 spectrogram is
      out[b * frames + f] = (single[0] - recipe.mean) / recipe.std;
    }
  }
  return out;
}

List<double> _softmax(List<double> logits) {
  final maxVal = logits.reduce(math.max);
  final exp = logits.map((v) => math.exp(v - maxVal)).toList();
  final sum = exp.fold<double>(0, (a, b) => a + b);
  return exp.map((v) => v / sum).toList();
}

class OmniTrainClient {
  late OrtSession _session;
  late PreprocessingInput _inputSpec;
  late PreprocessingOutput _outputSpec;
  late String _onnxInputName;
  late String _onnxOutputName;

  /// An object detector: use [detect]; [predict] is for classifiers and regressors.
  bool get isDetector => _outputSpec.type == 'detections';

  /// An audio classifier: [predict] takes the bytes of a WAV file.
  bool get isAudio => _inputSpec.type == 'audio' && _inputSpec.audio != null;

  /// A text classifier: use [predictText], which takes the text itself.
  bool get isText => _inputSpec.type == 'text' && _inputSpec.text != null;

  /// A token classifier: use [tag], which takes the text itself and returns a tag for every word.
  bool get isTagger => _inputSpec.type == 'text' && _inputSpec.tags != null && _outputSpec.type == 'tags';

  Future<void> load({
    String modelAssetPath = 'assets/model.onnx',
    String preprocessingAssetPath = 'assets/preprocessing.json',
  }) async {
    final manifestJson = jsonDecode(await rootBundle.loadString(preprocessingAssetPath)) as Map<String, dynamic>;
    _inputSpec = PreprocessingInput.fromJson((manifestJson['inputs'] as List).first as Map<String, dynamic>);
    _outputSpec = PreprocessingOutput.fromJson((manifestJson['outputs'] as List).first as Map<String, dynamic>);

    final modelBytes = (await rootBundle.load(modelAssetPath)).buffer.asUint8List();
    _session = OrtSession.fromBuffer(modelBytes, OrtSessionOptions());
    _onnxInputName = _matchTensorName(_session.inputNames, _inputSpec.column);
    _onnxOutputName = _matchTensorName(_session.outputNames, _outputSpec.column);
    if (isAudio && _inputSpec.audio!.recipe != 'log_mel_v1') {
      throw UnsupportedError(
        "Unknown audio recipe '${_inputSpec.audio!.recipe}'. "
        'See preprocessing.json inputs[0].audioSpectrogram and adapt this client.',
      );
    }
    if (isText && _inputSpec.text!.recipe != 'word_ids_v1') {
      throw UnsupportedError(
        "Unknown text recipe '${_inputSpec.text!.recipe}'. "
        'See preprocessing.json inputs[0].textTokens and adapt this client.',
      );
    }
    if (isTagger && _inputSpec.tags!.recipe != 'tag_word_ids_v1') {
      throw UnsupportedError(
        "Unknown tagging recipe '${_inputSpec.tags!.recipe}'. "
        'See preprocessing.json inputs[0].tagWords and adapt this client.',
      );
    }
  }

  /// One [TaggedWord] per word of [text] (the runs of characters that are not whitespace, as `str.split()` cuts
  /// them). The first maxTokens words are read; a later word was not, and gets the recipe's fallbackTag.
  Future<List<TaggedWord>> tag(String text) async {
    if (!isTagger) {
      throw UnsupportedError('This model does not tag words: call predict() or predictText().');
    }
    final recipe = _inputSpec.tags!;
    final words = splitTagWords(text, recipe);
    if (words.isEmpty) return const [];
    final inputTensor = OrtValueTensor.createTensorWithDataList(tagWordIds(words, recipe), [1, recipe.maxTokens]);
    final runOptions = OrtRunOptions();
    final outputs = await _session.runAsync(runOptions, {_onnxInputName: inputTensor}, [_onnxOutputName]);
    final raw = outputs?.first?.value;
    inputTensor.release();
    runOptions.release();
    outputs?.first?.release();
    return decodeTags(_flatten(raw), words, _outputSpec.classes ?? const <String>[], recipe);
  }

  /// Class probabilities of a text: the logits of its one row through a softmax, sorted descending.
  Future<Map<String, double>> predictText(String text, {int topK = 5}) async {
    if (!isText) {
      throw UnsupportedError('This model does not take text: call predict() with the bytes of a file.');
    }
    final recipe = _inputSpec.text!;
    final inputTensor = OrtValueTensor.createTensorWithDataList(textTokenIds(text, recipe), [1, recipe.maxTokens]);
    final runOptions = OrtRunOptions();
    final outputs = await _session.runAsync(runOptions, {_onnxInputName: inputTensor}, [_onnxOutputName]);
    final raw = outputs?.first?.value;
    inputTensor.release();
    runOptions.release();
    outputs?.first?.release();

    final probs = _softmax(_flatten(raw));
    final classes = _outputSpec.classes ?? const <String>[];
    final ranked = List.generate(classes.length, (i) => MapEntry(classes[i], probs[i]))
      ..sort((a, b) => b.value.compareTo(a.value));
    return {for (final e in ranked.take(topK)) e.key: (e.value * 10000).round() / 10000};
  }

  /// The recording as a batch of spectrograms, `[windows, 1, bands, frames]` flattened, by `audioSpectrogram`.
  (Float32List, int) _audioSpectrograms(Uint8List bytes) {
    final recipe = _inputSpec.audio!;
    final (:channels, :rate) = readWav(bytes);
    final frames = channels.first.length;
    // The channels' mean, at double precision as the platform does it.
    final mono = Float64List(frames);
    for (final channel in channels) {
      for (var i = 0; i < frames; i++) {
        mono[i] += channel[i];
      }
    }
    for (var i = 0; i < frames; i++) {
      mono[i] /= channels.length;
    }
    final wave = resampleAudio(mono, rate, recipe.sampleRate);
    if (wave.isEmpty) throw const FormatException('The recording has no samples');

    final length = recipe.windowSamples;
    final filters = melFilterbank(recipe.sampleRate, recipe.nFft, recipe.bands, recipe.fMin, recipe.fMax);
    final starts = audioWindowStarts(wave.length, length, recipe.windowsPerAudio);
    final size = recipe.bands * recipe.frames;
    final batch = Float32List(starts.length * size);
    for (var w = 0; w < starts.length; w++) {
      final window = Float32List(length); // zero-padded at the end when the recording runs out
      window.setRange(0, math.min(length, wave.length - starts[w]), wave, starts[w]);
      batch.setRange(w * size, (w + 1) * size, audioSpectrogram(window, recipe, filters));
    }
    return (batch, starts.length);
  }

  /// Class probabilities of a recording: each window's logits through a softmax, averaged over the windows.
  Future<Map<String, double>> _classifyAudio(Uint8List bytes, int topK) async {
    final recipe = _inputSpec.audio!;
    final (batch, windows) = _audioSpectrograms(bytes);
    final inputTensor = OrtValueTensor.createTensorWithDataList(batch, [windows, 1, recipe.bands, recipe.frames]);
    final runOptions = OrtRunOptions();
    final outputs = await _session.runAsync(runOptions, {_onnxInputName: inputTensor}, [_onnxOutputName]);
    final raw = outputs?.first?.value;
    inputTensor.release();
    runOptions.release();
    outputs?.first?.release();

    final logits = _flatten(raw);
    final classes = _outputSpec.classes ?? const <String>[];
    final mean = List<double>.filled(classes.length, 0);
    for (var w = 0; w < windows; w++) {
      final probs = _softmax(logits.sublist(w * classes.length, (w + 1) * classes.length));
      for (var c = 0; c < classes.length; c++) {
        mean[c] += probs[c] / windows;
      }
    }
    final ranked = List.generate(classes.length, (i) => MapEntry(classes[i], mean[i]))
      ..sort((a, b) => b.value.compareTo(a.value));
    return {for (final e in ranked.take(topK)) e.key: (e.value * 10000).round() / 10000};
  }

  (int width, int height) get _imageDims {
    final pp = _inputSpec.imageSize ?? const <String, dynamic>{};
    final width = (pp['width'] as num?)?.toInt() ?? 224;
    final height = (pp['height'] as num?)?.toInt() ?? 224;
    return (width, height);
  }

  /// Rounds .5 to the nearest even integer, as Python's round() does (and torchvision's CenterCrop relies on).
  static int _roundHalfEven(double x) {
    final floor = x.floor();
    final diff = x - floor;
    if (diff < 0.5) return floor;
    if (diff > 0.5) return floor + 1;
    return floor.isEven ? floor : floor + 1;
  }

  Float32List _preprocessImage(Uint8List bytes) {
    final decoded = img.decodeImage(bytes);
    if (decoded == null) throw const FormatException('Could not decode image bytes');
    final (width, height) = _imageDims;
    final mode = _inputSpec.imageResizeMode;
    final img.Image resized;
    if (mode == null) {
      resized = img.copyResize(decoded, width: width, height: height);
    } else if (mode == 'shorter_edge_center_crop') {
      // Scale so the SHORTER edge is `size` (the other edge truncated to a whole pixel), then centre-crop a
      // size x size square: the same arithmetic as torchvision's Resize(size) + CenterCrop(size), which is
      // what the model was evaluated with.
      final size = math.min(width, height);
      final landscape = decoded.width > decoded.height;
      final newWidth = landscape ? decoded.width * size ~/ decoded.height : size;
      final newHeight = landscape ? size : decoded.height * size ~/ decoded.width;
      final scaled = img.copyResize(
        decoded,
        width: newWidth,
        height: newHeight,
        interpolation: img.Interpolation.linear,
      );
      resized = img.copyCrop(
        scaled,
        x: _roundHalfEven((newWidth - size) / 2),
        y: _roundHalfEven((newHeight - size) / 2),
        width: width,
        height: height,
      );
    } else {
      throw UnsupportedError(
        "Unsupported imageResize mode '$mode'. See preprocessing.json inputs[0].imageResize and adapt this client.",
      );
    }

    final mean = _inputSpec.normMean;
    final std = _inputSpec.normStd;
    final chw = Float32List(3 * height * width);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final px = resized.getPixel(x, y);
        var r = px.r / 255.0;
        var g = px.g / 255.0;
        var b = px.b / 255.0;
        if (mean != null && std != null) {
          r = (r - mean[0]) / std[0];
          g = (g - mean[1]) / std[1];
          b = (b - mean[2]) / std[2];
        }
        chw[(0 * height * width) + (y * width) + x] = r;
        chw[(1 * height * width) + (y * width) + x] = g;
        chw[(2 * height * width) + (y * width) + x] = b;
      }
    }
    return chw;
  }

  /// Boxes found in an image, most confident first. [confidence] and [iou] default to the thresholds in
  /// preprocessing.json (`outputs[0].yoloOutput`).
  Future<List<Detection>> detect(
    Uint8List imageBytes, {
    double? confidence,
    double? iou,
    int maxDetections = 300,
  }) async {
    final size = _inputSpec.letterboxSize;
    if (_inputSpec.imageResizeMode != 'letterbox' || size == null || size.length != 2) {
      throw UnsupportedError(
        "Unsupported imageResize mode '${_inputSpec.imageResizeMode}' for a detector. "
        "See preprocessing.json inputs[0].imageResize and adapt this client.",
      );
    }
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) throw const FormatException('Could not decode image bytes');
    final input = letterboxImage(decoded, height: size[0], width: size[1], pad: _inputSpec.padValue);

    final inputTensor = OrtValueTensor.createTensorWithDataList(input.chw, [1, 3, input.height, input.width]);
    final runOptions = OrtRunOptions();
    final outputs = await _session.runAsync(runOptions, {_onnxInputName: inputTensor}, [_onnxOutputName]);
    final raw = outputs?.first?.value;
    inputTensor.release();
    runOptions.release();
    outputs?.first?.release();

    // [1, 4 + classes, candidates] -> the [4 + classes][candidates] matrix of the one image.
    final table = [
      for (final row in (raw as List).first as List) [for (final v in row as List) (v as num).toDouble()],
    ];
    return decodeDetections(
      table,
      input,
      _outputSpec.classes ?? const [],
      confidence: confidence ?? _outputSpec.confidenceThreshold,
      iou: iou ?? _outputSpec.iouThreshold,
      maxDetections: maxDetections,
    );
  }

  /// Runs inference on raw image bytes (the bytes of a WAV file for an audio classifier) and returns
  /// `{className: probability}` sorted descending (classification), or `{"value": x}` (regression).
  Future<Map<String, double>> predict(Uint8List imageBytes, {int topK = 5}) async {
    if (isDetector) {
      throw UnsupportedError('This model is an object detector: call detect(), which returns boxes.');
    }
    if (isAudio) return _classifyAudio(imageBytes, topK);
    if (isText) throw UnsupportedError('This model takes text: call predictText(text).');
    if (isTagger) throw UnsupportedError('This model tags every word of a text: call tag(text), which returns a tag for each.');
    if (_inputSpec.type != 'image') {
      throw UnsupportedError(
        "No preprocessing implemented for input type '${_inputSpec.type}'. "
        'See preprocessing.json inputs[0].imageSize and adapt this client.',
      );
    }

    final (width, height) = _imageDims;
    final chw = _preprocessImage(imageBytes);
    final inputTensor = OrtValueTensor.createTensorWithDataList(chw, [1, 3, height, width]);
    final runOptions = OrtRunOptions();

    final outputs = await _session.runAsync(runOptions, {_onnxInputName: inputTensor}, [_onnxOutputName]);
    final raw = outputs?.first?.value;
    inputTensor.release();
    runOptions.release();
    outputs?.first?.release();

    final logits = _flatten(raw);
    final classes = _outputSpec.classes;
    if (classes != null && classes.length == logits.length) {
      final probs = _softmax(logits);
      final ranked = List.generate(classes.length, (i) => MapEntry(classes[i], probs[i]))
        ..sort((a, b) => b.value.compareTo(a.value));
      return {for (final e in ranked.take(topK)) e.key: (e.value * 10000).round() / 10000};
    }
    return {'value': logits.isNotEmpty ? logits[0] : 0.0};
  }

  void dispose() {
    _session.release();
  }
}
