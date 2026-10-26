// OmniTrain-generated inference client.
//
// Wraps the exported ONNX model with the preprocessing/postprocessing
// Ludwig applied at training time, read from preprocessing.json — Ludwig's
// exported ONNX graph is the bare model only (no resize/normalize, no
// class-index decoding), so this client reconstructs that step from the
// manifest rather than assuming a fixed tensor layout: input/output tensor
// names are matched against the graph's actual names at runtime, not
// hardcoded, since ONNX export doesn't guarantee a stable naming convention
// across Ludwig versions.
//
// Implements preprocessing for `image` inputs with `category` outputs (image
// classification) and for a `detections` output (object detection, YOLO): the
// ONNX graph returns every candidate box, so for a detector this client does
// the confidence filter, per-class non-maximum suppression and the undoing of
// the letterbox, and `detect()` returns the boxes. Other input types throw with
// a pointer to preprocessing.json so you can adapt preprocessing yourself; see
// README.md.
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

class PreprocessingInput {
  PreprocessingInput({
    required this.type,
    required this.column,
    this.ludwigPreprocessing,
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
      ludwigPreprocessing: json['ludwigPreprocessing'] as Map<String, dynamic>?,
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
  final Map<String, dynamic>? ludwigPreprocessing;
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
  }

  (int width, int height) get _imageDims {
    final pp = _inputSpec.ludwigPreprocessing ?? const <String, dynamic>{};
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

  /// Runs inference on raw image bytes and returns `{className: probability}`
  /// sorted descending (classification), or `{"value": x}` (regression).
  Future<Map<String, double>> predict(Uint8List imageBytes, {int topK = 5}) async {
    if (isDetector) {
      throw UnsupportedError('This model is an object detector: call detect(), which returns boxes.');
    }
    if (_inputSpec.type != 'image') {
      throw UnsupportedError(
        "No preprocessing implemented for input type '${_inputSpec.type}'. "
        'See preprocessing.json inputs[0].ludwigPreprocessing and adapt this client.',
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
