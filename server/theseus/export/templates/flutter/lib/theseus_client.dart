// Theseus-generated inference client.
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
// Currently implements preprocessing for `image` inputs with `category`
// outputs (image classification) — the modality this export path has been
// verified against. Other input types throw with a pointer to
// preprocessing.json so you can adapt preprocessing yourself; see README.md.
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
    this.normMean,
    this.normStd,
  });

  factory PreprocessingInput.fromJson(Map<String, dynamic> json) {
    final norm = json['imageNormalization'] as Map<String, dynamic>?;
    return PreprocessingInput(
      type: json['type'] as String,
      column: json['column'] as String,
      ludwigPreprocessing: json['ludwigPreprocessing'] as Map<String, dynamic>?,
      normMean: (norm?['mean'] as List?)?.map((e) => (e as num).toDouble()).toList(),
      normStd: (norm?['std'] as List?)?.map((e) => (e as num).toDouble()).toList(),
    );
  }

  final String type;
  final String column;
  final Map<String, dynamic>? ludwigPreprocessing;
  final List<double>? normMean;
  final List<double>? normStd;
}

class PreprocessingOutput {
  PreprocessingOutput({required this.column, this.classes});

  factory PreprocessingOutput.fromJson(Map<String, dynamic> json) {
    return PreprocessingOutput(
      column: json['column'] as String,
      classes: (json['classes'] as List?)?.map((e) => e as String).toList(),
    );
  }

  final String column;
  final List<String>? classes;
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

class TheseusClient {
  late OrtSession _session;
  late PreprocessingInput _inputSpec;
  late PreprocessingOutput _outputSpec;
  late String _onnxInputName;
  late String _onnxOutputName;

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

  Float32List _preprocessImage(Uint8List bytes) {
    final decoded = img.decodeImage(bytes);
    if (decoded == null) throw const FormatException('Could not decode image bytes');
    final (width, height) = _imageDims;
    final resized = img.copyResize(decoded, width: width, height: height);

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

  /// Runs inference on raw image bytes and returns `{className: probability}`
  /// sorted descending (classification), or `{"value": x}` (regression).
  Future<Map<String, double>> predict(Uint8List imageBytes, {int topK = 5}) async {
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
