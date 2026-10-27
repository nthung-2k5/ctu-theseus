/**
 * OmniTrain-generated inference client (browser). Wraps the exported ONNX
 * model with the preprocessing/postprocessing the platform applied at training
 * time, read from preprocessing.json — see the devkit export's client for
 * the same logic in Python/TypeScript/C#/Java.
 *
 * Implements preprocessing for `image` inputs with `category` outputs
 * (image classification) and for a `detections` output (object detection,
 * YOLO): the ONNX graph returns every candidate box, so for a detector this
 * app does the confidence filter, per-class non-maximum suppression and the
 * undoing of the letterbox, and `predict()` returns the boxes — see README.md.
 *
 * An `audio` input that carries an `audioSpectrogram` recipe (audio classification)
 * is supported too: `predict()` takes an audio File/Blob, makes the log-mel spectrograms
 * the recipe describes (a WAV file is read here; any other format the browser can decode
 * is decoded by the browser at the model's sample rate, which can move a score slightly),
 * and averages the windows' probabilities.
 *
 * A `text` input that carries a `textTokens` recipe (text classification) is supported as
 * well: `predict()` takes the text itself (a string), splits it into words and looks them
 * up in the vocabulary that ships in preprocessing.json; the page shows a text box.
 *
 * A `text` input that carries a `tagWords` recipe and a `tags` output (token classification) is
 * supported too: `predict()` takes the text and returns `[{token, tag, confidence}]`, one per word;
 * the page shows a text box and a line per word.
 */

const HERE = new URL('.', import.meta.url)

let session = null
let inputSpec = null
let outputSpec = null
let onnxInputName = null
let onnxOutputName = null

function matchTensorName(candidates, preferred) {
  if (candidates.length === 1) return candidates[0]
  const match = candidates.find(
    (n) => n === preferred || n.startsWith(`${preferred}::`) || n.startsWith(`${preferred}_`),
  )
  if (!match) {
    throw new Error(
      `Could not match an ONNX tensor to feature '${preferred}' among [${candidates.join(', ')}]. ` +
        'Inspect preprocessing.json and adapt app.js if this model uses an unrecognized naming scheme.',
    )
  }
  return match
}

/**
 * Indices of the boxes to keep, most confident first: within each class, a box is dropped when it overlaps a more
 * confident one by more than `iouThreshold` (intersection over union). Boxes are `[x1, y1, x2, y2]`.
 */
function nonMaxSuppression(boxes, scores, classes, iouThreshold) {
  const kept = []
  for (const label of new Set(classes)) {
    let order = classes
      .map((c, i) => (c === label ? i : -1))
      .filter((i) => i >= 0)
      .sort((a, b) => scores[b] - scores[a] || a - b)
    while (order.length > 0) {
      const [first, ...rest] = order
      kept.push(first)
      const a = boxes[first]
      const areaFirst = (a[2] - a[0]) * (a[3] - a[1])
      order = rest.filter((i) => {
        const b = boxes[i]
        const width = Math.min(a[2], b[2]) - Math.max(a[0], b[0])
        const height = Math.min(a[3], b[3]) - Math.max(a[1], b[1])
        const intersection = Math.max(width, 0) * Math.max(height, 0)
        const union = areaFirst + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        return intersection / Math.max(union, 1e-9) <= iouThreshold
      })
    }
  }
  return kept.sort((a, b) => scores[b] - scores[a])
}

// ---- Audio ----
// `inputs[0].audioSpectrogram` spelled out. The platform computes the same thing in numpy; the arithmetic is plain
// enough (a polyphase resampler, a radix-2 FFT, a mel filterbank) to follow in any language.

/** A WAV file's samples, one Float32Array per channel, in [-1, 1], and its sample rate. */
function readWav(buffer) {
  const file = new Uint8Array(buffer)
  const view = new DataView(buffer)
  const tag = (at) => String.fromCharCode(file[at], file[at + 1], file[at + 2], file[at + 3])
  if (file.length < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('Not a WAV file')
  let format = null
  let dataAt = -1
  let dataSize = 0
  for (let at = 12; at + 8 <= file.length; ) {
    const size = view.getUint32(at + 4, true)
    const body = at + 8
    if (tag(at) === 'fmt ' && size >= 16) {
      let code = view.getUint16(body, true)
      if (code === 0xfffe && size >= 26) code = view.getUint16(body + 24, true) // extensible: the sub-format's tag
      format = { code, channels: view.getUint16(body + 2, true), rate: view.getUint32(body + 4, true), bits: view.getUint16(body + 14, true) }
    } else if (tag(at) === 'data') {
      dataAt = body
      dataSize = Math.min(size, file.length - body) // a streamed file may claim more than it holds
      break
    }
    at = body + size + (size & 1)
  }
  if (!format || dataAt < 0 || dataSize === 0) throw new Error('The WAV file has no format or no samples')
  const { code, channels, rate, bits } = format
  const width = bits >> 3
  if (channels < 1 || width < 1) throw new Error('The WAV file has no channels')
  const frames = Math.floor(dataSize / (width * channels))
  const read = (at) => {
    if (code === 1 && bits === 8) return (view.getUint8(at) - 128) / 128
    if (code === 1 && bits === 16) return view.getInt16(at, true) / 32767
    if (code === 1 && bits === 24) {
      const v = view.getUint8(at) | (view.getUint8(at + 1) << 8) | (view.getUint8(at + 2) << 16)
      return (v >= 0x800000 ? v - 0x1000000 : v) / 8388607
    }
    if (code === 1 && bits === 32) return view.getInt32(at, true) / 2147483647
    if (code === 3 && bits === 32) return view.getFloat32(at, true)
    if (code === 3 && bits === 64) return view.getFloat64(at, true)
    throw new Error(`Unsupported WAV encoding (format ${code}, ${bits} bits)`)
  }
  const out = Array.from({ length: channels }, () => new Float32Array(frames))
  for (let f = 0; f < frames; f++) {
    for (let c = 0; c < channels; c++) out[c][f] = read(dataAt + (f * channels + c) * width)
  }
  return { channels: out, rate }
}

/** Any other format: the browser decodes it, resampled to `sampleRate` (the browser's resampler, not the recipe's). */
async function decodeWithBrowser(buffer, sampleRate) {
  const context = new AudioContext({ sampleRate })
  try {
    const decoded = await context.decodeAudioData(buffer.slice(0))
    return { channels: Array.from({ length: decoded.numberOfChannels }, (_, c) => decoded.getChannelData(c)), rate: decoded.sampleRate }
  } finally {
    context.close()
  }
}

function gcd(a, b) {
  return b === 0 ? a : gcd(b, a % b)
}

/** Modified Bessel function of the first kind, order 0 (the Kaiser window's building block). */
function besselI0(x) {
  let sum = 1
  let term = 1
  for (let k = 1; k < 500; k++) {
    term *= (x / (2 * k)) ** 2
    sum += term
    if (term < sum * 1e-17) break
  }
  return sum
}

/**
 * `scipy.signal.resample_poly(x, up, down)`: out[i] = sum over k in -H..H of h[H + k] * x[(i * down + k) / up],
 * counting the k for which i * down + k is a multiple of `up` and the index is inside the recording. `h` is a
 * Kaiser-windowed (beta 5) sinc low-pass of 2H + 1 taps, H = 10 * max(up, down), scaled to a gain of `up`.
 */
function resample(x, rate, target) {
  const divisor = gcd(rate, target)
  const up = target / divisor
  const down = rate / divisor
  const big = Math.max(up, down)
  const half = 10 * big
  const sinc = (v) => (v === 0 ? 1 : Math.sin(Math.PI * v) / (Math.PI * v))
  const taps = new Float64Array(2 * half + 1)
  const i0Beta = besselI0(5)
  let total = 0
  for (let n = 0; n <= 2 * half; n++) {
    const ratio = (n - half) / half
    const window = besselI0(5 * Math.sqrt(Math.max(1 - ratio * ratio, 0))) / i0Beta
    taps[n] = (sinc((n - half) / big) / big) * window
    total += taps[n]
  }
  for (let n = 0; n <= 2 * half; n++) taps[n] *= up / total

  const outLength = Math.floor((x.length * up + down - 1) / down)
  const out = new Float32Array(outLength)
  for (let i = 0; i < outLength; i++) {
    const position = i * down
    const anchor = Math.floor(position / up)
    const phase = position % up
    const first = -Math.floor((half - phase) / up)
    const last = Math.floor((half + phase) / up)
    let acc = 0
    for (let d = first; d <= last; d++) {
      const index = anchor + d
      if (index >= 0 && index < x.length) acc += taps[half - phase + d * up] * x[index]
    }
    out[i] = acc
  }
  return out
}

/** Where the windows of a `total`-sample recording start (`audioSpectrogram.windowStarts`). */
function windowStarts(total, length, count) {
  if (total <= length) return [0]
  const room = total - length
  if (count <= 1) return [Math.floor(room / 2)]
  const starts = new Set()
  for (let i = 0; i < count; i++) starts.add(roundHalfEven((i * room) / (count - 1)))
  return [...starts].sort((a, b) => a - b)
}

/** In-place radix-2 FFT of `re` + i `im` (length a power of two). */
function fft(re, im) {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const angle = (-2 * Math.PI) / size
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < size / 2; k++) {
        const wr = Math.cos(angle * k)
        const wi = Math.sin(angle * k)
        const a = start + k
        const b = a + size / 2
        const tr = re[b] * wr - im[b] * wi
        const ti = re[b] * wi + im[b] * wr
        re[b] = re[a] - tr
        im[b] = im[a] - ti
        re[a] += tr
        im[a] += ti
      }
    }
  }
}

/** `[nFft / 2 + 1][bands]` triangular HTK mel filters, no area normalisation. */
function melFilterbank(sampleRate, nFft, bands, fMin, fMax) {
  const toMel = (hz) => 2595 * Math.log10(1 + hz / 700)
  const edges = Array.from({ length: bands + 2 }, (_, i) => {
    const mel = toMel(fMin) + ((toMel(fMax) - toMel(fMin)) * i) / (bands + 1)
    return 700 * (10 ** (mel / 2595) - 1)
  })
  const bins = nFft / 2 + 1
  const filters = Array.from({ length: bins }, () => new Float64Array(bands))
  for (let bin = 0; bin < bins; bin++) {
    const hz = (sampleRate / 2) * (bin / (bins - 1))
    for (let b = 0; b < bands; b++) {
      const rising = (hz - edges[b]) / (edges[b + 1] - edges[b])
      const falling = (edges[b + 2] - hz) / (edges[b + 2] - edges[b + 1])
      filters[bin][b] = Math.max(0, Math.min(rising, falling))
    }
  }
  return filters
}

/** The standardised log-mel spectrogram of one window, `[bands][frames]` flattened row by row. */
function spectrogramOf(window, recipe, filters) {
  const { nFft, hopLength } = recipe.stft
  const bands = recipe.mel.bands
  const frames = recipe.frames
  const hann = Float64Array.from({ length: nFft }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / nFft))
  const out = new Float32Array(bands * frames)
  const re = new Float64Array(nFft)
  const im = new Float64Array(nFft)
  const bins = nFft / 2 + 1
  const power = new Float64Array(bins)
  for (let f = 0; f < frames; f++) {
    for (let n = 0; n < nFft; n++) {
      re[n] = window[f * hopLength + n] * hann[n]
      im[n] = 0
    }
    fft(re, im)
    for (let k = 0; k < bins; k++) power[k] = re[k] * re[k] + im[k] * im[k]
    for (let b = 0; b < bands; b++) {
      let mel = 0
      for (let k = 0; k < bins; k++) mel += power[k] * filters[k][b]
      const logMel = Math.fround(Math.log(mel + recipe.logFloor))
      out[b * frames + f] = (logMel - recipe.normalization.mean) / recipe.normalization.std
    }
  }
  return out
}

/** The recording (a File/Blob) as a batch of spectrograms, `[windows, 1, bands, frames]`, by `audioSpectrogram`. */
async function audioSpectrograms(source) {
  const recipe = inputSpec.audioSpectrogram
  const buffer = await source.arrayBuffer()
  let decoded
  try {
    decoded = readWav(buffer)
  } catch (wavError) {
    try {
      decoded = await decodeWithBrowser(buffer, recipe.sampleRate)
    } catch {
      throw wavError // neither a WAV file nor something the browser can decode: say what is wrong with the WAV
    }
  }
  const { channels, rate } = decoded
  const frames = channels[0].length
  // The channels' mean, at double precision as the platform does it.
  const mono = new Float64Array(frames)
  for (const channel of channels) for (let i = 0; i < frames; i++) mono[i] += channel[i]
  for (let i = 0; i < frames; i++) mono[i] /= channels.length
  const wave = resample(mono, rate, recipe.sampleRate)
  if (wave.length === 0) throw new Error('The recording has no samples')

  const length = recipe.windowSamples
  const filters = melFilterbank(recipe.sampleRate, recipe.stft.nFft, recipe.mel.bands, recipe.mel.fMin, recipe.mel.fMax)
  const starts = windowStarts(wave.length, length, recipe.windowsPerAudio)
  const size = recipe.mel.bands * recipe.frames
  const batch = new Float32Array(starts.length * size)
  starts.forEach((start, w) => {
    const window = new Float32Array(length) // zero-padded at the end when the recording runs out
    window.set(wave.subarray(start, Math.min(start + length, wave.length)))
    batch.set(spectrogramOf(window, recipe, filters), w * size)
  })
  return new ort.Tensor('float32', batch, [starts.length, 1, recipe.mel.bands, recipe.frames])
}

/** Class probabilities of a recording: each window's logits through a softmax, averaged over the windows. */
async function classifyAudio(source, topK) {
  const input = await audioSpectrograms(source)
  const outputs = await session.run({ [onnxInputName]: input })
  const logits = outputs[onnxOutputName].data
  const windows = input.dims[0]
  const classes = outputSpec.classes ?? []
  const mean = new Float64Array(classes.length)
  for (let w = 0; w < windows; w++) {
    const probs = softmax(Array.from(logits.subarray(w * classes.length, (w + 1) * classes.length)))
    for (let c = 0; c < classes.length; c++) mean[c] += probs[c] / windows
  }
  return Object.fromEntries(
    classes
      .map((name, i) => [name, mean[i]])
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([name, p]) => [name, Math.round(p * 10000) / 10000]),
  )
}

function softmax(logits) {
  const max = Math.max(...logits)
  const exp = logits.map((v) => Math.exp(v - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map((v) => v / sum)
}

/** An audio classifier: `predict()` takes an audio File/Blob. */
function isAudio() {
  return inputSpec?.type === 'audio' && inputSpec.audioSpectrogram !== undefined
}

/** A text classifier: `predict()` takes the text itself. */
function isText() {
  return inputSpec?.type === 'text' && inputSpec.textTokens !== undefined
}

let textSeparators = null
let textIds = null

/** A token classifier: `predict()` takes the text itself and returns a tag for every word. */
function isTagger() {
  return inputSpec?.type === 'text' && inputSpec.tagWords !== undefined && outputSpec?.type === 'tags'
}

let tagWhitespace = null
let tagIds = null

/** The words of a text: runs of code units that are not whitespace (the ranges are the ones `str.split()` cuts at). */
function splitWords(text) {
  const words = []
  let start = -1
  for (let i = 0; i < text.length; i++) {
    if (tagWhitespace[text.charCodeAt(i)] === 1) {
      if (start >= 0) words.push(text.slice(start, i))
      start = -1
    } else if (start < 0) {
      start = i
    }
  }
  if (start >= 0) words.push(text.slice(start))
  return words
}

/**
 * One `{token, tag, confidence}` per word of the text: the highest of the word's row of logits and its softmax
 * probability. The first maxTokens words are read; a later word was not, and gets the recipe's fallbackTag with
 * confidence 0.
 */
async function tagText(source) {
  const text = typeof source === 'string' ? source : await source.text()
  const recipe = inputSpec.tagWords
  const words = splitWords(text)
  if (words.length === 0) return []
  const { from, to, add } = recipe.lowercase
  const ids = new BigInt64Array(recipe.maxTokens).fill(BigInt(recipe.padId))
  words.slice(0, recipe.maxTokens).forEach((word, i) => {
    let form = ''
    for (let k = 0; k < word.length; k++) {
      const unit = word.charCodeAt(k)
      form += String.fromCharCode(unit >= from && unit <= to ? unit + add : unit)
    }
    ids[i] = BigInt(tagIds.get(form) ?? recipe.unknownId)
  })
  const outputs = await session.run({ [onnxInputName]: new ort.Tensor('int64', ids, [1, recipe.maxTokens]) })
  const logits = Array.from(outputs[onnxOutputName].data)
  const classes = outputSpec.classes ?? []
  return words.map((token, i) => {
    if (i >= recipe.maxTokens) return { token, tag: recipe.fallbackTag, confidence: 0 }
    const probs = softmax(logits.slice(i * classes.length, (i + 1) * classes.length))
    let best = 0
    probs.forEach((p, k) => {
      if (p > probs[best]) best = k
    })
    return { token, tag: classes[best], confidence: Math.round(probs[best] * 10000) / 10000 }
  })
}

/** The tokens of a text: the ASCII letters A-Z lower-cased, cut at every run of separator code units. */
function tokenize(text, recipe) {
  const { from, to, add } = recipe.lowercase
  const tokens = []
  let current = ''
  for (let i = 0; i < text.length; i++) {
    const unit = text.charCodeAt(i)
    if (textSeparators[unit] === 1) {
      if (current !== '') tokens.push(current)
      current = ''
    } else {
      current += String.fromCharCode(unit >= from && unit <= to ? unit + add : unit)
    }
  }
  if (current !== '') tokens.push(current)
  return tokens
}

/** `[1, maxTokens]` int64 word ids: the first maxTokens tokens looked up in the vocabulary, padded at the end. */
function textTokens(text) {
  const recipe = inputSpec.textTokens
  const ids = new BigInt64Array(recipe.maxTokens).fill(BigInt(recipe.padId))
  tokenize(text, recipe)
    .slice(0, recipe.maxTokens)
    .forEach((token, i) => {
      ids[i] = BigInt(textIds.get(token) ?? recipe.unknownId)
    })
  return new ort.Tensor('int64', ids, [1, recipe.maxTokens])
}

/** Class probabilities of a text: the logits of its one row through a softmax. */
async function classifyText(source, topK) {
  const text = typeof source === 'string' ? source : await source.text()
  const outputs = await session.run({ [onnxInputName]: textTokens(text) })
  const probs = softmax(Array.from(outputs[onnxOutputName].data))
  const classes = outputSpec.classes ?? []
  return Object.fromEntries(
    classes
      .map((name, i) => [name, probs[i]])
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([name, p]) => [name, Math.round(p * 10000) / 10000]),
  )
}

let loading = null
/** Loads the manifest and the model once, however many callers ask. */
function init() {
  loading ??= load()
  return loading
}

async function load() {
  const res = await fetch(new URL('preprocessing.json', HERE))
  const manifest = await res.json()
  inputSpec = manifest.inputs[0]
  outputSpec = manifest.outputs[0]
  if (isAudio() && inputSpec.audioSpectrogram.recipe !== 'log_mel_v1') {
    throw new Error(
      `Unknown audio recipe '${inputSpec.audioSpectrogram.recipe}'. ` +
        "See preprocessing.json's inputs[0].audioSpectrogram and adapt app.js.",
    )
  }
  if (isText()) {
    const recipe = inputSpec.textTokens
    if (recipe.recipe !== 'word_ids_v1') {
      throw new Error(
        `Unknown text recipe '${recipe.recipe}'. See preprocessing.json's inputs[0].textTokens and adapt app.js.`,
      )
    }
    textSeparators = new Uint8Array(0x10000)
    for (const [lo, hi] of recipe.separatorRanges) textSeparators.fill(1, lo, Math.min(hi, 0xffff) + 1)
    textIds = new Map(recipe.vocabulary.map((word, i) => [word, recipe.firstWordId + i]))
  }
  if (isTagger()) {
    const recipe = inputSpec.tagWords
    if (recipe.recipe !== 'tag_word_ids_v1') {
      throw new Error(
        `Unknown tagging recipe '${recipe.recipe}'. See preprocessing.json's inputs[0].tagWords and adapt app.js.`,
      )
    }
    tagWhitespace = new Uint8Array(0x10000)
    for (const [lo, hi] of recipe.whitespaceRanges) tagWhitespace.fill(1, lo, Math.min(hi, 0xffff) + 1)
    tagIds = new Map(recipe.vocabulary.map((word, i) => [word, recipe.firstWordId + i]))
  }

  session = await ort.InferenceSession.create(new URL('model.onnx', HERE).toString())
  onnxInputName = matchTensorName(session.inputNames, inputSpec.column)
  onnxOutputName = matchTensorName(session.outputNames, outputSpec.column)
}

/** Rounds .5 to the nearest even integer, as Python's round() does (and torchvision's CenterCrop relies on). */
function roundHalfEven(x) {
  const floor = Math.floor(x)
  const diff = x - floor
  if (diff < 0.5) return floor
  if (diff > 0.5) return floor + 1
  return floor % 2 === 0 ? floor : floor + 1
}

/**
 * For `imageResize.mode === 'shorter_edge_center_crop'`: scale so the SHORTER edge is `size` (the other edge is
 * truncated to a whole pixel), then centre-crop a size x size square. Same arithmetic as torchvision's
 * Resize(size) followed by CenterCrop(size), which is what the model was evaluated with.
 */
function shorterEdgeGeometry(width, height, size) {
  const [newWidth, newHeight] =
    width <= height ? [size, Math.floor((size * height) / width)] : [Math.floor((size * width) / height), size]
  return {
    width: newWidth,
    height: newHeight,
    left: roundHalfEven((newWidth - size) / 2),
    top: roundHalfEven((newHeight - size) / 2),
  }
}

/** Decode an ImageBitmap into a CHW float32 tensor. */
function preprocessImage(imageBitmap) {
  const pp = inputSpec.imageSize ?? {}
  const width = pp.width ?? 224
  const height = pp.height ?? 224

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  const mode = inputSpec.imageResize?.mode
  if (mode === undefined) {
    ctx.drawImage(imageBitmap, 0, 0, width, height) // the whole image squashed to the target size
  } else if (mode === 'shorter_edge_center_crop') {
    const g = shorterEdgeGeometry(imageBitmap.width, imageBitmap.height, Math.min(width, height))
    // Draw the part of the SOURCE that the centre crop of the resized image covers, so it is resampled once.
    const scaleX = imageBitmap.width / g.width
    const scaleY = imageBitmap.height / g.height
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(imageBitmap, g.left * scaleX, g.top * scaleY, width * scaleX, height * scaleY, 0, 0, width, height)
  } else {
    throw new Error(
      `Unsupported imageResize mode '${mode}'. See preprocessing.json inputs[0].imageResize and adapt app.js.`,
    )
  }
  const { data } = ctx.getImageData(0, 0, width, height) // RGBA, HWC, uint8

  const norm = inputSpec.imageNormalization
  const chw = new Float32Array(3 * height * width)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4
      for (let c = 0; c < 3; c++) {
        let v = data[i + c] / 255
        if (norm) v = (v - norm.mean[c]) / norm.std[c]
        chw[c * height * width + y * width + x] = v
      }
    }
  }
  return { chw, width, height }
}

/**
 * The image as the detector's input tensor, and what is needed to map its boxes back. For
 * `imageResize.mode === 'letterbox'`: scale so the LONGER edge is `size`, pad the shorter one, centred, with
 * `padValue` grey up to a size x size square (the arithmetic of ultralytics.data.augment.LetterBox). RGB, 0..1,
 * no mean/std.
 */
function letterbox(imageBitmap) {
  const resize = inputSpec.imageResize
  if (resize?.mode !== 'letterbox' || !resize.size) {
    throw new Error(
      `Unsupported imageResize mode '${resize?.mode}' for a detector. See preprocessing.json inputs[0].imageResize and adapt app.js.`,
    )
  }
  const [height, width] = resize.size
  const pad = resize.padValue ?? 114
  const originalWidth = imageBitmap.width
  const originalHeight = imageBitmap.height
  const scale = Math.min(height / originalHeight, width / originalWidth)
  const newWidth = Math.round(originalWidth * scale)
  const newHeight = Math.round(originalHeight * scale)
  const left = Math.round((width - newWidth) / 2 - 0.1)
  const top = Math.round((height - newHeight) / 2 - 0.1)

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  ctx.fillStyle = `rgb(${pad}, ${pad}, ${pad})`
  ctx.fillRect(0, 0, width, height)
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(imageBitmap, left, top, newWidth, newHeight)
  const { data } = ctx.getImageData(0, 0, width, height) // RGBA, HWC, uint8

  const chw = new Float32Array(3 * height * width)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4
      for (let c = 0; c < 3; c++) chw[c * height * width + y * width + x] = data[i + c] / 255
    }
  }
  return { tensor: new ort.Tensor('float32', chw, [1, 3, height, width]), scale, left, top, originalWidth, originalHeight }
}

/**
 * Boxes found in an image File/Blob, most confident first: `[{label, confidence, box: {x, y, w, h}}]`, `box` being
 * the top-left corner and the size as fractions (0..1) of the image. `confidence` and `iou` default to the
 * thresholds in preprocessing.json (`outputs[0].yoloOutput`).
 */
export async function detect(imageSource, { confidence, iou, maxDetections = 300 } = {}) {
  await init()
  const info = outputSpec.yoloOutput ?? {}
  const minConfidence = confidence ?? info.confidenceThreshold ?? 0.25
  const maxOverlap = iou ?? info.iouThreshold ?? 0.7
  const classes = outputSpec.classes ?? []

  const imageBitmap = await createImageBitmap(imageSource)
  const { tensor, scale, left, top, originalWidth, originalHeight } = letterbox(imageBitmap)
  const outputs = await session.run({ [onnxInputName]: tensor })
  const raw = outputs[onnxOutputName]
  // [1, 4 + classes, candidates]: row r of the table is data[r * candidates .. (r + 1) * candidates].
  const candidates = raw.dims[2]
  const data = raw.data

  const boxes = []
  const scores = []
  const labels = []
  for (let i = 0; i < candidates; i++) {
    let best = 0
    let bestScore = -Infinity
    for (let c = 0; c < classes.length; c++) {
      const score = data[(4 + c) * candidates + i]
      if (score > bestScore) {
        bestScore = score
        best = c
      }
    }
    if (bestScore < minConfidence) continue
    const [cx, cy, w, h] = [0, 1, 2, 3].map((r) => data[r * candidates + i])
    boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    scores.push(bestScore)
    labels.push(best)
  }

  const clamp = (v, max) => Math.min(Math.max(v, 0), max)
  const round = (v, places) => Math.round(v * 10 ** places) / 10 ** places
  return nonMaxSuppression(boxes, scores, labels, maxOverlap)
    .slice(0, maxDetections)
    .map((index) => {
      // Undo the letterbox: remove the padding, divide by the scale, clip to the image.
      const [bx1, by1, bx2, by2] = boxes[index]
      const x1 = clamp((Math.max(bx1, 0) - left) / scale, originalWidth)
      const x2 = clamp((Math.max(bx2, 0) - left) / scale, originalWidth)
      const y1 = clamp((Math.max(by1, 0) - top) / scale, originalHeight)
      const y2 = clamp((Math.max(by2, 0) - top) / scale, originalHeight)
      return {
        label: classes[labels[index]],
        confidence: round(scores[index], 4),
        box: {
          x: round(x1 / originalWidth, 6),
          y: round(y1 / originalHeight, 6),
          w: round((x2 - x1) / originalWidth, 6),
          h: round((y2 - y1) / originalHeight, 6),
        },
      }
    })
}

/**
 * Run inference on an image File/Blob (an audio File/Blob for an audio classifier, the text itself for a text
 * classifier) and return
 * {className: probability} (classification), {value} (regression), or the list of boxes found (an object detector;
 * see `detect`).
 */
export async function predict(imageSource, topK = 5) {
  await init()
  if (outputSpec.type === 'detections') return detect(imageSource)
  if (isAudio()) return classifyAudio(imageSource, topK)
  if (isText()) return classifyText(imageSource, topK)
  if (isTagger()) return tagText(imageSource)

  if (inputSpec.type !== 'image') {
    throw new Error(
      `No preprocessing implemented for input type '${inputSpec.type}'. ` +
        'See preprocessing.json inputs[0].imageSize and adapt app.js.',
    )
  }

  const imageBitmap = await createImageBitmap(imageSource)
  const { chw, width, height } = preprocessImage(imageBitmap)
  const tensor = new ort.Tensor('float32', chw, [1, 3, height, width])
  const outputs = await session.run({ [onnxInputName]: tensor })
  const logits = outputs[onnxOutputName].data

  const classes = outputSpec.classes
  if (classes && classes.length === logits.length) {
    const probs = softmax(Array.from(logits))
    const ranked = classes
      .map((name, i) => [name, probs[i]])
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
    return Object.fromEntries(ranked.map(([name, p]) => [name, Math.round(p * 10000) / 10000]))
  }
  return { value: logits[0] }
}

// ---- UI wiring ----

const dropzone = document.getElementById('dropzone')
const fileInput = document.getElementById('file-input')
const preview = document.getElementById('preview')
const overlay = document.getElementById('overlay')
const resultsSection = document.getElementById('results')
const predictionsList = document.getElementById('predictions')
const statusEl = document.getElementById('status')
const audioPreview = document.getElementById('audio-preview')
const textSection = document.getElementById('text-input')
const textBox = document.getElementById('text-box')
const classifyBtn = document.getElementById('classify-btn')

// An audio model asks for audio files and a text model for a text; the rest of the page is the same.
init()
  .then(() => {
    if (isText() || isTagger()) {
      dropzone.hidden = true
      textSection.hidden = false
      if (isTagger()) {
        document.querySelector('label[for="text-box"]').textContent = 'Text to tag'
        classifyBtn.textContent = 'Tag'
      }
      return
    }
    if (!isAudio()) return
    fileInput.accept = 'audio/*,.wav'
    document.getElementById('dropzone-label').textContent = 'Choose an audio file, or drag one here'
  })
  .catch(() => {})

/** Draws a detector's boxes over the preview image; the canvas has the image's own size, CSS scales both alike. */
function drawBoxes(detections) {
  const width = preview.naturalWidth
  const height = preview.naturalHeight
  overlay.width = width
  overlay.height = height
  const ctx = overlay.getContext('2d')
  ctx.clearRect(0, 0, width, height)
  ctx.lineWidth = Math.max(2, Math.round(Math.max(width, height) / 250))
  ctx.font = `${Math.max(12, Math.round(Math.max(width, height) / 40))}px system-ui, sans-serif`
  ctx.textBaseline = 'top'
  for (const { label, confidence, box } of detections) {
    const [x, y, w, h] = [box.x * width, box.y * height, box.w * width, box.h * height]
    ctx.strokeStyle = '#1864ab'
    ctx.strokeRect(x, y, w, h)
    const text = `${label} ${confidence.toFixed(2)}`
    const textWidth = ctx.measureText(text).width
    ctx.fillStyle = '#1864ab'
    ctx.fillRect(x, y, textWidth + 8, parseInt(ctx.font, 10) + 6)
    ctx.fillStyle = '#fff'
    ctx.fillText(text, x + 4, y + 3)
  }
  overlay.hidden = false
}

async function handleFile(file) {
  if (!file) return
  statusEl.textContent = 'Running inference…'
  overlay.hidden = true
  if (isText() || isTagger()) {
    // Nothing to preview: the text is in the box.
  } else if (isAudio()) {
    audioPreview.src = URL.createObjectURL(file)
    audioPreview.hidden = false
  } else {
    preview.src = URL.createObjectURL(file)
    preview.hidden = false
  }

  try {
    const predictions = await predict(file)
    predictionsList.innerHTML = ''
    if (isTagger()) {
      // A token classifier: one line per word, the word, its tag and the model's confidence.
      for (const { token, tag, confidence } of predictions) {
        const li = document.createElement('li')
        li.textContent = `${token}: ${tag} (${confidence.toFixed(4)})`
        predictionsList.appendChild(li)
      }
      if (predictions.length === 0) predictionsList.innerHTML = '<li>No words found.</li>'
    } else if (Array.isArray(predictions)) {
      // An object detector: one line per box, and the boxes drawn on the picture.
      await preview.decode()
      drawBoxes(predictions)
      for (const { label, confidence, box } of predictions) {
        const li = document.createElement('li')
        li.textContent = `${label}: ${confidence.toFixed(4)}  x=${box.x.toFixed(3)} y=${box.y.toFixed(3)} w=${box.w.toFixed(3)} h=${box.h.toFixed(3)}`
        predictionsList.appendChild(li)
      }
      if (predictions.length === 0) predictionsList.innerHTML = '<li>No objects found.</li>'
    } else {
      for (const [className, probability] of Object.entries(predictions)) {
        const li = document.createElement('li')
        li.textContent = `${className}: ${probability.toFixed(4)}`
        predictionsList.appendChild(li)
      }
    }
    resultsSection.hidden = false
    statusEl.textContent = ''
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`
  }
}

// A text model is given the text in the box (a string, not a file).
classifyBtn.addEventListener('click', () => {
  if (textBox.value.trim() === '') {
    statusEl.textContent = 'Type or paste some text first.'
    return
  }
  handleFile(textBox.value)
})

fileInput.addEventListener('change', () => handleFile(fileInput.files?.[0]))
dropzone.addEventListener('dragover', (e) => e.preventDefault())
dropzone.addEventListener('drop', (e) => {
  e.preventDefault()
  handleFile(e.dataTransfer.files?.[0])
})

// ---- Built-in self-check (bundled sample vs expected.json) ----

// Object detection. The platform's prediction comes from the original weights through Ultralytics, which pads an
// image only up to a multiple of 32 before the network; this client pads to the full square the exported graph was
// built for. The same box therefore comes out a little different, and a borderline one may appear in only one list.
// So only confident boxes are held to account, and by overlap and a loose confidence margin, not exact numbers.
const CONFIDENT = 0.35 // a box at least this confident on either side must have a partner on the other
const MIN_IOU = 0.7
const CONFIDENCE_MARGIN = 0.1

function boxIou(a, b) {
  const width = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)
  const height = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y)
  const intersection = Math.max(width, 0) * Math.max(height, 0)
  const union = a.w * a.h + b.w * b.h - intersection
  return union > 0 ? intersection / union : 0
}

/** Whether `others` has a box of the same class that overlaps enough and is about as confident. */
function hasPartner(box, others) {
  const same = others.filter((o) => o.label === box.label)
  if (same.length === 0) return false
  const best = same.reduce((a, b) => (boxIou(box.box, b.box) > boxIou(box.box, a.box) ? b : a))
  return boxIou(box.box, best.box) >= MIN_IOU && Math.abs(best.confidence - box.confidence) <= CONFIDENCE_MARGIN
}

function verifyDetections(expected, actual) {
  const mismatches = []
  for (const [source, side, other, otherName] of [
    [expected, 'platform', actual, 'this app'],
    [actual, 'this app', expected, 'the platform'],
  ]) {
    for (const box of source) {
      if (box.confidence >= CONFIDENT && !hasPartner(box, other)) {
        mismatches.push(`${side} found '${box.label}' (${box.confidence.toFixed(2)}); ${otherName} has no matching box`)
      }
    }
  }
  const confident = expected.filter((box) => box.confidence >= CONFIDENT).length
  return mismatches.length === 0
    ? `OK — ${confident} confident box(es) match the platform's output (IoU >= ${MIN_IOU}).`
    : `MISMATCH — ${mismatches.join('; ')}`
}

const verifyBtn = document.getElementById('verify-btn')
const verifyResult = document.getElementById('verify-result')

verifyBtn.addEventListener('click', async () => {
  verifyResult.textContent = 'Running…'
  try {
    const res = await fetch(new URL('expected.json', HERE))
    if (!res.ok) {
      verifyResult.textContent = 'No expected.json in this bundle — nothing to verify against.'
      return
    }
    const expected = await res.json()
    await init()
    if (isAudio() && !expected.sampleFile.toLowerCase().endsWith('.wav')) {
      verifyResult.textContent = "The bundled sample is not a WAV file, which this app can't read exactly — nothing to verify."
      return
    }
    const sampleRes = await fetch(new URL(expected.sampleFile, HERE))
    // A text model is given the text in the sample file (UTF-8), the rest the file itself.
    const sampleBlob = isText() || isTagger() ? await sampleRes.text() : await sampleRes.blob()

    if (isTagger()) {
      // A token classifier returns one tag per word, which must be the platform's, word for word.
      if (!expected.tags) {
        verifyResult.textContent = 'expected.json has no tags to compare against.'
        return
      }
      const found = (await predict(sampleBlob)).map((w) => w.tag)
      const wrong = []
      if (found.length !== expected.tags.length) wrong.push(`expected ${expected.tags.length} tags, got ${found.length}`)
      expected.tags.forEach((want, i) => {
        if (i < found.length && found[i] !== want) wrong.push(`word ${i}: expected '${want}', got '${found[i]}'`)
      })
      verifyResult.textContent =
        wrong.length === 0 ? `OK — all ${found.length} tags match the platform's output.` : `MISMATCH — ${wrong.join('; ')}`
      return
    }

    if (outputSpec?.type === 'detections' || expected.outputType === 'detections') {
      if (!expected.detections) {
        verifyResult.textContent = 'expected.json has no detections to compare against.'
        return
      }
      verifyResult.textContent = verifyDetections(expected.detections, await detect(sampleBlob))
      return
    }

    const entries = Object.entries(expected.predictions)
    const actual = await predict(sampleBlob, entries.length || 5)

    if (entries.length === 0) {
      verifyResult.textContent = 'expected.json has no predictions to compare against.'
      return
    }

    const TOLERANCE = 1e-3
    const expectedTop1 = entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0]
    const actualEntries = Object.entries(actual)
    const actualTop1 = actualEntries.length > 0 ? actualEntries.reduce((a, b) => (b[1] > a[1] ? b : a))[0] : null

    const mismatches = []
    if (actualTop1 !== expectedTop1) mismatches.push(`top-1 class: expected '${expectedTop1}', got '${actualTop1}'`)
    for (const [className, expectedProb] of entries) {
      const actualProb = actual[className]
      if (actualProb === undefined) {
        mismatches.push(`class '${className}' missing from this client's output`)
        continue
      }
      const diff = Math.abs(actualProb - expectedProb)
      if (diff > TOLERANCE)
        mismatches.push(`class '${className}': expected ${expectedProb.toFixed(4)}, got ${actualProb.toFixed(4)}`)
    }

    verifyResult.textContent =
      mismatches.length === 0
        ? `OK — top-1 '${actualTop1}' matches platform output within ${TOLERANCE}.`
        : `MISMATCH — ${mismatches.join('; ')}`
  } catch (err) {
    verifyResult.textContent = `Error: ${err.message}`
  }
})

// Register the service worker so the app is installable and works offline
// once the model has been fetched once.
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register(new URL('sw.js', HERE)).catch(() => {})
}
