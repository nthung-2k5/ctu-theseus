/**
 * OmniTrain-generated inference client (browser). Wraps the exported ONNX
 * model with the preprocessing/postprocessing Ludwig applied at training
 * time, read from preprocessing.json — see the devkit export's client for
 * the same logic in Python/TypeScript/C#/Java.
 *
 * Implements preprocessing for `image` inputs with `category` outputs
 * (image classification) and for a `detections` output (object detection,
 * YOLO): the ONNX graph returns every candidate box, so for a detector this
 * app does the confidence filter, per-class non-maximum suppression and the
 * undoing of the letterbox, and `predict()` returns the boxes — see README.md.
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

function softmax(logits) {
  const max = Math.max(...logits)
  const exp = logits.map((v) => Math.exp(v - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map((v) => v / sum)
}

async function init() {
  const res = await fetch(new URL('preprocessing.json', HERE))
  const manifest = await res.json()
  inputSpec = manifest.inputs[0]
  outputSpec = manifest.outputs[0]

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
  const pp = inputSpec.ludwigPreprocessing ?? {}
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
  if (!session) await init()
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
 * Run inference on an image File/Blob and return {className: probability} (classification), {value} (regression),
 * or the list of boxes found (an object detector; see `detect`).
 */
export async function predict(imageSource, topK = 5) {
  if (!session) await init()
  if (outputSpec.type === 'detections') return detect(imageSource)

  if (inputSpec.type !== 'image') {
    throw new Error(
      `No preprocessing implemented for input type '${inputSpec.type}'. ` +
        'See preprocessing.json inputs[0].ludwigPreprocessing and adapt app.js.',
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
  preview.src = URL.createObjectURL(file)
  preview.hidden = false

  try {
    const predictions = await predict(file)
    predictionsList.innerHTML = ''
    if (Array.isArray(predictions)) {
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
    const sampleRes = await fetch(new URL(expected.sampleFile, HERE))
    const sampleBlob = await sampleRes.blob()

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
