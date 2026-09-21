/**
 * Theseus-generated inference client (browser). Wraps the exported ONNX
 * model with the preprocessing/postprocessing Ludwig applied at training
 * time, read from preprocessing.json — see the devkit export's client for
 * the same logic in Python/TypeScript/C#/Java.
 *
 * Implements preprocessing for `image` inputs with `category` outputs
 * (image classification) only — see README.md.
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

/** Decode an ImageBitmap into a CHW float32 tensor. */
function preprocessImage(imageBitmap) {
  const pp = inputSpec.ludwigPreprocessing ?? {}
  const width = pp.width ?? 224
  const height = pp.height ?? 224

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  ctx.drawImage(imageBitmap, 0, 0, width, height)
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

/** Run inference on an image File/Blob and return {className: probability} (classification) or {value} (regression). */
export async function predict(imageSource, topK = 5) {
  if (!session) await init()

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
const resultsSection = document.getElementById('results')
const predictionsList = document.getElementById('predictions')
const statusEl = document.getElementById('status')

async function handleFile(file) {
  if (!file) return
  statusEl.textContent = 'Running inference…'
  preview.src = URL.createObjectURL(file)
  preview.hidden = false

  try {
    const predictions = await predict(file)
    predictionsList.innerHTML = ''
    for (const [className, probability] of Object.entries(predictions)) {
      const li = document.createElement('li')
      li.textContent = `${className}: ${probability.toFixed(4)}`
      predictionsList.appendChild(li)
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
