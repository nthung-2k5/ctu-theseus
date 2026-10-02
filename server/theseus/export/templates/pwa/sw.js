// Minimal offline-cache service worker. Caches this app's own files
// (including model.onnx / preprocessing.json, fetched via relative URLs by
// app.js) on first load so the app keeps working — and stays installable —
// without a network connection afterward.

const CACHE_NAME = 'theseus-app-v1'
const CORE_ASSETS = ['./', './index.html', './app.js', './style.css', './manifest.webmanifest', './icon.svg']

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE_ASSETS)))
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))),
  )
  self.clients.claim()
})

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return
  event.respondWith(
    caches.match(event.request).then(
      (cached) =>
        cached ??
        fetch(event.request).then((response) => {
          const copy = response.clone()
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy))
          return response
        }),
    ),
  )
})
