/**
 * PixelPen Service Worker - Offline Shell & Asset Caching
 */

const CACHE_NAME = "pixelpen-v1.0.0";
const OFFLINE_ASSETS = [
  "./",
  "./index.html",
  "./style.css",
  "./canvas.js",
  "./upload.js",
  "./pipeline.js",
  "./game.js",
  "./app.js",
  "./manifest.json",
  "./icons/icon-192.svg",
  "./icons/icon-512.svg"
];

// Install: Cache core application shell
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("[ServiceWorker] Caching app shell assets...");
      return cache.addAll(OFFLINE_ASSETS);
    }).then(() => self.skipWaiting())
  );
});

// Activate: Clean up old cache versions
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(
        keyList.map((key) => {
          if (key !== CACHE_NAME) {
            console.log("[ServiceWorker] Removing obsolete cache:", key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch: Network-first for dynamic API calls, Cache-first / Stale-While-Revalidate for app shell
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Skip caching for backend API requests (/predict, /health, /feedback)
  if (url.pathname.includes("/predict") || url.pathname.includes("/health") || url.pathname.includes("/feedback")) {
    event.respondWith(fetch(event.request));
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        // Return cached version and update in background
        fetch(event.request).then((networkResponse) => {
          if (networkResponse && networkResponse.status === 200) {
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, networkResponse);
            });
          }
        }).catch(() => {});
        return cachedResponse;
      }
      return fetch(event.request);
    })
  );
});
