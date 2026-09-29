/**
 * Background Service Worker (Manifest V3)
 * Handles asynchronous communication with the local FastAPI phishing detection engine.
 */

const API_ENDPOINT = "http://127.0.0.1:8000/predict";
const urlCache = new Map();

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "check_url") {
    const targetUrl = request.url;

    if (!targetUrl || !targetUrl.startsWith("http")) {
      sendResponse({ status: "ignored" });
      return true;
    }

    // Check in-memory cache
    if (urlCache.has(targetUrl)) {
      sendResponse(urlCache.get(targetUrl));
      return true;
    }

    // Query local FastAPI detector
    fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: targetUrl, include_regional: true })
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        urlCache.set(targetUrl, data);
        sendResponse(data);
      })
      .catch(error => {
        console.warn("[PhishingDetector] Local API unreachable:", error);
        sendResponse({ error: "Backend detector offline" });
      });

    return true; // Keep message channel open for async response
  }
});
