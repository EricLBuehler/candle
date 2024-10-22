//load Candle Bert Module wasm module
import init, { Model } from "./build/m.js";

async function fetchArrayBuffer(url) {
  const cacheName = "bert-candle-cache";
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(url);
  if (cachedResponse) {
    const data = await cachedResponse.arrayBuffer();
    return new Uint8Array(data);
  }
  const res = await fetch(url, { cache: "force-cache" });
  cache.put(url, res.clone());
  return new Uint8Array(await res.arrayBuffer());
}
class Bert {
  static instance = {};

  static async getInstance(weightsURL, tokenizerURL, configURL, modelID, useWgpu) {
    if (!this.instance[modelID + useWgpu]) {
      await init();

      self.postMessage({ status: "loading", message: "Loading Model" });
      const [weightsArrayU8, tokenizerArrayU8, mel_filtersArrayU8] =
        await Promise.all([
          fetchArrayBuffer(weightsURL),
          fetchArrayBuffer(tokenizerURL),
          fetchArrayBuffer(configURL),
        ]);

      this.instance[modelID + useWgpu] = await new Model(
        weightsArrayU8,
        tokenizerArrayU8,
        mel_filtersArrayU8,
        useWgpu
      );
    } else {
      self.postMessage({ status: "ready", message: "Model Already Loaded" });
    }
    return this.instance[modelID + useWgpu];
  }
}

self.addEventListener("message", async (event) => {
  const {
    weightsURL,
    tokenizerURL,
    configURL,
    modelID,
    sentences,
    normalize = true,
    useWgpu
  } = event.data;
  try {
    self.postMessage({ status: "ready", message: "Starting Bert Model" });
    const model = await Bert.getInstance(
      weightsURL,
      tokenizerURL,
      configURL,
      modelID,
      useWgpu
    );
    self.postMessage({
      status: "embedding",
      message: "Calculating Embeddings",
    });
    const output = await model.get_embeddings({
      sentences: sentences,
      normalize_embeddings: normalize,
    });

    self.postMessage({
      status: "complete",
      message: "complete",
      output: output.data,
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
});
