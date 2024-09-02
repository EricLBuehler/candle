## Running [wuersthcnen] Examples

### Xtask
one can compile this example for wasm and start a web server with the following command:
```bash
cargo xtask run-wasm -- --release
```
Then open `http://localhost:80` in your browser.


### Vanilla JS

To build and test the UI made in Vanilla JS and WebWorkers, first we need to build the WASM library:

```bash
sh build-lib.sh
```

This will bundle the library under `./build` and we can import it inside our WebWorker like a normal JS module:

```js
import init, { Model } from "./build/m.js";
```

The full example can be found under `./index.html`. All needed assets are fetched from the web, so no need to download anything.
Finally, you can preview the example by running a local HTTP server. For example:

```bash
python -m http.server
```

Then open `http://localhost:8000/index.html` in your browser.


Please note that the model download will take some time. The Chrome Network tab may not show the download accurately. 