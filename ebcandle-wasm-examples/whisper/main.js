import init, { run_app } from './pkg/ebcandle_wasm_example_whisper.js';
async function main() {
   await init('/pkg/ebcandle_wasm_example_whisper_bg.wasm');
   run_app();
}
main()
