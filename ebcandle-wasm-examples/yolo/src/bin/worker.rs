use yew_agent::PublicWorker;
fn main() {
    console_error_panic_hook::set_once();
    ebcandle_wasm_example_yolo::Worker::register();
}
