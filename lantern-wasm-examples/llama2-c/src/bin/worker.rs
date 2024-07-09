use yew_agent::PublicWorker;
fn main() {
    console_error_panic_hook::set_once();
    lantern_wasm_example_llama2::Worker::register();
}
