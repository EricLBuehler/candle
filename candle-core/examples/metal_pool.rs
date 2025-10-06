use candle_core::{DType, Device, Result, Tensor};

#[cfg(feature = "metal")]
fn run() -> Result<()> {
    if !candle_core::utils::metal_is_available() {
        eprintln!("Metal is not available on this system.");
        return Ok(());
    }

    let device = Device::new_metal(0)?;

    let input = Tensor::arange(0f32, 16f32, &Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_device(&device)?;

    // Allocate a 4 MB pool for intermediate activations.
    let pooled = input.start_pool(4 * 1024 * 1024)?;

    let logits = pooled.sin()?.mul(&pooled.cos()?)?;
    let final_tensor = logits.tanh()?.leave_pool()?;

    println!("final tensor: {:?}", final_tensor.to_vec1::<f32>()?);

    Ok(())
}

#[cfg(not(feature = "metal"))]
fn run() -> Result<()> {
    eprintln!("Rebuild candle-core with the `metal` feature to run this example.");
    Ok(())
}

fn main() -> Result<()> {
    run()
}
