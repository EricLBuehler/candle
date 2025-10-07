use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    // This requires the code to be run with MTL_CAPTURE_ENABLED=1
    let device = Device::new_metal(0)?;
    
    let x = Tensor::randn(0f32, 1.0, (128, 128), &device)?;
    let x1 = x.add(&x)?;
    println!("{x1}");
    Ok(())
}
