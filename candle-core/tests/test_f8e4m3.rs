use candle_core::{DType, Device, Tensor, Result};

#[test]
fn test_f8e4m3_dtype() -> Result<()> {
    // Test basic tensor creation
    let device = Device::Cpu;
    
    // Create a small tensor with F8E4M3 dtype
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(data, (2, 2), &device)?;
    
    // Convert to F8E4M3
    let tensor_f8 = tensor.to_dtype(DType::F8E4M3)?;
    assert_eq!(tensor_f8.dtype(), DType::F8E4M3);
    
    // Convert back to F32
    let tensor_back = tensor_f8.to_dtype(DType::F32)?;
    
    // Values should be approximately the same (within F8E4M3 precision)
    let original_vec = tensor.flatten_all()?.to_vec1::<f32>()?;
    let back_vec = tensor_back.flatten_all()?.to_vec1::<f32>()?;
    
    for (orig, back) in original_vec.iter().zip(back_vec.iter()) {
        // F8E4M3 has limited precision, so we allow for some error
        let diff = (orig - back).abs();
        assert!(diff < 0.2, "Value {} converted to {} (diff: {})", orig, back, diff);
    }
    
    Ok(())
}

#[test] 
fn test_f8e4m3_safetensors() -> Result<()> {
    let device = Device::Cpu;
    
    // Create tensor and convert to F8E4M3
    let data = vec![0.5f32, 1.0, 1.5, 2.0];
    let tensor = Tensor::from_vec(data, (2, 2), &device)?;
    let tensor_f8 = tensor.to_dtype(DType::F8E4M3)?;
    
    // Save to safetensors
    tensor_f8.save_safetensors("test_f8e4m3", "test_f8e4m3.safetensors")?;
    
    // Load it back
    let loaded = candle_core::safetensors::load("test_f8e4m3.safetensors", &device)?;
    let loaded_tensor = loaded.get("test_f8e4m3").expect("Tensor not found");
    
    assert_eq!(loaded_tensor.dtype(), DType::F8E4M3);
    assert_eq!(loaded_tensor.dims(), tensor_f8.dims());
    
    // Clean up
    std::fs::remove_file("test_f8e4m3.safetensors").ok();
    
    Ok(())
}