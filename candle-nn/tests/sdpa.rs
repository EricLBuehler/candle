#[cfg(feature = "metal")]
#[test]
fn sdpa() -> candle::Result<()> {
    use candle::{DType, Device, Tensor};

    const BS: usize = 1;
    const R: usize = 4;
    const L: usize = 4;
    const DK: usize = 64;
    const H: usize = 3;
    let scale: f64 = f64::from(DK as u32).sqrt().recip();

    let device = Device::new_metal(0)?;

    let q = Tensor::randn(0f32, 1f32, (BS, H, R, DK), &device)?;
    let k = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;
    let v = Tensor::randn(0f32, 1f32, (BS, H, L, DK), &device)?;

    let ground_truth = {
        let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
        let att =
            candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
        att.matmul(&v.clone())?
    };

    let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, scale as f32)?;

    assert_eq!(ground_truth.shape(), sdpa_output.shape());

    let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
        .sum_all()?
        .to_scalar()?;

    assert!(error <= 0.0001);

    Ok(())
}
