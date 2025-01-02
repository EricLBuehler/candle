#![allow(unused_imports, unexpected_cfgs)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{test_device, test_utils, DType, Device, IndexOp, Result, Tensor, D};
async fn zeros(device: &Device) -> Result<()> {
    let tensor = Tensor::zeros((5, 2), DType::F32, device)?;
    let (dim1, dim2) = tensor.dims2()?;
    assert_eq!(dim1, 5);
    assert_eq!(dim2, 2);
    Ok(())
}
async fn ones(device: &Device) -> Result<()> {
    if device.is_dtype_available(DType::U8) {
        assert_eq!(
            Tensor::ones((2, 3), DType::U8, device) ?.to_vec2_async::< u8 > (). await ?,
            [[1, 1, 1], [1, 1, 1]],
        );
    }
    if device.is_dtype_available(DType::U32) {
        assert_eq!(
            Tensor::ones((2, 3), DType::U32, device) ?.to_vec2_async::< u32 > (). await
            ?, [[1, 1, 1], [1, 1, 1]],
        );
    }
    if device.is_dtype_available(DType::I64) {
        assert_eq!(
            Tensor::ones((2, 3), DType::I64, device) ?.to_vec2_async::< i64 > (). await
            ?, [[1, 1, 1], [1, 1, 1]],
        );
    }
    assert_eq!(
        Tensor::ones((2, 3), DType::F32, device) ?.to_vec2_async::< f32 > (). await ?,
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    );
    if device.is_dtype_available(DType::F64) {
        assert_eq!(
            Tensor::ones((2, 3), DType::F64, device) ?.to_vec2_async::< f64 > (). await
            ?, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        );
    }
    if device.is_dtype_available(DType::F16) {
        assert_eq!(
            Tensor::ones((2, 3), DType::F16, device) ?.to_vec2_async::< half::f16 > ().
            await ?, [[half::f16::from_f32(1.0), half::f16::from_f32(1.0),
            half::f16::from_f32(1.0)], [half::f16::from_f32(1.0),
            half::f16::from_f32(1.0), half::f16::from_f32(1.0)]],
        );
        assert_eq!(
            Tensor::ones((2, 3), DType::BF16, device) ?.to_vec2_async::< half::bf16 > ().
            await ?, [[half::bf16::from_f32(1.0), half::bf16::from_f32(1.0),
            half::bf16::from_f32(1.0)], [half::bf16::from_f32(1.0),
            half::bf16::from_f32(1.0), half::bf16::from_f32(1.0)]],
        );
    }
    Ok(())
}
async fn full(device: &Device) -> Result<()> {
    assert_eq!(
        Tensor::full(42u32, (2, 3), device) ?.to_vec2_async::< u32 > (). await ?, [[42,
        42, 42], [42, 42, 42]],
    );
    Ok(())
}
async fn arange(device: &Device) -> Result<()> {
    if device.is_dtype_available(DType::U8) {
        assert_eq!(
            Tensor::arange(0u8, 5u8, device) ?.to_vec1_async::< u8 > (). await ?, [0, 1,
            2, 3, 4],
        );
        assert_eq!(
            Tensor::arange_step(0u8, 5u8, 2, device) ?.to_vec1_async::< u8 > (). await ?,
            [0, 2, 4],
        );
        assert_eq!(
            Tensor::arange_step(0u8, 5u8, 3, device) ?.to_vec1_async::< u8 > (). await ?,
            [0, 3],
        );
    }
    if device.is_dtype_available(DType::I64) {
        assert_eq!(
            Tensor::arange_step(5i64, 0i64, - 1, device) ?.to_vec1_async::< i64 > ().
            await ?, [5, 4, 3, 2, 1],
        );
    }
    Ok(())
}
async fn add_mul(device: &Device) -> Result<()> {
    let tensor = Tensor::new(&[3f32, 1., 4.], device)?;
    let dim1 = tensor.dims1()?;
    assert_eq!(dim1, 3);
    let content: Vec<f32> = tensor.to_vec1_async().await?;
    assert_eq!(content, [3., 1., 4.]);
    let tensor = Tensor::add(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1_async().await?;
    assert_eq!(content, [6., 2., 8.]);
    let tensor = Tensor::mul(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1_async().await?;
    assert_eq!(content, [36., 4., 64.]);
    Ok(())
}
async fn tensor_2d(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2_async().await?;
    assert_eq!(content, data);
    Ok(())
}
async fn clamp(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let tensor = tensor.clamp(1.5, 6.2)?;
    assert_eq!(
        tensor.to_vec2_async::< f32 > (). await ?, [[3.0, 1.5, 4.0, 1.5, 5.0], [2.0, 1.5,
        6.2, 6.2, 2.0]],
    );
    Ok(())
}
async fn asort(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1.1, 5.], [2.1, 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let indexes = tensor.arg_sort_last_dim(true)?;
    assert_eq!(
        indexes.to_vec2_async::< u32 > (). await ?, [[1, 3, 0, 2, 4], [1, 4, 0, 2, 3]],
    );
    let indexes = tensor.arg_sort_last_dim(false)?;
    assert_eq!(
        indexes.to_vec2_async::< u32 > (). await ?, [[4, 2, 0, 3, 1], [3, 2, 0, 4, 1]],
    );
    let (sorted, indexes) = tensor.sort_last_dim(true)?;
    assert_eq!(
        indexes.to_vec2_async::< u32 > (). await ?, [[1, 3, 0, 2, 4], [1, 4, 0, 2, 3]],
    );
    assert_eq!(
        sorted.to_vec2_async::< f32 > (). await ?, [[1.0, 1.1, 3.0, 4.0, 5.0], [1.0, 2.0,
        2.1, 7.0, 8.0]]
    );
    let (sorted, indexes) = tensor.sort_last_dim(false)?;
    assert_eq!(
        indexes.to_vec2_async::< u32 > (). await ?, [[4, 2, 0, 3, 1], [3, 2, 0, 4, 1]],
    );
    assert_eq!(
        sorted.to_vec2_async::< f32 > (). await ?, [[5.0, 4.0, 3.0, 1.1, 1.0], [8.0, 7.0,
        2.1, 2.0, 1.0]]
    );
    Ok(())
}
async fn unary_op(device: &Device) -> Result<()> {
    let data = &[[-3f32, 1., 4., -0.1, 0.5], [2.7, -1.8, -0.28, 1.8, 2.8]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        to_vec2_round_async(& tensor.gelu() ?, 4). await ?, [[- 0.0036, 0.8412, 3.9999, -
        0.046, 0.3457], [2.6911, - 0.0647, - 0.1091, 1.7353, 2.7933]]
    );
    if device.is_dtype_available(DType::F16) {
        let t_f16 = tensor.to_dtype(DType::F16)?.gelu()?.to_dtype(DType::F32)?;
        let max_diff = (tensor.gelu()? - t_f16)?.flatten_all()?.max(0)?;
        assert!(max_diff.to_vec0_async::< f32 > (). await ? < 5e-3);
        assert_eq!(
            to_vec2_round_async(& tensor.gelu_erf() ?, 4). await ?, [[- 0.004, 0.8413,
            3.9999, - 0.046, 0.3457], [2.6906, - 0.0647, - 0.1091, 1.7353, 2.7928]]
        );
    }
    assert_eq!(
        to_vec2_round_async(& tensor.erf() ?, 4). await ?, [[- 1.0, 0.8427, 1.0, -
        0.1125, 0.5205], [0.9999, - 0.9891, - 0.3079, 0.9891, 0.9999]]
    );
    assert_eq!(
        to_vec2_round_async(& tensor.silu() ?, 4). await ?, [[- 0.1423, 0.7311, 3.9281, -
        0.0475, 0.3112], [2.53, - 0.2553, - 0.1205, 1.5447, 2.6395]]
    );
    assert_eq!(
        to_vec2_round_async(& tensor.ceil() ?, 4). await ?, [[- 3.0, 1.0, 4.0, - 0.0,
        1.0], [3.0, - 1.0, - 0.0, 2.0, 3.0]]
    );
    assert_eq!(
        to_vec2_round_async(& tensor.floor() ?, 4). await ?, [[- 3.0, 1.0, 4.0, - 1.0,
        0.0], [2.0, - 2.0, - 1.0, 1.0, 2.0]]
    );
    assert_eq!(
        to_vec2_round_async(& tensor.round() ?, 4). await ?, [[- 3.0, 1.0, 4.0, - 0.0,
        1.0], [3.0, - 2.0, - 0.0, 2.0, 3.0]]
    );
    let tensor = Tensor::new(&[2997.9246, 314.15926f32], device)?;
    assert_eq!(
        to_vec1_round_async(& tensor.round_to(2) ?, 4). await ?, [2997.92, 314.16]
    );
    assert_eq!(
        to_vec1_round_async(& tensor.round_to(- 2) ?, 4). await ?, [3000.0, 300.]
    );
    let tensor = Tensor::new(
        &[-1.01f32, -0.9, -0.1, 0.0, -0.0, 0.1, 0.9, 1.0, 1.1],
        device,
    )?;
    assert_eq!(
        tensor.sign() ?.to_vec1_async::< f32 > (). await ?, [- 1., - 1., - 1., 0., 0.,
        1., 1., 1., 1.]
    );
    let tensor = Tensor::new(&[-1.0f32, 0., -2., 3.], device)?;
    let y = tensor.elu(2.)?;
    assert_eq!(
        to_vec1_round_async(& y, 4). await ?, [- 1.2642, 0.0000, - 1.7293, 3.0000]
    );
    let y = tensor.reshape((2, 2))?.t()?.elu(2.)?.flatten_all()?;
    assert_eq!(
        to_vec1_round_async(& y, 4). await ?, [- 1.2642, - 1.7293, 0.0000, 3.0000]
    );
    Ok(())
}
async fn binary_op(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor1 = Tensor::new(data, device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 1., 7., 8., 2.]];
    let tensor2 = Tensor::new(data2, device)?;
    let tensor = (&tensor1 + (&tensor1 * &tensor1)? / (&tensor1 + &tensor2))?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2_async().await?;
    assert_eq!(content[0], [4.125, 1.1666666, 5.7777777, 1.1666666, 7.5]);
    assert_eq!(content[1], [3.0, 1.5, 10.5, 12.0, 3.0]);
    #[allow(clippy::eq_op)]
    let tensor = (&tensor - &tensor)?;
    let content: Vec<Vec<f32>> = tensor.to_vec2_async().await?;
    assert_eq!(content[0], [0., 0., 0., 0., 0.]);
    let min = tensor1.minimum(&(&tensor2 * 0.5)?)?;
    let max = tensor1.maximum(&(&tensor2 * 0.5)?)?;
    assert_eq!(
        min.to_vec2_async::< f32 > (). await ?, [[2.5, 1.0, 2.5, 1.0, 2.5], [1.0, 0.5,
        3.5, 4.0, 1.0]],
    );
    assert_eq!(
        max.to_vec2_async::< f32 > (). await ?, [[3.0, 2.5, 4.0, 2.5, 5.0], [2.0, 1.0,
        7.0, 8.0, 2.0]]
    );
    Ok(())
}
async fn transpose(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?.t()?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (5, 2));
    assert_eq!(
        tensor.to_vec2_async::< f32 > (). await ?, & [[3f32, 2.], [1., 1.], [4., 7.],
        [1., 8.], [5., 2.]]
    );
    assert_eq!(tensor.t() ?.to_vec2_async::< f32 > (). await ?, data);
    assert_eq!(tensor.contiguous() ?.t() ?.to_vec2_async::< f32 > (). await ?, data);
    assert_eq!(((tensor + 1.) ?.t() ? - 1.) ?.to_vec2_async::< f32 > (). await ?, data);
    Ok(())
}
async fn var(device: &Device) -> Result<()> {
    let data = &[
        [0.2035f32, 1.2959, 1.8101, -0.4644],
        [1.5027, -0.3270, 0.5905, 0.6538],
        [-1.5745, 1.3330, -0.5596, -0.6548],
        [0.1264, -0.5080, 1.6420, 0.1992],
    ];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        to_vec2_round_async(& tensor.var_keepdim(1) ?, 4). await ?, & [[1.0631], [0.559],
        [1.4893], [0.8258]]
    );
    Ok(())
}
async fn sum(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.sum_keepdim(2) ?.to_vec3_async::< u32 > (). await ?, & [[[8], [15]],
        [[10], [18]]]
    );
    assert_eq!(
        tensor.sum_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[5, 2, 11], [9,
        7, 17]]],
    );
    assert_eq!(
        tensor.sum_keepdim((0, 2, 1)) ?.to_vec3_async::< u32 > (). await ?, & [[[51]]],
    );
    assert_eq!(
        tensor.t() ?.sum_keepdim(1) ?.t() ?.to_vec3_async::< u32 > (). await ?, & [[[8],
        [15]], [[10], [18]]]
    );
    assert_eq!(
        tensor.sum_keepdim((2, 1)) ?.to_vec3_async::< u32 > (). await ?, & [[[8 + 15]],
        [[10 + 18]]]
    );
    let data: Vec<u32> = (0..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.sum_keepdim(0) ?.to_vec1_async::< u32 > (). await ?, & [7998000]);
    let tensor = tensor.reshape((2000, 2))?;
    assert_eq!(
        tensor.sum_keepdim((0, 1)) ?.to_vec2_async::< u32 > (). await ?, & [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0) ?.sum_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(1) ?.sum_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[3998000,
        4000000]]
    );
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.sum_keepdim((0, 1)) ?.to_vec2_async::< u32 > (). await ?, & [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0) ?.sum_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(1) ?.sum_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[3998000,
        4000000]]
    );
    let t1 = tensor.reshape((200, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.sum_keepdim((0, 1, 2)) ?.to_vec3_async::< u32 > (). await ?, &
            [[[7998000]]]
        );
        assert_eq!(
            tensor.sum_keepdim(0) ? .sum_keepdim(2) ? .sum_keepdim(1) ? .to_vec3_async::<
            u32 > (). await ?, & [[[7998000]]]
        );
        assert_eq!(
            tensor.sum_keepdim(0) ? .sum_keepdim((1, 2)) ? .to_vec3_async::< u32 > ().
            await ?, & [[[7998000]]]
        );
        assert_eq!(
            tensor.sum_keepdim(1) ? .sum_keepdim((0, 2)) ? .to_vec3_async::< u32 > ().
            await ?, & [[[7998000]]]
        );
        assert_eq!(
            tensor.sum_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[398000,
            398200, 398400, 398600], [398800, 399000, 399200, 399400], [399600, 399800,
            400000, 400200], [400400, 400600, 400800, 401000], [401200, 401400, 401600,
            401800]]]
        );
    }
    Ok(())
}
async fn min(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.min_keepdim(2) ?.to_vec3_async::< u32 > (). await ?, & [[[1], [1]], [[1],
        [2]]]
    );
    assert_eq!(
        tensor.min_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[2, 1, 4], [1, 2,
        8]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.min_keepdim(0) ?.to_vec1_async::< u32 > (). await ?, & [200]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.min_keepdim(0) ?.min_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[200]]
    );
    assert_eq!(
        tensor.min_keepdim(1) ?.min_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[200]]
    );
    assert_eq!(
        tensor.min_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[200, 201]]
    );
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.min_keepdim(0) ?.min_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[200]]
    );
    assert_eq!(
        tensor.min_keepdim(1) ?.min_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[200]]
    );
    assert_eq!(
        tensor.min_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[200, 201]]
    );
    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.min_keepdim(0) ? .min_keepdim(2) ? .min_keepdim(1) ? .to_vec3_async::<
            u32 > (). await ?, & [[[200]]]
        );
        assert_eq!(
            tensor.min_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[200, 201,
            202, 203], [204, 205, 206, 207], [208, 209, 210, 211], [212, 213, 214, 215],
            [216, 217, 218, 219]]]
        );
    }
    Ok(())
}
async fn max(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.max_keepdim(2) ?.to_vec3_async::< u32 > (). await ?, & [[[4], [9]], [[7],
        [8]]]
    );
    assert_eq!(
        tensor.max_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[3, 1, 7], [8, 5,
        9]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.max_keepdim(0) ?.to_vec1_async::< u32 > (). await ?, & [3999]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.max_keepdim(0) ?.max_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(1) ?.max_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[3998, 3999]]
    );
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.max_keepdim(0) ?.max_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(1) ?.max_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[3998, 3999]]
    );
    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.max_keepdim(0) ? .max_keepdim(2) ? .max_keepdim(1) ? .to_vec3_async::<
            u32 > (). await ?, & [[[3999]]]
        );
        assert_eq!(
            tensor.max_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[3980, 3981,
            3982, 3983], [3984, 3985, 3986, 3987], [3988, 3989, 3990, 3991], [3992, 3993,
            3994, 3995], [3996, 3997, 3998, 3999]]]
        );
    }
    Ok(())
}
async fn argmin(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.argmin_keepdim(2) ?.to_vec3_async::< u32 > (). await ?, & [[[1], [0]],
        [[1], [1]]]
    );
    assert_eq!(
        tensor.argmin_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[1, 0, 0], [0,
        1, 1]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.argmin_keepdim(0) ?.to_vec1_async::< u32 > (). await ?, & [0]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.argmin_keepdim(0) ? .argmin_keepdim(1) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmin_keepdim(1) ? .argmin_keepdim(0) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmin_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[0, 0]]
    );
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.argmin_keepdim(0) ? .argmin_keepdim(1) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmin_keepdim(1) ? .argmin_keepdim(0) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmin_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[0, 0]]
    );
    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.argmin_keepdim(0) ? .argmin_keepdim(2) ? .argmin_keepdim(1) ?
            .to_vec3_async::< u32 > (). await ?, & [[[0]]]
        );
        assert_eq!(
            tensor.argmin_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[0, 0, 0,
            0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],]]
        );
    }
    Ok(())
}
async fn argmax(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.argmax_keepdim(2) ?.to_vec3_async::< u32 > (). await ?, & [[[2], [2]],
        [[2], [0]]]
    );
    assert_eq!(
        tensor.argmax_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[0, 0, 1], [1,
        0, 0]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.argmax_keepdim(0) ?.to_vec1_async::< u32 > (). await ?, & [3799]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.argmax_keepdim(0) ? .argmax_keepdim(1) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmax_keepdim(1) ? .argmax_keepdim(0) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmax_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[1899, 1899]]
    );
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.argmax_keepdim(0) ? .argmax_keepdim(1) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmax_keepdim(1) ? .argmax_keepdim(0) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmax_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[1899, 1899]]
    );
    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.argmax_keepdim(0) ? .argmax_keepdim(2) ? .argmax_keepdim(1) ?
            .to_vec3_async::< u32 > (). await ?, & [[[0]]]
        );
        assert_eq!(
            tensor.argmax_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[189, 189,
            189, 189], [189, 189, 189, 189], [189, 189, 189, 189], [189, 189, 189, 189],
            [189, 189, 189, 189],]]
        );
    }
    Ok(())
}
async fn narrow(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.narrow(2, 1, 2) ?.to_vec3_async::< f32 > (). await ?, & [[[1.0, 4.0],
        [5.0, 9.0]], [[1.0, 7.0], [2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(1, 1, 1) ?.to_vec3_async::< f32 > (). await ?, & [[[1.0, 5.0,
        9.0]], [[8.0, 2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 0, 1) ?.to_vec3_async::< f32 > (). await ?, & [[[3.0, 1.0, 4.0],
        [1.0, 5.0, 9.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 1, 1) ?.to_vec3_async::< f32 > (). await ?, & [[[2.0, 1.0, 7.0],
        [8.0, 2.0, 8.0]]],
    );
    assert_eq!(
        tensor.t() ?.narrow(1, 1, 2) ?.to_vec3_async::< f32 > (). await ?, & [[[1.0,
        5.0], [4.0, 9.0]], [[1.0, 2.0], [7.0, 8.0]]],
    );
    Ok(())
}
async fn broadcast(device: &Device) -> Result<()> {
    let data = &[3f32, 1., 4.];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.broadcast_left((3, 1)) ?.to_vec3_async::< f32 > (). await ?, & [[[3.0,
        1.0, 4.0]], [[3.0, 1.0, 4.0]], [[3.0, 1.0, 4.0]]]
    );
    Ok(())
}
async fn slice_set(device: &Device) -> Result<()> {
    let (b, h, max_t, d) = (2, 4, 7, 3);
    let cache = Tensor::zeros((b, h, max_t, d), DType::F32, device)?;
    let tensor = Tensor::randn(0f32, 1f32, (b, h, 4, d), device)?;
    cache.slice_set(&tensor, 2, 0)?;
    let cache_t = cache.narrow(2, 0, 4)?;
    let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    cache.slice_set(&tensor, 2, 1)?;
    let cache_t = cache.narrow(2, 1, 4)?;
    let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    let ones = Tensor::ones((b, h, 1, d), DType::F32, device)?;
    cache.slice_set(&ones, 2, 6)?;
    let diff = cache.narrow(2, 5, 1)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    let diff = (cache.narrow(2, 6, 1)? - 1.)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>()
        .await?;
    assert_eq!(diff, 0.);
    Ok(())
}
async fn cat(device: &Device) -> Result<()> {
    let t1 = Tensor::new(&[3f32, 1., 4.], device)?;
    let t2 = Tensor::new(&[1f32, 5., 9., 2.], device)?;
    let t3 = Tensor::new(&[6f32, 5., 3., 5., 8., 9.], device)?;
    assert_eq!(
        Tensor::cat(& [& t1], 0) ?.to_vec1_async::< f32 > (). await ?, [3f32, 1., 4.],
    );
    assert_eq!(
        Tensor::cat(& [& t1, & t2], 0) ?.to_vec1_async::< f32 > (). await ?, [3f32, 1.,
        4., 1., 5., 9., 2.],
    );
    assert_eq!(
        Tensor::cat(& [& t1, & t2, & t3], 0) ?.to_vec1_async::< f32 > (). await ?, [3f32,
        1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8., 9.],
    );
    let data = &[[3f32, 1., 4., 1., 5.], [2., 7., 1., 8., 2.]];
    let t1 = Tensor::new(data, device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 7., 1., 8., 2.]];
    let t2 = Tensor::new(data2, device)?;
    assert_eq!(
        Tensor::cat(& [& t1, & t2], 0) ?.to_vec2_async::< f32 > (). await ?, [[3.0, 1.0,
        4.0, 1.0, 5.0], [2.0, 7.0, 1.0, 8.0, 2.0], [5.0, 5.0, 5.0, 5.0, 5.0], [2.0, 7.0,
        1.0, 8.0, 2.0]]
    );
    assert_eq!(
        Tensor::cat(& [& t1.t() ?, & t2.t() ?], 1) ? .t() ? .to_vec2_async::< f32 > ().
        await ?, [[3.0, 1.0, 4.0, 1.0, 5.0], [2.0, 7.0, 1.0, 8.0, 2.0], [5.0, 5.0, 5.0,
        5.0, 5.0], [2.0, 7.0, 1.0, 8.0, 2.0]]
    );
    assert_eq!(
        Tensor::cat(& [& t1, & t2], 1) ?.to_vec2_async::< f32 > (). await ?, [[3.0, 1.0,
        4.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [2.0, 7.0, 1.0, 8.0, 2.0, 2.0, 7.0, 1.0,
        8.0, 2.0]]
    );
    if device.is_dtype_available(DType::I64) {
        let t1 = Tensor::arange(0, 48i64, device)?.reshape((2, 6, 4))?;
        let t2 = Tensor::arange(100, 124i64, device)?.reshape((2, 3, 4))?;
        let t3 = Tensor::arange(10000, 10032i64, device)?.reshape((2, 4, 4))?;
        let t_cat = Tensor::cat(&[&t1, &t2, &t3], 1)?;
        let t1 = t1.t()?.contiguous()?.t()?;
        let t2 = t2.t()?.contiguous()?.t()?;
        let t3 = t3.t()?.contiguous()?.t()?;
        let t_cat2 = Tensor::cat(&[&t1, &t2, &t3], 1)?;
        let diff = t_cat.eq(&t_cat2)?.to_dtype(DType::F32)?.sum_all()?;
        assert_eq!(diff.to_vec0_async::< f32 > (). await ?, 104.0);
        assert_eq!(t_cat.i((0, 0, 0)) ?.to_vec0_async::< i64 > (). await ?, 0);
        assert_eq!(t_cat.i((0, 4, 0)) ?.to_vec0_async::< i64 > (). await ?, 16);
        assert_eq!(t_cat.i((0, 5, 0)) ?.to_vec0_async::< i64 > (). await ?, 20);
        assert_eq!(t_cat.i((1, 5, 0)) ?.to_vec0_async::< i64 > (). await ?, 44);
        assert_eq!(t_cat.i((0, 6, 0)) ?.to_vec0_async::< i64 > (). await ?, 100);
        assert_eq!(t_cat.i((1, 6, 0)) ?.to_vec0_async::< i64 > (). await ?, 112);
        assert_eq!(t_cat.i((0, 6, 1)) ?.to_vec0_async::< i64 > (). await ?, 101);
        assert_eq!(t_cat.i((0, 7, 1)) ?.to_vec0_async::< i64 > (). await ?, 105);
        assert_eq!(t_cat.i((0, 12, 1)) ?.to_vec0_async::< i64 > (). await ?, 10013);
        assert_eq!(t_cat.i((1, 12, 3)) ?.to_vec0_async::< i64 > (). await ?, 10031);
    }
    Ok(())
}
async fn embeddings(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 2u32, 1u32], device)?;
    let t = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], device)?;
    let hs = t.embedding(&ids)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]
    );
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]
    );
    if device.is_dtype_available(DType::I64) {
        let hs = t.index_select(&ids.to_dtype(DType::I64)?, 0)?;
        assert_eq!(
            hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]
        );
    }
    Ok(())
}
async fn cmp(device: &Device) -> Result<()> {
    let t1 = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], device)?;
    let t2 = Tensor::new(&[[1f32, 0f32], [3f32, 3f32], [4f32, 7f32]], device)?;
    assert_eq!(
        t1.eq(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[0, 0], [0, 1], [1, 0]]
    );
    assert_eq!(
        t1.ne(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[1, 1], [1, 0], [0, 1]]
    );
    assert_eq!(
        t1.le(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[1, 0], [1, 1], [1, 1]]
    );
    assert_eq!(
        t1.lt(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[1, 0], [1, 0], [0, 1]]
    );
    assert_eq!(
        t1.gt(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[0, 1], [0, 0], [0, 0]]
    );
    assert_eq!(
        t1.ge(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[0, 1], [0, 1], [1, 0]]
    );
    Ok(())
}
async fn index_select(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 2u32, 1u32], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    for dtype in [DType::U8, DType::U32, DType::I64] {
        if device.is_dtype_available(dtype) {
            let ids = ids.to_dtype(dtype)?;
            let hs = t.index_select(&ids, 1)?;
            assert_eq!(
                hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 2.0, 1.0], [3.0, 5.0,
                4.0], [6.0, 8.0, 7.0], [9.0, 11.0, 10.0]]
            );
            let hs = t.index_select(&ids, 0)?;
            assert_eq!(
                hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [6.0, 7.0,
                8.0], [3.0, 4.0, 5.0]]
            );
        }
    }
    let ids = Tensor::new(&[0u32, 2u32, 1u32, 0u32, 2u32, 1u32], device)?;
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [3.0,
        4.0, 5.0], [0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [3.0, 4.0, 5.0],]
    );
    let ids = Tensor::new(&[1u32, 0u32, 1u32], device)?;
    let t = Tensor::arange(1f32, 5f32, device)?.reshape((2, 2))?;
    assert_eq!(t.to_vec2_async::< f32 > (). await ?, & [[1.0, 2.0], [3.0, 4.0]]);
    let hs = t.index_select(&ids, 1)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[2.0, 1.0, 2.0], [4.0, 3.0, 4.0]]
    );
    Ok(())
}
async fn index_add(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 1u32, 1u32], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    let init = Tensor::ones((4, 2), DType::F32, device)?;
    let hs = init.index_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[1.0, 4.0], [4.0, 10.0], [7.0, 16.0],
        [10.0, 22.0]],
    );
    let init = Tensor::zeros((4, 2), DType::F32, device)?;
    let ids = Tensor::new(&[1u32, 0u32, 0u32], device)?;
    let hs = init.index_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[3.0, 0.0], [9.0, 3.0], [15.0, 6.0],
        [21.0, 9.0]],
    );
    let init = Tensor::zeros((6, 3), DType::F32, device)?;
    let ids = Tensor::new(&[5u32, 0u32, 1u32, 0u32], device)?;
    let hs = init.index_add(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[12.0, 14.0, 16.0], [6.0, 7.0, 8.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]
    );
    Ok(())
}
async fn slice_scatter(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    let src = Tensor::arange(100f32, 106f32, device)?.reshape((2, 3))?;
    assert_eq!(
        t.slice_scatter0(& src, 0) ?.to_vec2_async::< f32 > (). await ?, & [[100.0,
        101.0, 102.0], [103.0, 104.0, 105.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    assert_eq!(
        t.slice_scatter0(& src, 1) ?.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0,
        2.0], [100.0, 101.0, 102.0], [103.0, 104.0, 105.0], [9.0, 10.0, 11.0]]
    );
    assert_eq!(
        t.slice_scatter0(& src, 2) ?.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0,
        2.0], [3.0, 4.0, 5.0], [100.0, 101.0, 102.0], [103.0, 104.0, 105.0],]
    );
    Ok(())
}
async fn scatter_add(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    let ids = Tensor::new(&[[0u32, 1, 2], [3, 4, 0], [3, 3, 1], [2, 0, 4]], device)?;
    let init = Tensor::ones((4, 5), DType::F32, device)?;
    let hs = init.scatter_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[1.0, 2.0, 3.0, 1.0, 1.0], [6.0, 1.0,
        1.0, 4.0, 5.0], [1.0, 9.0, 1.0, 14.0, 1.0], [11.0, 1.0, 10.0, 1.0, 12.0]]
    );
    let init = Tensor::ones((6, 3), DType::F32, device)?;
    let hs = init.scatter_add(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[1.0, 11.0, 6.0], [1.0, 2.0, 9.0],
        [10.0, 1.0, 3.0], [10.0, 8.0, 1.0], [1.0, 5.0, 12.0], [1.0, 1.0, 1.0]]
    );
    Ok(())
}
async fn gather(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[[0u32], [2u32], [1u32], [0u32]], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    let hs = t.gather(&ids, 1)?;
    assert_eq!(hs.to_vec2_async::< f32 > (). await ?, & [[0.0], [5.0], [7.0], [9.0]]);
    let ids = Tensor::new(
        &[[0u32, 0u32], [2u32, 0u32], [1u32, 1u32], [0u32, 2u32]],
        device,
    )?;
    let hs = t.gather(&ids, 1)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 0.0], [5.0, 3.0], [7.0, 7.0],
        [9.0, 11.0]]
    );
    let ids = Tensor::new(&[[0u32, 2u32, 0u32]], device)?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 7.0, 2.0]]);
    let ids = Tensor::new(&[[0u32, 2u32, 0u32], [0u32, 1u32, 1u32]], device)?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 7.0, 2.0], [0.0, 4.0, 5.0]]
    );
    let t = Tensor::new(
        &[
            [
                [108_f32, -47., 16., -56., -83., -130., 210.],
                [253., 95., 151., 228., -210., -123., -127.],
                [-9., -217., 2., -78., 163., 245., -204.],
                [-246., 79., -238., 88., -226., -184., 171.],
                [8., -48., -153., 234., -34., 166., -153.],
                [124., 0., -10., -61., -242., -15., -238.],
            ],
            [
                [12., -64., -199., 244., -240., 156., -128.],
                [173., -57., 4., -198., 233., -110., 238.],
                [95., 82., 0., 240., 53., -211., 209.],
                [-122., 167., -212., 227., -144., 61., 118.],
                [-63., -146., 200., 244., 168., -167., 116.],
                [-125., -147., 110., -253., -178., -250., -18.],
            ],
            [
                [57., 86., -50., 56., 92., 205., -78.],
                [-137., -156., -18., 248., -61., -239., 14.],
                [-248., -30., -50., -70., -251., 250., -83.],
                [-221., 67., 72., 59., -24., -154., 232.],
                [-144., -23., -74., 5., 93., 171., 205.],
                [46., -77., -38., -226., 246., 161., -17.],
            ],
            [
                [-153., -231., -236., 161., 126., 2., -22.],
                [-229., -41., 209., 164., 234., 160., 57.],
                [223., 254., -186., -162., -46., -160., -102.],
                [65., 30., 213., -253., 59., 224., -154.],
                [-82., -203., -177., 17., 31., -256., -246.],
                [176., -135., -65., 54., -56., 210., 76.],
            ],
            [
                [-10., -245., 168., 124., -14., -33., -178.],
                [25., -43., -39., 132., -89., 169., 179.],
                [187., -215., 32., -133., 87., -7., -168.],
                [-224., -215., -5., -230., -58., -162., 128.],
                [158., -137., -122., -100., -202., -83., 136.],
                [30., -185., -144., 250., 209., -40., 127.],
            ],
            [
                [-196., 108., -245., 122., 146., -228., 62.],
                [-1., -66., 160., 137., 13., -172., -21.],
                [244., 199., -164., 28., 119., -175., 198.],
                [-62., 253., -162., 195., -95., -230., -211.],
                [123., -72., -26., -107., -139., 64., 245.],
                [11., -126., -182., 108., -12., 184., -127.],
            ],
            [
                [-159., 126., 176., 161., 73., -111., -138.],
                [-187., 214., -217., -33., -223., -201., -212.],
                [-61., -120., -166., -172., -95., 53., 196.],
                [-33., 86., 134., -152., 154., -53., 74.],
                [186., -28., -154., -174., 141., -109., 217.],
                [82., 35., 252., 145., 181., 74., -87.],
            ],
        ],
        device,
    )?;
    let ids = Tensor::new(
        &[
            [
                [6_u32, 6, 4, 3, 4, 4, 6],
                [3, 3, 2, 4, 4, 4, 6],
                [3, 3, 0, 2, 4, 6, 4],
                [2, 5, 1, 2, 6, 6, 1],
                [2, 1, 6, 5, 3, 2, 3],
                [6, 1, 0, 1, 0, 2, 6],
            ],
            [
                [4, 6, 4, 3, 3, 3, 2],
                [4, 3, 2, 4, 4, 4, 6],
                [2, 3, 0, 2, 4, 6, 4],
                [6, 5, 1, 2, 6, 6, 1],
                [4, 1, 6, 5, 3, 2, 3],
                [1, 1, 0, 1, 0, 2, 6],
            ],
            [
                [3, 6, 4, 3, 3, 3, 2],
                [2, 3, 2, 4, 4, 4, 6],
                [4, 3, 0, 2, 4, 6, 4],
                [0, 5, 1, 2, 6, 6, 1],
                [6, 1, 6, 5, 3, 2, 3],
                [4, 1, 0, 1, 0, 2, 6],
            ],
            [
                [0, 6, 4, 3, 3, 3, 2],
                [5, 3, 2, 4, 4, 4, 6],
                [0, 3, 0, 2, 4, 6, 4],
                [3, 5, 1, 2, 6, 6, 1],
                [0, 1, 6, 5, 3, 2, 3],
                [3, 1, 0, 1, 0, 2, 6],
            ],
        ],
        device,
    )?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(
        hs.to_vec3_async::< f32 > (). await ?, & [[[- 159_f32, 126., 168., 161., - 14., -
        33., - 138.], [- 229., - 41., - 18., 132., - 89., 169., - 212.], [223., 254., 2.,
        - 70., 87., 53., - 168.], [- 221., 253., - 212., 59., 154., - 53., 118.], [-
        144., - 146., - 154., - 107., 31., 171., - 246.], [82., - 147., - 10., - 253., -
        242., 161., - 87.]], [[- 10., 126., 168., 161., 126., 2., - 78.], [25., - 41., -
        18., 132., - 89., 169., - 212.], [- 248., 254., 2., - 70., 87., 53., - 168.], [-
        33., 253., - 212., 59., 154., - 53., 118.], [158., - 146., - 154., - 107., 31.,
        171., - 246.], [- 125., - 147., - 10., - 253., - 242., 161., - 87.]], [[- 153.,
        126., 168., 161., 126., 2., - 78.], [- 137., - 41., - 18., 132., - 89., 169., -
        212.], [187., 254., 2., - 70., 87., 53., - 168.], [- 246., 253., - 212., 59.,
        154., - 53., 118.], [186., - 146., - 154., - 107., 31., 171., - 246.], [30., -
        147., - 10., - 253., - 242., 161., - 87.]], [[108., 126., 168., 161., 126., 2., -
        78.], [- 1., - 41., - 18., 132., - 89., 169., - 212.], [- 9., 254., 2., - 70.,
        87., 53., - 168.], [65., 253., - 212., 59., 154., - 53., 118.], [8., - 146., -
        154., - 107., 31., 171., - 246.], [176., - 147., - 10., - 253., - 242., 161., -
        87.]]]
    );
    let t = Tensor::new(
        &[
            [
                [-117_f32, -175., 69., -163.],
                [200., 242., -21., -67.],
                [179., 150., -126., -75.],
                [-118., 38., -138., -13.],
                [-221., 136., -185., 180.],
                [58., 182., -204., -149.],
            ],
            [
                [3., -148., -58., -154.],
                [-43., 45., -108., 4.],
                [-69., -249., -71., -21.],
                [80., 110., -152., -235.],
                [-88., 7., 92., -250.],
                [-186., 207., -242., 98.],
            ],
            [
                [238., 19., 64., -242.],
                [-150., -97., 218., 58.],
                [111., -233., 204., -212.],
                [-242., -232., 83., 42.],
                [153., 62., -251., 219.],
                [-117., 36., -119., 10.],
            ],
            [
                [215., 159., -169., -27.],
                [-83., 101., -88., 169.],
                [-205., 93., 225., -64.],
                [-162., 240., 214., 23.],
                [-112., 6., 21., 245.],
                [-38., 113., 93., 215.],
            ],
            [
                [91., -188., -148., 101.],
                [74., 203., -35., 55.],
                [-116., -130., -153., -96.],
                [58., 22., -45., -194.],
                [-221., -134., 73., 159.],
                [-203., -254., 31., 235.],
            ],
            [
                [105., -53., 61., 186.],
                [-195., 234., 75., -1.],
                [51., 139., 160., -108.],
                [-173., -167., 161., 19.],
                [83., -246., 156., -222.],
                [109., 39., -149., 137.],
            ],
        ],
        device,
    )?;
    let ids = Tensor::new(
        &[
            [[4_u32, 4, 4, 2]],
            [[0, 4, 4, 3]],
            [[1, 5, 3, 4]],
            [[0, 3, 3, 2]],
            [[1, 1, 5, 2]],
            [[1, 4, 5, 4]],
        ],
        device,
    )?;
    let hs = t.gather(&ids, 1)?;
    assert_eq!(
        hs.to_vec3_async::< f32 > (). await ?, & [[[- 221., 136., - 185., - 75.]], [[3.,
        7., 92., - 235.]], [[- 150., 36., 83., 219.]], [[215., 240., 214., - 64.]],
        [[74., 203., 31., - 96.]], [[- 195., - 246., - 149., - 222.]]]
    );
    let t = Tensor::new(
        &[
            [[-162_f32, 202.], [-126., -39.], [35., -65.], [1., 80.]],
            [[37., 248.], [-191., 89.], [117., -40.], [-217., 220.]],
        ],
        device,
    )?;
    let ids = Tensor::new(&[[[1_u32], [0], [1], [1]], [[0], [1], [0], [1]]], device)?;
    let hs = t.gather(&ids, 2)?;
    assert_eq!(
        hs.to_vec3_async::< f32 > (). await ?, & [[[202.], [- 126.], [- 65.], [80.]],
        [[37.], [89.], [117.], [220.]]]
    );
    let t = Tensor::new(
        &[
            [[-21_f32, -197.], [194., 122.]],
            [[255., -106.], [-191., 250.]],
            [[33., -117.], [43., 10.]],
            [[-130., 238.], [-217., -92.]],
        ],
        device,
    )?;
    let ids = Tensor::new(
        &[[[0_u32, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [0, 1]], [[1, 0], [1, 0]]],
        device,
    )?;
    let hs = t.gather(&ids, 2)?;
    assert_eq!(
        hs.to_vec3_async::< f32 > (). await ?, & [[[- 21., - 197.], [122., 194.]], [[-
        106., 255.], [- 191., 250.]], [[33., - 117.], [43., 10.]], [[238., - 130.], [-
        92., - 217.]]]
    );
    Ok(())
}
async fn broadcasting(device: &Device) -> Result<()> {
    let t1 = Tensor::arange(0f32, 24f32, device)?.reshape((4, 2, 3))?;
    let t2 = Tensor::new(&[100f32, 200f32], device)?;
    let s = t1.broadcast_add(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[100.0, 101.0, 102.0], [203.0, 204.0,
        205.0]], [[106.0, 107.0, 108.0], [209.0, 210.0, 211.0]], [[112.0, 113.0, 114.0],
        [215.0, 216.0, 217.0]], [[118.0, 119.0, 120.0], [221.0, 222.0, 223.0]]]
    );
    let s = t1.t()?.broadcast_add(&t2)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[100.0, 203.0], [101.0, 204.0], [102.0,
        205.0]], [[106.0, 209.0], [107.0, 210.0], [108.0, 211.0]], [[112.0, 215.0],
        [113.0, 216.0], [114.0, 217.0]], [[118.0, 221.0], [119.0, 222.0], [120.0,
        223.0]]]
    );
    let s = t1.broadcast_sub(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[- 100.0, - 99.0, - 98.0], [- 197.0, -
        196.0, - 195.0]], [[- 94.0, - 93.0, - 92.0], [- 191.0, - 190.0, - 189.0]], [[-
        88.0, - 87.0, - 86.0], [- 185.0, - 184.0, - 183.0]], [[- 82.0, - 81.0, - 80.0],
        [- 179.0, - 178.0, - 177.0]]]
    );
    let s = t1.t()?.broadcast_sub(&t2)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[- 100.0, - 197.0], [- 99.0, - 196.0],
        [- 98.0, - 195.0]], [[- 94.0, - 191.0], [- 93.0, - 190.0], [- 92.0, - 189.0]],
        [[- 88.0, - 185.0], [- 87.0, - 184.0], [- 86.0, - 183.0]], [[- 82.0, - 179.0], [-
        81.0, - 178.0], [- 80.0, - 177.0]]]
    );
    let t1 = t1.i(2..)?;
    let s = t1.broadcast_add(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[112.0, 113.0, 114.0], [215.0, 216.0,
        217.0]], [[118.0, 119.0, 120.0], [221.0, 222.0, 223.0]]]
    );
    let s = t1.t()?.broadcast_add(&t2)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[112.0, 215.0], [113.0, 216.0], [114.0,
        217.0]], [[118.0, 221.0], [119.0, 222.0], [120.0, 223.0]]]
    );
    let s = t1.broadcast_sub(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[- 88.0, - 87.0, - 86.0], [- 185.0, -
        184.0, - 183.0]], [[- 82.0, - 81.0, - 80.0], [- 179.0, - 178.0, - 177.0]]]
    );
    let s = t1.t()?.broadcast_sub(&t2)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[- 88.0, - 185.0], [- 87.0, - 184.0],
        [- 86.0, - 183.0]], [[- 82.0, - 179.0], [- 81.0, - 178.0], [- 80.0, - 177.0]]]
    );
    let t3 = Tensor::new(1f32, device)?.broadcast_div(&t2)?;
    let s = t1.broadcast_mul(&t2.reshape((2, 1))?)?;
    let s_div = t1.broadcast_div(&t3.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[1200.0, 1300.0, 1400.0], [3000.0,
        3200.0, 3400.0]], [[1800.0, 1900.0, 2000.0], [4200.0, 4400.0, 4600.0]]]
    );
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, s_div.to_vec3_async::< f32 > (). await ?,
    );
    let s = t1.t()?.broadcast_mul(&t2)?;
    let s_div = t1.t()?.broadcast_div(&t3)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[1200.0, 3000.0], [1300.0, 3200.0],
        [1400.0, 3400.0]], [[1800.0, 4200.0], [1900.0, 4400.0], [2000.0, 4600.0]]]
    );
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, s_div.to_vec3_async::< f32 > (). await ?,
    );
    Ok(())
}
async fn randn(device: &Device) -> Result<()> {
    let tensor = Tensor::randn(0f32, 1f32, (5, 3), device)?;
    assert_eq!(tensor.dims(), [5, 3]);
    let tensor2 = Tensor::randn(0f32, 1f32, (5, 3), device)?;
    assert_ne!(
        tensor.to_vec2_async::< f32 > (). await ?, tensor2.to_vec2_async::< f32 > ().
        await ?
    );
    let tensor = Tensor::rand(0f32, 1f32, (5, 3), device)?;
    assert_eq!(tensor.dims(), [5, 3]);
    let tensor2 = Tensor::rand(0f32, 1f32, (5, 3), device)?;
    assert_ne!(
        tensor.to_vec2_async::< f32 > (). await ?, tensor2.to_vec2_async::< f32 > ().
        await ?
    );
    const N: usize = 2;
    
            let mut v = Vec::new();
            for _ in 0..100 {
                let t = Tensor::randn(0f32, 1f32, N, device)?;
                let vec = t.to_vec1_async::<f32>().await?;
                v.push(vec);
            }
            
    assert!(
        (0..N).all(| i | v.windows(2).any(| pair | pair[0] [i] != pair[1] [i])),
        "There are deterministic values in the randn tensors"
    );
    
            let mut v = Vec::new();
            for _ in 0..100 {
                let t = Tensor::randn(0f32, 1f32, N, device)?;
                let vec = t.to_vec1_async::<f32>().await?;
                v.push(vec);
            }
            
    assert!(
        (0..N).all(| i | v.windows(2).any(| pair | pair[0] [i] != pair[1] [i])),
        "There are deterministic values in the rand tensors"
    );
    Ok(())
}
async fn where_cond(device: &Device) -> Result<()> {
    let cond = Tensor::new(&[0u32, 2u32, 1u32, 0, 0, 0, 35, 255, 53, 0, 29, 0], device)?
        .reshape((4, 3))?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    let t_f = Tensor::arange(12f32, 24f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t_f.to_vec2_async::< f32 > (). await ?, & [[12.0, 13.0, 14.0], [15.0, 16.0,
        17.0], [18.0, 19.0, 20.0], [21.0, 22.0, 23.0]]
    );
    for dtype in [DType::U8, DType::U32, DType::I64] {
        if device.is_dtype_available(dtype) {
            let cond = cond.to_dtype(dtype)?;
            let hs = cond.where_cond(&t, &t_f)?;
            assert_eq!(
                hs.to_vec2_async::< f32 > (). await ?, & [[12.0, 1.0, 2.0], [15.0, 16.0,
                17.0], [6.0, 7.0, 8.0], [21.0, 10.0, 23.0]]
            );
        }
    }
    Ok(())
}
async fn zero_dim(device: &Device) -> Result<()> {
    let t = Tensor::zeros((4, 0, 1), DType::F32, device)?;
    assert_eq!(t.dims3() ?, (4, 0, 1));
    let t2 = Tensor::zeros((4, 3, 1), DType::F32, device)?;
    let t_cat = Tensor::cat(&[&t, &t2], 1)?;
    assert_eq!(t_cat.dims3() ?, (4, 3, 1));
    let t_cat = Tensor::cat(&[&t, &t], 1)?;
    assert_eq!(t_cat.dims3() ?, (4, 0, 1));
    let t_unary = t.sqrt()?;
    assert_eq!(t_unary.dims3() ?, (4, 0, 1));
    let t_plus = (&t + 1.)?;
    assert_eq!(t_plus.dims3() ?, (4, 0, 1));
    let t_mm = t2.matmul(&t.t()?)?;
    assert_eq!(t_mm.dims3() ?, (4, 3, 0));
    let t_mm = t.matmul(&t2.t()?)?;
    assert_eq!(t_mm.dims3() ?, (4, 0, 3));
    let t_mm = t.t()?.matmul(&t)?;
    assert_eq!(t_mm.dims3() ?, (4, 1, 1));
    Ok(())
}
candle_wasm_tests::test_device!(zeros, zeros_cpu, zeros_gpu, zeros_metal, zeros_wgpu);
candle_wasm_tests::test_device!(ones, ones_cpu, ones_gpu, ones_metal, ones_wgpu);
candle_wasm_tests::test_device!(full, full_cpu, full_gpu, full_metal, full_wgpu);
candle_wasm_tests::test_device!(
    arange, arange_cpu, arange_gpu, arange_metal, arange_wgpu
);
candle_wasm_tests::test_device!(
    add_mul, add_mul_cpu, add_mul_gpu, add_mul_metal, add_mul_wgpu
);
candle_wasm_tests::test_device!(
    tensor_2d, tensor_2d_cpu, tensor_2d_gpu, tensor_2d_metal, tensor_2d_wgpu
);
candle_wasm_tests::test_device!(
    narrow, narrow_cpu, narrow_gpu, narrow_metal, narrow_wgpu
);
candle_wasm_tests::test_device!(
    broadcast, broadcast_cpu, broadcast_gpu, broadcast_metal, broadcast_wgpu
);
candle_wasm_tests::test_device!(slice_set, ss_cpu, ss_gpu, ss_metal, ss_wgpu);
candle_wasm_tests::test_device!(cat, cat_cpu, cat_gpu, cat_metal, cat_wgpu);
candle_wasm_tests::test_device!(sum, sum_cpu, sum_gpu, sum_metal, sum_wgpu);
candle_wasm_tests::test_device!(min, min_cpu, min_gpu, min_metal, min_wgpu);
candle_wasm_tests::test_device!(max, max_cpu, max_gpu, max_metal, max_wgpu);
candle_wasm_tests::test_device!(
    argmax, argmax_cpu, argmax_gpu, argmax_metal, argmax_wgpu
);
candle_wasm_tests::test_device!(
    argmin, argmin_cpu, argmin_gpu, argmin_metal, argmin_wgpu
);
candle_wasm_tests::test_device!(
    transpose, transpose_cpu, transpose_gpu, transpose_metal, transpose_wgpu
);
candle_wasm_tests::test_device!(
    unary_op, unary_op_cpu, unary_op_gpu, unary_op_metal, unary_op_wgpu
);
candle_wasm_tests::test_device!(
    binary_op, binary_op_cpu, binary_op_gpu, binary_op_metal, binary_op_wgpu
);
candle_wasm_tests::test_device!(
    embeddings, embeddings_cpu, embeddings_gpu, embeddings_metal, embeddings_wgpu
);
candle_wasm_tests::test_device!(cmp, cmp_cpu, cmp_gpu, cmp_metal, cmp_wgpu);
candle_wasm_tests::test_device!(
    broadcasting, broadcasting_cpu, broadcasting_gpu, broadcasting_metal,
    broadcasting_wgpu
);
candle_wasm_tests::test_device!(
    index_select, index_select_cpu, index_select_gpu, index_select_metal,
    index_select_wgpu
);
candle_wasm_tests::test_device!(
    where_cond, where_cond_cpu, where_cond_gpu, where_cond_metal, where_cond_wgpu
);
candle_wasm_tests::test_device!(
    index_add, index_add_cpu, index_add_gpu, index_add_metal, index_add_wgpu
);
candle_wasm_tests::test_device!(
    gather, gather_cpu, gather_gpu, gather_metal, gather_wgpu
);
candle_wasm_tests::test_device!(
    scatter_add, scatter_add_cpu, scatter_add_gpu, scatter_add_metal, scatter_add_wgpu
);
candle_wasm_tests::test_device!(
    slice_scatter, slice_scatter_cpu, slice_scatter_gpu, slice_scatter_metal,
    slice_scatter_wgpu
);
candle_wasm_tests::test_device!(randn, randn_cpu, randn_gpu, randn_metal, randn_wgpu);
candle_wasm_tests::test_device!(clamp, clamp_cpu, clamp_gpu, clamp_metal, clamp_wgpu);
candle_wasm_tests::test_device!(asort, asort_cpu, asort_gpu, asort_metal);
candle_wasm_tests::test_device!(var, var_cpu, var_gpu, var_metal, var_wgpu);
candle_wasm_tests::test_device!(
    zero_dim, zero_dim_cpu, zero_dim_gpu, zero_dim_metal, zero_dim_wgpu
);
#[test]
async fn randn_hasneg() -> Result<()> {
    let t = Tensor::randn(0f32, 1f32, 200, &Device::Cpu)?.to_vec1_async::<f32>().await?;
    if t.iter().all(|&v| v >= 0.) {
        candle::bail!("all values in tensors are non-negative")
    }
    Ok(())
}
#[test]
async fn pad_with_same() -> Result<()> {
    let t = Tensor::arange(1f32, 5f32, &Device::Cpu)?.reshape((2, 2))?;
    let t0 = t.pad_with_same(0, 1, 2)?;
    assert_eq!(
        t0.to_vec2_async::< f32 > (). await ?, [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0,
        4.0], [3.0, 4.0]]
    );
    let t1 = t.pad_with_same(1, 1, 2)?;
    assert_eq!(
        t1.to_vec2_async::< f32 > (). await ?, [[1.0, 1.0, 2.0, 2.0, 2.0], [3.0, 3.0,
        4.0, 4.0, 4.0]]
    );
    Ok(())
}
#[test]
async fn i64_abs() -> Result<()> {
    let t = Tensor::new(&[-42i64, 1337], &Device::Cpu)?;
    let t = t.abs()?;
    assert_eq!(t.to_vec1_async::< i64 > (). await ?, [42, 1337]);
    Ok(())
}
#[test]
async fn tril_triu_eye() -> Result<()> {
    let t = Tensor::tril2(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0,
        0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
    );
    let t = Tensor::triu2(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0,
        1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    );
    let t = Tensor::eye(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0,
        0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    );
    Ok(())
}
#[test]
async fn cumsum() -> Result<()> {
    let t = &[3f32, 1., 4., 1., 5.];
    let t = Tensor::new(t, &Device::Cpu)?;
    assert_eq!(t.cumsum(0) ?.to_vec1_async::< f32 > (). await ?, [3., 4., 8., 9., 14.]);
    let t = t.unsqueeze(1)?;
    assert_eq!(
        t.cumsum(0) ?.to_vec2_async::< f32 > (). await ?, [[3.0], [4.0], [8.0], [9.0],
        [14.0]]
    );
    assert_eq!(
        t.cumsum(1) ?.to_vec2_async::< f32 > (). await ?, [[3.0], [1.0], [4.0], [1.0],
        [5.0]]
    );
    let t = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let t = Tensor::new(t, &Device::Cpu)?;
    assert_eq!(
        t.cumsum(1) ?.to_vec2_async::< f32 > (). await ?, [[3.0, 4.0, 8.0, 9.0, 14.0],
        [2.0, 3.0, 10.0, 18.0, 20.0]],
    );
    assert_eq!(
        t.cumsum(0) ?.to_vec2_async::< f32 > (). await ?, [[3.0, 1.0, 4.0, 1.0, 5.0],
        [5.0, 2.0, 11.0, 9.0, 7.0]]
    );
    Ok(())
}
/// A helper function for floating point comparison. Both a and b must be 1D Tensor and contains the same amount of data.
/// Assertion passes if the difference of all pairs of a and b is smaller than epsilon.
async fn assert_close(a: &Tensor, b: &Tensor, epsilon: f64) -> Result<()> {
    let a_vec: Vec<f64> = a.to_vec1_async().await?;
    let b_vec: Vec<f64> = b.to_vec1_async().await?;
    assert_eq!(a_vec.len(), b_vec.len());
    for (a, b) in a_vec.iter().zip(b_vec.iter()) {
        assert!((a - b).abs() < epsilon);
    }
    Ok(())
}
#[test]
async fn log_sum_exp() -> Result<()> {
    let input = Tensor::new(
        &[
            [[1f64, 2., 3.], [4., 5., 6.]],
            [[-1000.0, -999.0, -1001.0], [1000.0, 999.0, 1001.0]],
        ],
        &Device::Cpu,
    )?;
    let output = input.log_sum_exp(D::Minus1)?;
    let expected = Tensor::new(
        &[[3.4076, 6.4076], [-998.5924, 1001.4076]],
        &Device::Cpu,
    )?;
    assert_eq!(output.dims(), expected.dims());
    assert_close(&output.flatten_all()?, &expected.flatten_all()?, 0.00001).await?;
    assert_eq!(
        input.log_sum_exp((0, 1)) ?.to_vec1_async::< f64 > (). await ?, [1000.0, 999.0,
        1001.0]
    );
    assert_eq!(
        input.log_sum_exp(()) ?.to_vec3_async::< f64 > (). await ?, input
        .to_vec3_async::< f64 > (). await ?
    );
    Ok(())
}
#[test]
async fn pow() -> Result<()> {
    let lhs = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let rhs = (&lhs - 2.)?;
    let res = lhs.pow(&rhs)?;
    assert_eq!(
        to_vec2_round_async(& res, 3). await ?, [[1.0, 1.0, 3.0], [16.0, 125.0, 1296.0]]
    );
    Ok(())
}
