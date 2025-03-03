use super::GgmlDType;
use crate::Result;
use half::{bf16, f16};
use rayon::prelude::*;

pub trait GgmlType: Sized + Clone + Send + Sync {
    const DTYPE: GgmlDType;
    const BLCK_SIZE: usize;
    type VecDotType: GgmlType;
    const SUPPORTS_I8MM: bool;

    // This is only safe for types that include immediate values such as float/int/...
    fn zeros() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()>;
    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()>;
    fn from_float_imatrix(
        _xs: &[f32],
        _ys: &mut [Self],
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        crate::bail!(
            "`from_float_imatrix` is unimplemented for {:?}",
            Self::DTYPE
        );
    }

    /// Dot product used as a building block for quantized mat-mul.
    /// n is the number of elements to be considered.
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32>;

    /// Generic implementation of the dot product without simd optimizations.
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32>;

    /// Multiply 2 rows by 2 columns and return a 2x2 matrix
    /// based on aarch64 NEON i8mm instructions
    #[cfg(feature = "arm-nightly-feat")]
    fn matmul_i8mm(
        n: usize,
        xs_0: &[Self],
        xs_1: &[Self],
        ys_0: &[Self::VecDotType],
        ys_1: &[Self::VecDotType],
    ) -> Result<[f32; 4]>;
}

impl GgmlType for f32 {
    const DTYPE: GgmlDType = GgmlDType::F32;
    const BLCK_SIZE: usize = 1;
    type VecDotType = f32;
    const SUPPORTS_I8MM: bool = false;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if xs.len() < n {
            crate::bail!("size mismatch {} < {n}", xs.len())
        }
        if ys.len() < n {
            crate::bail!("size mismatch {} < {n}", ys.len())
        }
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f32(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        Ok(res)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        ys.copy_from_slice(xs);
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        ys.copy_from_slice(xs);
        Ok(())
    }

    #[allow(unused)]
    #[cfg(feature = "arm-nightly-feat")]
    fn matmul_i8mm(
        n: usize,
        xs_0: &[Self],
        xs_1: &[Self],
        ys_0: &[Self::VecDotType],
        ys_1: &[Self::VecDotType],
    ) -> Result<[f32; 4]> {
        crate::bail!("Unsupported block type for i8mm");
    }
}

impl GgmlType for f16 {
    const DTYPE: GgmlDType = GgmlDType::F16;
    const BLCK_SIZE: usize = 1;
    type VecDotType = f16;
    const SUPPORTS_I8MM: bool = false;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if xs.len() < n {
            crate::bail!("size mismatch {} < {n}", xs.len())
        }
        if ys.len() < n {
            crate::bail!("size mismatch {} < {n}", ys.len())
        }
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        Ok(res)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = f16::from_f32(*x)
        }
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = x.to_f32()
        }
        Ok(())
    }

    #[allow(unused)]
    #[cfg(feature = "arm-nightly-feat")]
    fn matmul_i8mm(
        n: usize,
        xs_0: &[Self],
        xs_1: &[Self],
        ys_0: &[Self::VecDotType],
        ys_1: &[Self::VecDotType],
    ) -> Result<[f32; 4]> {
        crate::bail!("Unsupported block type for i8mm");
    }
}

impl GgmlType for bf16 {
    const DTYPE: GgmlDType = GgmlDType::BF16;
    const BLCK_SIZE: usize = 1;
    type VecDotType = bf16;
    const SUPPORTS_I8MM: bool = false;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if xs.len() < n {
            crate::bail!("size mismatch {} < {n}", xs.len())
        }
        if ys.len() < n {
            crate::bail!("size mismatch {} < {n}", ys.len())
        }
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_bf16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        Ok(res)
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = bf16::from_f32(*x)
        }
        Ok(())
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        if xs.len() != ys.len() {
            crate::bail!("size mismatch {} {}", xs.len(), ys.len());
        }
        // TODO: vectorize
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            *y = x.to_f32()
        }
        Ok(())
    }

    #[allow(unused)]
    #[cfg(feature = "arm-nightly-feat")]
    fn matmul_i8mm(
        n: usize,
        xs_0: &[Self],
        xs_1: &[Self],
        ys_0: &[Self::VecDotType],
        ys_1: &[Self::VecDotType],
    ) -> Result<[f32; 4]> {
        crate::bail!("Unsupported block type for i8mm");
    }
}

fn matmul_ref<T: GgmlType>(
    mkn: (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) -> Result<()> {
    let (m, k, n) = mkn;
    if m * k != lhs.len() {
        crate::bail!("unexpected lhs length {} {mkn:?}", lhs.len());
    }

    let k_in_lhs_blocks = k.div_ceil(T::BLCK_SIZE);
    let k_in_rhs_blocks = k.div_ceil(T::VecDotType::BLCK_SIZE);
    // TODO: Do not make this copy if the DotType is f32.
    // TODO: Pre-allocate this.
    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_lhs_blocks];
    for row_idx in 0..m {
        let lhs_b = &mut lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let lhs = &lhs[row_idx * k..(row_idx + 1) * k];
        T::VecDotType::from_float(lhs, lhs_b)?
    }
    let lhs_b = lhs_b.as_slice();

    for row_idx in 0..m {
        let lhs_row = &lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let dst_row = &mut dst[row_idx * n..(row_idx + 1) * n];

        let result: Result<Vec<_>> = dst_row
            .into_par_iter()
            .enumerate()
            .with_min_len(128)
            .with_max_len(512)
            .map(|(col_idx, dst)| {
                let rhs_col = &rhs_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
                T::vec_dot(k, rhs_col, lhs_row).map(|value| *dst = value)
            })
            .collect();

        result?;
    }
    Ok(())
}

// https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L10605
#[cfg(not(all(feature = "arm-nightly-feat", target_feature = "i8mm")))]
pub fn matmul<T: GgmlType>(
    mkn: (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) -> Result<()> {
    matmul_ref(mkn, lhs, rhs_t, dst)
}

#[cfg(all(feature = "arm-nightly-feat", target_feature = "i8mm"))]
pub fn matmul<T: GgmlType>(
    mkn: (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) -> Result<()> {
    if !T::SUPPORTS_I8MM {
        return matmul_ref(mkn, lhs, rhs_t, dst);
    }

    let (m, k, n) = mkn;
    if m * k != lhs.len() {
        crate::bail!("unexpected lhs length {} {mkn:?}", lhs.len());
    }

    let k_in_lhs_blocks = (k + T::BLCK_SIZE - 1) / T::BLCK_SIZE;
    let k_in_rhs_blocks = (k + T::VecDotType::BLCK_SIZE - 1) / T::VecDotType::BLCK_SIZE;
    // TODO: Do not make this copy if the DotType is f32.
    // TODO: Pre-allocate this.
    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_lhs_blocks];
    for row_idx in 0..m {
        let lhs_b = &mut lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let lhs = &lhs[row_idx * k..(row_idx + 1) * k];
        T::VecDotType::from_float(lhs, lhs_b)?
    }
    let lhs_b = lhs_b.as_slice();

    let m_even = m % 2 == 0;
    let m_limit = if m_even { m } else { m - 1 };
    let n_even = n % 2 == 0;
    let n_limit = if n_even { n } else { n - 1 };

    for row_idx in (0..m_limit).step_by(2) {
        let lhs_row_0 = &lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let lhs_row_1 = &lhs_b[(row_idx + 1) * k_in_lhs_blocks..(row_idx + 2) * k_in_lhs_blocks];

        let dst_2_rows = &mut dst[row_idx * n..(row_idx + 2) * n];
        let (dst_row_0, dst_row_1) = dst_2_rows.split_at_mut(dst_2_rows.len() / 2);

        let dst_row_0_n = &mut dst_row_0[0..n_limit];
        let dst_row_1_n = &mut dst_row_1[0..n_limit];

        let _result: Vec<_> = dst_row_0_n
            .par_chunks_mut(2)
            .zip(dst_row_1_n.par_chunks_mut(2))
            .enumerate()
            .with_min_len(128)
            .with_max_len(512)
            .map(|(half_of_col_idx, (dst_0, dst_1))| {
                let col_idx = half_of_col_idx * 2; // each step has 2 columns
                let rhs_col_0 = &rhs_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
                let rhs_col_1 =
                    &rhs_t[(col_idx + 1) * k_in_rhs_blocks..(col_idx + 2) * k_in_rhs_blocks];

                T::matmul_i8mm(k, rhs_col_0, rhs_col_1, lhs_row_0, lhs_row_1).map(|mm| {
                    dst_0[0] = mm[0];
                    dst_0[1] = mm[1];
                    dst_1[0] = mm[2];
                    dst_1[1] = mm[3];
                })
            })
            .collect();
        if !n_even {
            let col_idx = n - 1;
            let rhs_col = &rhs_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
            dst_row_0[col_idx] = T::vec_dot(k, rhs_col, lhs_row_0).unwrap();
            dst_row_1[col_idx] = T::vec_dot(k, rhs_col, lhs_row_1).unwrap();
        }
    }
    if !m_even {
        let row_idx = m - 1;
        let lhs_row = &lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];

        let dst_row = &mut dst[row_idx * n..(row_idx + 1) * n];
        let result: Result<Vec<_>> = dst_row
            .into_par_iter()
            .enumerate()
            .with_min_len(128)
            .with_max_len(512)
            .map(|(col_idx, dst)| {
                let rhs_col = &rhs_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
                T::vec_dot(k, rhs_col, lhs_row).map(|value| *dst = value)
            })
            .collect();

        result?;
    }
    Ok(())
}
