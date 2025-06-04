use std::ptr;

use half::f16;
use utils::{quantize_row_iq3_xxs_impl, quantize_row_iq4_nl_impl};

use crate::{bail, Result};

mod utils;

use super::{k_quants::BlockQ8_0, BlockQ8K, GgmlDType, GgmlType, QK_K};

pub const QK4_NL: usize = 32;

pub(super) const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

const K_MASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

const K_SIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];

const IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ4xs {
    pub(crate) d: f16,
    pub(crate) scales_h: u16,
    pub(crate) scales_l: [u8; QK_K / 64],
    pub(crate) qs: [u8; QK_K / 2],
}

const _: () = assert!(
    std::mem::size_of::<BlockIQ4xs>()
        == std::mem::size_of::<f16>() + std::mem::size_of::<u16>() + QK_K / 64 + QK_K / 2,
    "wrong iq4_xs block size/padding"
);

impl GgmlType for BlockIQ4xs {
    const DTYPE: GgmlDType = GgmlDType::Iq4Xs;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;
    const SUPPORTS_I8MM: bool = false;

    fn to_float(xs: &[Self], mut ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("dequantize block iq4xs {k} is not divisible by {QK_K}");
        }

        let nb = k / QK_K;
        for block in xs.iter().take(nb) {
            let d = block.d.to_f32();
            let mut qs = &block.qs[..];

            for ib in 0..(QK_K / 32) {
                let ib_div_2 = ib / 2;
                let ib_mod_2 = ib % 2;

                let ls_low = (block.scales_l[ib_div_2] as i32 >> (4 * ib_mod_2 as i32)) & 0xF;
                let ls_high = ((block.scales_h as i32 >> (2 * ib as i32)) & 3) << 4;
                let ls = ls_low | ls_high;

                let dl = d * (ls as f32 - 32.);

                for j in 0..16 {
                    ys[j] = dl * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
                    ys[j + 16] = dl * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                }

                qs = &qs[16..];
                ys = &mut ys[32..];
            }
        }
        Ok(())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        let k = xs.len();
        if k % QK_K != 0 {
            bail!("Input length must be multiple of QK_K = {}", QK_K);
        }

        quantize_iq4_xs(xs, ys, 1, k, None)?;

        Ok(())
    }

    fn from_float_imatrix(
        xs: &[f32],
        ys: &mut [Self],
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let k = xs.len();
        if k % QK_K != 0 {
            bail!("Input length must be multiple of QK_K = {}", QK_K);
        }
        let nrow = xs.len() / n_per_row;

        quantize_iq4_xs(xs, ys, nrow, n_per_row, Some(imatrix_weights))?;

        Ok(())
    }

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        // #[cfg(target_feature = "avx")]
        // todo!();

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_iq4_xs_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if n % QK_K != 0 {
            bail!("n must be a multiple of QK_K");
        }
        let nb = n / QK_K;

        let mut sumf = 0.0f32;

        // Loop over each block
        for ibl in 0..nb {
            // x[ibl], y[ibl]
            let x = &xs[ibl];
            let y = &ys[ibl];

            // Convert x.d from fp16 to fp32, then multiply by y.d
            let d4d8 = x.d.to_f32() * y.d;

            // We'll track the "h" scales
            let mut h = x.scales_h;

            let mut qsi = 0; // index for x.qs
            let mut q8i = 0; // index for y.qs

            // so we step by 2 in Rust as well
            for ib2 in (0..(QK_K / 32)).step_by(2) {
                // Reproduce the logic of ls1, ls2
                let ls1 = (x.scales_l[ib2 / 2] & 0x0f) | (((h << 4) & 0x30) as u8);
                let ls2 = (x.scales_l[ib2 / 2] >> 4) | (((h << 2) & 0x30) as u8);
                // Then we shift h by 4 in the original code
                h >>= 4;

                // Convert ls1, ls2 to "scaled" floats
                let d1 = d4d8 * ((ls1 as i32) - 32) as f32;
                let d2 = d4d8 * ((ls2 as i32) - 32) as f32;

                // Two sets of 16 items each
                // sum of the first 16-lane block
                let mut sumi1 = 0;
                let mut sumi2 = 0;

                // The first pass
                for j in 0..16 {
                    // q8i + j   vs q8i + j + 16
                    // qs[qsi + j] & 0xf vs (qs[qsi + j] >> 4)
                    sumi1 += (y.qs[q8i + j] as i32)
                        * KVALUES_IQ4NL[(x.qs[qsi + j] & 0x0f) as usize] as i32;
                    sumi2 += (y.qs[q8i + j + 16] as i32)
                        * KVALUES_IQ4NL[((x.qs[qsi + j] >> 4) & 0x0f) as usize] as i32;
                }
                sumf += d1 * ((sumi1 + sumi2) as f32);

                qsi += 16;
                q8i += 32;

                // The second pass
                sumi1 = 0;
                sumi2 = 0;
                for j in 0..16 {
                    sumi1 += (y.qs[q8i + j] as i32)
                        * KVALUES_IQ4NL[(x.qs[qsi + j] & 0x0f) as usize] as i32;
                    sumi2 += (y.qs[q8i + j + 16] as i32)
                        * KVALUES_IQ4NL[((x.qs[qsi + j] >> 4) & 0x0f) as usize] as i32;
                }
                sumf += d2 * ((sumi1 + sumi2) as f32);

                qsi += 16;
                q8i += 32;
            }
        }

        Ok(sumf)
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

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ4nl {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_NL / 2],
}

const _: () = assert!(
    std::mem::size_of::<BlockIQ4nl>() == std::mem::size_of::<f16>() + QK4_NL / 2,
    "wrong iq4_nl block size/padding"
);

impl GgmlType for BlockIQ4nl {
    const DTYPE: GgmlDType = GgmlDType::Iq4Nl;
    const BLCK_SIZE: usize = QK4_NL;
    type VecDotType = BlockQ8_0;
    const SUPPORTS_I8MM: bool = false;

    fn to_float(xs: &[Self], mut ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK4_NL != 0 {
            crate::bail!("dequantize block iq4nl {k} is not divisible by {QK4_NL}");
        }

        let nb = k / QK4_NL;
        for block in xs.iter().take(nb) {
            let d = block.d.to_f32();
            let qs = &block.qs[..];

            for j in 0..(QK4_NL / 2) {
                ys[j] = d * KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as f32;
                ys[j + QK4_NL / 2] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
            }
            ys = &mut ys[QK4_NL..];
        }
        Ok(())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        let k = xs.len();
        if k % QK4_NL != 0 {
            bail!("Input length must be multiple of QK4_NL = {}", QK4_NL);
        }

        quantize_iq4_nl(xs, ys, 1, k, None)?;

        Ok(())
    }

    fn from_float_imatrix(
        xs: &[f32],
        ys: &mut [Self],
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let k = xs.len();
        if k % QK4_NL != 0 {
            bail!("Input length must be multiple of QK4_NL = {}", QK4_NL);
        }
        let nrow = xs.len() / n_per_row;

        quantize_iq4_nl(xs, ys, nrow, n_per_row, Some(imatrix_weights))?;

        Ok(())
    }

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        // #[cfg(target_feature = "avx")]
        // todo!();

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_iq4_nl_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        if n % QK4_NL != 0 {
            bail!("n must be a multiple of QK4_NL");
        }
        let nb = n / QK4_NL;

        let mut sumf = 0.0f32;

        // Loop over each block
        for ibl in 0..nb {
            let x = &xs[ibl];
            let y = &ys[ibl];

            let d = x.d.to_f32() * y.d.to_f32();

            let mut sumi1 = 0;
            let mut sumi2 = 0;

            for j in 0..QK4_NL / 2 {
                sumi1 += y.qs[j] as i32 * KVALUES_IQ4NL[(x.qs[j] & 0xf) as usize] as i32;
                sumi2 +=
                    y.qs[j + QK4_NL / 2] as i32 * KVALUES_IQ4NL[(x.qs[j] >> 4) as usize] as i32;
            }

            sumf += d * (sumi1 + sumi2) as f32;
        }

        Ok(sumf)
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

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockIQ3xxs {
    pub(crate) d: f16,
    pub(crate) qs: [u8; 3 * QK_K / 8],
}

const _: () = assert!(
    std::mem::size_of::<BlockIQ3xxs>() == std::mem::size_of::<f16>() + 3 * QK_K / 8,
    "wrong iq3_xxs block size/padding"
);

impl GgmlType for BlockIQ3xxs {
    const DTYPE: GgmlDType = GgmlDType::Iq3Xxs;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;
    const SUPPORTS_I8MM: bool = false;

    fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
        let k = ys.len();
        if k % QK_K != 0 {
            crate::bail!("dequantize block iq3xxs {k} is not divisible by {QK_K}");
        }
        let nb = k / QK_K;

        let mut aux32: u32;

        let mut y = ys.as_mut_ptr();

        unsafe {
            // Process each block.
            for i in 0..nb {
                // Access the i'th block.
                let block = &xs[i];
                // Convert FP16 to f32.
                let d = block.d.to_f32();
                // Get a pointer to the beginning of qs.
                let mut qs = block.qs.as_ptr();
                // scales_and_signs is located QK_K/4 bytes after qs.
                let scales_and_signs = qs.add((QK_K / 4) as usize);

                // Loop over each 32-byte subblock.
                for ib32 in 0..(QK_K / 32) {
                    // Copy 4 bytes from scales_and_signs + 4*ib32 into aux32.
                    aux32 = ptr::read_unaligned(
                        scales_and_signs.add((4 * ib32) as usize) as *const u32
                    );
                    // Compute db = d * (0.5 + (aux32 >> 28)) * 0.5
                    let db = d * (0.5 + ((aux32 >> 28) as f32)) * 0.5;

                    // Process 4 groups per 32-byte subblock.
                    for l in 0..4 {
                        let shift = 7 * l;
                        let idx = ((aux32 >> shift) & 127) as usize;
                        // Get the corresponding 'signs' value.
                        let signs = K_SIGNS_IQ2XS[idx];

                        // Get pointers to grid1 and grid2.
                        // qs[2*l+0] and qs[2*l+1] are used as offsets into IQ3XXS_GRID.
                        let idx1 = *qs.add((2 * l + 0) as usize) as usize;
                        let grid1 = IQ3XXS_GRID.as_ptr().add(idx1) as *const u8;
                        let idx2 = *qs.add((2 * l + 1) as usize) as usize;
                        let grid2 = IQ3XXS_GRID.as_ptr().add(idx2) as *const u8;

                        // For each of 4 values in grid1 and grid2.
                        for j in 0..4 {
                            let mask1 = K_MASK_IQ2XS[(j + 0) as usize];
                            let mask2 = K_MASK_IQ2XS[(j + 4) as usize];
                            let sign1 = if (signs & mask1) != 0 { -1.0 } else { 1.0 };
                            let sign2 = if (signs & mask2) != 0 { -1.0 } else { 1.0 };

                            let grid1_val = *grid1.add(j as usize) as f32;
                            let grid2_val = *grid2.add(j as usize) as f32;

                            // Write the dequantized values.
                            ptr::write(y.add(j + 0), db * grid1_val * sign1);
                            ptr::write(y.add(j + 4), db * grid2_val * sign2);
                        }
                        // Advance y by 8 floats.
                        y = y.add(8);
                    }
                    // Advance qs by 8 bytes.
                    qs = qs.add(8);
                }
            }
        }
        Ok(())
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
        let k = xs.len();
        if k % QK_K != 0 {
            bail!("Input length must be multiple of QK_K = {}", QK_K);
        }

        unsafe { quantize_iq3_xxs(xs, ys, 1, k, None)? };

        Ok(())
    }

    fn from_float_imatrix(
        xs: &[f32],
        ys: &mut [Self],
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let k = xs.len();
        if k % QK_K != 0 {
            bail!("Input length must be multiple of QK_K = {}", QK_K);
        }
        let nrow = xs.len() / n_per_row;

        unsafe { quantize_iq3_xxs(xs, ys, nrow, n_per_row, Some(imatrix_weights))? };

        Ok(())
    }

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
        todo!()
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

fn quantize_iq4_xs(
    src: &[f32],
    ys: &mut [BlockIQ4xs],
    nrow: usize,
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Result<()> {
    if n_per_row % QK_K != 0 {
        bail!("n_per_row must be multiple of QK_K = {}", QK_K);
    }

    let nblock = n_per_row / QK_K;
    // We expect exactly nrow * nblock blocks in `ys`.
    if ys.len() != nrow * nblock {
        bail!(
            "Output buffer size mismatch: want {} blocks, got {}",
            nrow * nblock,
            ys.len()
        );
    }

    let mut lbuf = vec![0u8; QK_K]; // L[QK_K]
    let mut weight = vec![0f32; 32]; // weight[32] (the block_size is 32)
    let mut scales = vec![0f32; QK_K / 32]; // scales[QK_K/32], e.g. 256/32=8

    let mut src_offset = 0;
    let mut dst_offset = 0;

    for _row in 0..nrow {
        // Each row has `nblock` blocks:
        for ibl in 0..nblock {
            let block = &mut ys[dst_offset + ibl];

            let qw = quant_weights.map(|qw_all| {
                let start = QK_K * ibl;
                &qw_all[start..start + QK_K]
            });

            quantize_row_iq4_nl_impl(
                /* super_block_size = */ QK_K,
                /* block_size       = */ 32,
                /* x                = */
                &src[src_offset + QK_K * ibl..src_offset + QK_K * (ibl + 1)],
                /* dh               = */ &mut block.d,
                /* q4               = */ &mut block.qs,
                /* scales_h         = */ Some(&mut block.scales_h),
                /* scales_l         = */ Some(&mut block.scales_l),
                /* scales           = */ &mut scales,
                /* weight           = */ &mut weight,
                /* L                = */ &mut lbuf,
                /* values           = */ &KVALUES_IQ4NL,
                /* quant_weights    = */ qw,
                /* ntry             = */ 7,
            );
        }
        src_offset += n_per_row;
        dst_offset += nblock;
    }

    Ok(())
}

fn quantize_iq4_nl(
    src: &[f32],
    ys: &mut [BlockIQ4nl],
    nrow: usize,
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Result<()> {
    if n_per_row % QK4_NL != 0 {
        bail!("n_per_row must be multiple of QK4_NL = {}", QK4_NL);
    }

    let nblock = n_per_row / QK4_NL;
    // We expect exactly nrow * nblock blocks in `ys`.
    if ys.len() != nrow * nblock {
        bail!(
            "Output buffer size mismatch: want {} blocks, got {}",
            nrow * nblock,
            ys.len()
        );
    }

    let mut lbuf = vec![0u8; QK4_NL]; // L[QK4_NL]
    let mut weight = vec![0f32; QK4_NL]; // weight[QK4_NL]
    let mut scales = vec![0f32]; // scales[1]

    let mut src_offset = 0;
    let mut dst_offset = 0;

    for _row in 0..nrow {
        // Each row has `nblock` blocks:
        for ibl in 0..nblock {
            let block = &mut ys[dst_offset + ibl];

            let qw = quant_weights.map(|qw_all| {
                let start = QK4_NL * ibl;
                &qw_all[start..start + QK4_NL]
            });

            quantize_row_iq4_nl_impl(
                /* super_block_size = */ QK4_NL,
                /* block_size       = */ 32,
                /* x                = */
                &src[src_offset + QK4_NL * ibl..src_offset + QK4_NL * (ibl + 1)],
                /* dh               = */ &mut block.d,
                /* q4               = */ &mut block.qs,
                /* scales_h         = */ None,
                /* scales_l         = */ None,
                /* scales           = */ &mut scales,
                /* weight           = */ &mut weight,
                /* L                = */ &mut lbuf,
                /* values           = */ &KVALUES_IQ4NL,
                /* quant_weights    = */ qw,
                /* ntry             = */ 7,
            );
        }
        src_offset += n_per_row;
        dst_offset += nblock;
    }

    Ok(())
}

unsafe fn quantize_iq3_xxs(
    src: &[f32],
    ys: &mut [BlockIQ3xxs],
    nrow: usize,
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Result<()> {
    // Assert that n_per_row is a multiple of QK_K.
    assert!(
        n_per_row % QK_K == 0,
        "n_per_row must be a multiple of QK_K"
    );
    let nblock = n_per_row / QK_K;

    let mut src_ptr = src.as_ptr();
    let mut dst_ptr = ys.as_mut_ptr();

    for _row in 0..nrow {
        quantize_row_iq3_xxs_impl(256, src_ptr, dst_ptr, n_per_row as i64, quant_weights);

        src_ptr = src_ptr.add(n_per_row);
        dst_ptr = dst_ptr.add(nblock);
    }
    Ok(())
}
