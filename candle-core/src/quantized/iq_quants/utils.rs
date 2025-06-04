use std::{ffi::c_void, mem, ptr};

use half::f16;

use crate::quantized::{k_quants::QK_K, GgmlType};

use super::BlockIQ3xxs;

const GROUP_MAX_EPS: f32 = 1e-15;
const GROUP_MAX_EPS_IQ3_XXS: f32 = 1e-8;

fn nearest_int(x: f32) -> i32 {
    x.round() as i32
}

#[allow(clippy::too_many_arguments)]
pub(super) fn quantize_row_iq4_nl_impl(
    super_block_size: usize,
    block_size: usize,
    x: &[f32],
    dh: &mut f16,
    q4: &mut [u8],
    scales_h: Option<&mut u16>,
    scales_l: Option<&mut [u8]>,
    scales: &mut [f32],
    weight: &mut [f32],
    lbuf: &mut [u8],
    values: &[i8],
    quant_weights: Option<&[f32]>,
    ntry: i32,
) {
    // For safety, confirm the slices have correct lengths:
    let sb_div_2 = super_block_size / 2;
    let sb_div_32 = super_block_size / 32;
    let sb_div_64 = super_block_size / 64;
    assert_eq!(q4.len(), sb_div_2);
    assert_eq!(scales.len(), sb_div_32);
    assert_eq!(lbuf.len(), super_block_size);
    assert_eq!(weight.len(), block_size);

    // 1. compute sigma2
    let mut sigma2 = 0f32;
    for x in x.iter().take(super_block_size) {
        sigma2 += x * x;
    }
    sigma2 *= 2.0 / (super_block_size as f32);

    // 2. zero out q4, set dh to 0
    for qi in q4.iter_mut() {
        *qi = 0;
    }
    *dh = f16::from_f32(0.0);

    // Track the max absolute scale across sub-blocks
    let mut max_scale = 0.0_f32;
    let mut amax_scale = 0.0_f32;

    // For each 32-float block within the 256-float super-block:
    let nblocks = super_block_size / block_size;

    for ib in 0..nblocks {
        let xb = &x[ib * block_size..ib * block_size + block_size];
        let lb = &mut lbuf[ib * block_size..ib * block_size + block_size];

        // If we have external `quant_weights`, fill `weight[j] = quant_weights[j]*sqrt(...)`,
        // else `weight[j] = xb[j]*xb[j]`
        if let Some(qw) = quant_weights {
            let qw_block = &qw[ib * block_size..ib * block_size + block_size];
            for j in 0..block_size {
                let val = xb[j];
                weight[j] = qw_block[j] * (sigma2 + val * val).sqrt();
            }
        } else {
            for j in 0..block_size {
                let val = xb[j];
                weight[j] = val * val;
            }
        }

        // 3. find amax (largest absolute value in block)
        let mut amax = 0.0_f32;
        let mut max_v = 0.0_f32;
        for &xx in xb {
            let ax = xx.abs();
            if ax > amax {
                amax = ax;
                max_v = xx;
            }
        }

        // If amax is extremely small, scale = 0
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }

        // 4. initial guess for d
        let sign_factor = if ntry > 0 { -1.0 } else { 1.0 };
        let mut d = sign_factor * max_v / (values[0] as f32);
        let id = 1.0 / d;

        // 5. compute an initial sumqx, sumq2
        let mut sumqx = 0.0_f32;
        let mut sumq2 = 0.0_f32;
        for j in 0..block_size {
            let val = xb[j];
            let al = id * val;
            let l = best_index_int8(values, al);
            lb[j] = l as u8;

            let q = values[l] as f32;
            let w = weight[j];
            sumqx += w * q * val;
            sumq2 += w * q * q;
        }
        d = sumqx / sumq2;
        let mut best = d * sumqx;

        // 6. do extra tries around that initial guess
        for itry in -ntry..=ntry {
            let test_id = (itry as f32 + values[0] as f32) / max_v;
            let mut tmp_sumqx = 0.0_f32;
            let mut tmp_sumq2 = 0.0_f32;
            for j in 0..block_size {
                let val = xb[j];
                let al = test_id * val;
                let l = best_index_int8(values, al);
                let q = values[l] as f32;
                let w = weight[j];
                tmp_sumqx += w * q * val;
                tmp_sumq2 += w * q * q;
            }
            if tmp_sumq2 > 0.0 {
                let maybe_d = tmp_sumqx / tmp_sumq2;
                let maybe_best = maybe_d * tmp_sumqx;
                if maybe_best > best {
                    best = maybe_best;
                    d = maybe_d;
                }
            }
        }

        // 7. record the chosen scale
        scales[ib] = d;
        let abs_d = d.abs();
        if abs_d > amax_scale {
            amax_scale = abs_d;
            max_scale = d;
        }
    }

    // 8. If we have more than one 32-float block in the super-block:
    if nblocks > 1 {
        let scales_h = scales_h.expect("Expected scales_h, nblocks > 1");
        let scales_l = scales_l.expect("Expected scales_l, nblocks > 1");
        assert_eq!(scales_l.len(), sb_div_64);

        // zero scales_h, because we store 2 bits per block in it
        // for nblocks=8, we store them in a single 16-bit value
        *scales_h = 0;
        for sl in scales_l.iter_mut() {
            *sl = 0;
        }

        let d = -max_scale / 32.0;
        *dh = f16::from_f32(d);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        for ib in 0..nblocks {
            // l = nearest_int(id * scales[ib]), clamp to [-32..31]
            let mut l = (id * scales[ib]).round() as i32;
            l = l.clamp(-32, 31);

            // refine block
            let dl = d * (l as f32);
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };

            let xb = &x[ib * block_size..ib * block_size + block_size];
            let lb = &mut lbuf[ib * block_size..ib * block_size + block_size];
            for j in 0..block_size {
                let val = xb[j];
                lb[j] = best_index_int8(values, idl * val) as u8;
            }

            // store l in 4 bits + 4 bits
            let l_offset = (l + 32) as u8; // now in [0..64)
            let l_low = l_offset & 0x0f;
            let l_high = l_offset >> 4;

            // scales_l[ib/2] uses the nibble for this block
            if ib % 2 == 0 {
                scales_l[ib / 2] = l_low;
            } else {
                scales_l[ib / 2] |= l_low << 4;
            }
            // scales_h for each block (2 bits per block) => stored in a 16-bit
            // scaled_h[ib/8] with (l_high << (2*(ib%8)))
            let shift = 2 * (ib % 8);
            *scales_h |= (l_high as u16) << shift;
        }
    } else {
        // single 32-float block => just store d
        *dh = f16::from_f32(scales[0]);
        if ntry > 0 {
            let id = if scales[0] != 0.0 {
                1.0 / scales[0]
            } else {
                0.0
            };
            for j in 0..super_block_size {
                lbuf[j] = best_index_int8(values, id * x[j]) as u8;
            }
        }
    }

    // 9. Finally, pack all 4-bit values from L into q4
    //    q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4)
    for i in 0..(super_block_size / 32) {
        for j in 0..16 {
            let lo = lbuf[32 * i + j] & 0x0f;
            let hi = (lbuf[32 * i + 16 + j] & 0x0f) << 4;
            q4[16 * i + j] = lo | hi;
        }
    }
}

/// Finds the best index i in [0..values.len()) such that
/// `values[i]` is closest to `x`. The array `values` is strictly
/// ascending/
fn best_index_int8(values: &[i8], x: f32) -> usize {
    // Quick boundary checks
    if x <= values[0] as f32 {
        return 0;
    }
    let n = values.len();
    let last = (n - 1).max(0);
    if x >= values[last] as f32 {
        return last;
    }

    // Binary search
    let mut ml = 0;
    let mut mu = last;
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < values[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }

    // Return whichever is closer among values[mu-1], values[mu]
    // But watch out if mu == 0 or mu == n-1 ...
    // (the boundary checks above should keep mu>0)
    let dist_left = (x - values[ml] as f32).abs();
    let dist_right = (values[mu] as f32 - x).abs();
    if dist_left <= dist_right {
        ml
    } else {
        mu
    }
}

/// Global state analogous to the C code’s iq3_data.
/// (Using unsafe mutable statics; in production code consider using a safe wrapper.)
#[derive(Debug)]
struct Iq3Entry {
    grid: Option<Vec<u32>>,
    map: Option<Vec<i32>>,
    neighbours: Option<Vec<u16>>,
}

static mut IQ3_DATA: [Iq3Entry; 2] = [
    Iq3Entry {
        grid: None,
        map: None,
        neighbours: None,
    },
    Iq3Entry {
        grid: None,
        map: None,
        neighbours: None,
    },
];

static KGRID_256: [u16; 256] = [
    0, 2, 4, 9, 11, 15, 16, 18, 25, 34, 59, 61, 65, 67, 72, 74, 81, 85, 88, 90, 97, 108, 120, 128,
    130, 132, 137, 144, 146, 153, 155, 159, 169, 175, 189, 193, 199, 200, 202, 213, 248, 267, 287,
    292, 303, 315, 317, 321, 327, 346, 362, 413, 436, 456, 460, 462, 483, 497, 513, 515, 520, 522,
    529, 531, 536, 538, 540, 551, 552, 576, 578, 585, 592, 594, 641, 643, 648, 650, 657, 664, 698,
    704, 706, 720, 729, 742, 758, 769, 773, 808, 848, 852, 870, 889, 901, 978, 992, 1024, 1026,
    1033, 1035, 1040, 1042, 1046, 1049, 1058, 1089, 1091, 1093, 1096, 1098, 1105, 1112, 1139, 1143,
    1144, 1152, 1154, 1161, 1167, 1168, 1170, 1183, 1184, 1197, 1217, 1224, 1228, 1272, 1276, 1309,
    1323, 1347, 1367, 1377, 1404, 1473, 1475, 1486, 1509, 1537, 1544, 1546, 1553, 1555, 1576, 1589,
    1594, 1600, 1602, 1616, 1625, 1636, 1638, 1665, 1667, 1672, 1685, 1706, 1722, 1737, 1755, 1816,
    1831, 1850, 1856, 1862, 1874, 1901, 1932, 1950, 1971, 2011, 2032, 2052, 2063, 2077, 2079, 2091,
    2095, 2172, 2192, 2207, 2208, 2224, 2230, 2247, 2277, 2308, 2345, 2356, 2389, 2403, 2424, 2501,
    2504, 2506, 2520, 2570, 2593, 2616, 2624, 2630, 2646, 2669, 2700, 2714, 2746, 2754, 2795, 2824,
    2835, 2839, 2874, 2882, 2905, 2984, 3028, 3042, 3092, 3108, 3110, 3124, 3153, 3185, 3215, 3252,
    3288, 3294, 3364, 3397, 3434, 3483, 3523, 3537, 3587, 3589, 3591, 3592, 3610, 3626, 3670, 3680,
    3722, 3749, 3754, 3776, 3789, 3803, 3824, 3857, 3873, 3904, 3906, 3924, 3992,
];

static KGRID_512: [u16; 512] = [
    0, 1, 2, 5, 7, 8, 9, 10, 12, 14, 16, 17, 21, 27, 32, 34, 37, 39, 41, 43, 48, 50, 57, 60, 63,
    64, 65, 66, 68, 72, 73, 77, 80, 83, 87, 89, 93, 100, 113, 117, 122, 128, 129, 133, 135, 136,
    139, 142, 145, 149, 152, 156, 162, 165, 167, 169, 171, 184, 187, 195, 201, 205, 208, 210, 217,
    219, 222, 228, 232, 234, 247, 249, 253, 256, 267, 271, 273, 276, 282, 288, 291, 297, 312, 322,
    324, 336, 338, 342, 347, 353, 357, 359, 374, 379, 390, 393, 395, 409, 426, 441, 448, 450, 452,
    464, 466, 470, 475, 488, 492, 512, 513, 514, 516, 520, 521, 523, 525, 527, 528, 530, 537, 540,
    542, 556, 558, 561, 570, 576, 577, 579, 582, 584, 588, 593, 600, 603, 609, 616, 618, 632, 638,
    640, 650, 653, 655, 656, 660, 666, 672, 675, 685, 688, 698, 705, 708, 711, 712, 715, 721, 727,
    728, 732, 737, 754, 760, 771, 773, 778, 780, 793, 795, 802, 806, 808, 812, 833, 840, 843, 849,
    856, 858, 873, 912, 916, 919, 932, 934, 961, 963, 968, 970, 977, 989, 993, 1010, 1016, 1024,
    1025, 1027, 1029, 1031, 1032, 1034, 1036, 1038, 1041, 1043, 1047, 1048, 1050, 1057, 1059, 1061,
    1064, 1066, 1079, 1080, 1083, 1085, 1088, 1090, 1096, 1099, 1103, 1106, 1109, 1113, 1116, 1122,
    1129, 1153, 1156, 1159, 1169, 1171, 1176, 1183, 1185, 1195, 1199, 1209, 1212, 1216, 1218, 1221,
    1225, 1234, 1236, 1241, 1243, 1250, 1256, 1270, 1281, 1287, 1296, 1299, 1306, 1309, 1313, 1338,
    1341, 1348, 1353, 1362, 1375, 1376, 1387, 1400, 1408, 1410, 1415, 1425, 1453, 1457, 1477, 1481,
    1494, 1496, 1507, 1512, 1538, 1545, 1547, 1549, 1551, 1554, 1561, 1563, 1565, 1570, 1572, 1575,
    1577, 1587, 1593, 1601, 1603, 1605, 1612, 1617, 1619, 1632, 1648, 1658, 1662, 1664, 1674, 1680,
    1690, 1692, 1704, 1729, 1736, 1740, 1745, 1747, 1751, 1752, 1761, 1763, 1767, 1773, 1787, 1795,
    1801, 1806, 1810, 1817, 1834, 1840, 1844, 1857, 1864, 1866, 1877, 1882, 1892, 1902, 1915, 1934,
    1953, 1985, 1987, 2000, 2002, 2013, 2048, 2052, 2058, 2064, 2068, 2071, 2074, 2081, 2088, 2104,
    2114, 2119, 2121, 2123, 2130, 2136, 2141, 2147, 2153, 2157, 2177, 2179, 2184, 2189, 2193, 2203,
    2208, 2223, 2226, 2232, 2244, 2249, 2251, 2256, 2258, 2265, 2269, 2304, 2306, 2324, 2335, 2336,
    2361, 2373, 2375, 2385, 2418, 2443, 2460, 2480, 2504, 2509, 2520, 2531, 2537, 2562, 2568, 2572,
    2578, 2592, 2596, 2599, 2602, 2614, 2620, 2625, 2627, 2629, 2634, 2641, 2650, 2682, 2688, 2697,
    2707, 2712, 2718, 2731, 2754, 2759, 2760, 2775, 2788, 2793, 2805, 2811, 2817, 2820, 2832, 2842,
    2854, 2890, 2902, 2921, 2923, 2978, 3010, 3012, 3026, 3081, 3083, 3085, 3097, 3099, 3120, 3136,
    3152, 3159, 3188, 3210, 3228, 3234, 3245, 3250, 3256, 3264, 3276, 3281, 3296, 3349, 3363, 3378,
    3392, 3395, 3420, 3440, 3461, 3488, 3529, 3531, 3584, 3588, 3591, 3600, 3602, 3614, 3616, 3628,
    3634, 3650, 3657, 3668, 3683, 3685, 3713, 3716, 3720, 3726, 3729, 3736, 3753, 3778, 3802, 3805,
    3819, 3841, 3845, 3851, 3856, 3880, 3922, 3938, 3970, 3993, 4032,
];

/// Returns the index into IQ3_DATA for a given grid size.
/// Panics if grid_size is not 256 or 512.
fn iq3_data_index(grid_size: i32) -> usize {
    assert!(
        grid_size == 256 || grid_size == 512,
        "grid_size must be 256 or 512"
    );
    if grid_size == 256 {
        0
    } else {
        1
    }
}

/// Helper: given a grid value (stored as u32) reinterpreted as 4 bytes,
/// compute the “index” value using: for each byte b, compute ((b-1)/2)
/// and pack it as bits (3 bits per coordinate).
fn compute_index_from_grid_val(val: u32) -> usize {
    let bytes = val.to_ne_bytes();
    let mut index = 0usize;
    for k in 0..4 {
        // (b - 1) / 2
        let q = (bytes[k].saturating_sub(1)) / 2;
        index |= (q as usize) << (3 * k);
    }
    index
}

/// Computes the squared Euclidean distance between two 4–byte positions.
fn dist2(a: &[u8; 4], b: &[u8; 4]) -> i32 {
    let mut d2 = 0;
    for k in 0..4 {
        let diff = a[k] as i32 - b[k] as i32;
        d2 += diff * diff;
    }
    d2
}

/// Main initialization function. This reproduces the C function iq3xs_init_impl.
fn iq3xs_init_impl(grid_size: i32) {
    // Determine which slot to use.
    let gindex = iq3_data_index(grid_size);
    // Use unsafe to access the global mutable state.
    unsafe {
        if IQ3_DATA[gindex].grid.is_some() {
            return;
        }
    }

    // Choose constants based on grid_size.
    let (kgrid, nwant) = if grid_size == 256 {
        (&KGRID_256[..], 2)
    } else {
        (&KGRID_512[..], 3)
    };
    let kmap_size = 4096;

    // --- Allocate and initialize the grid ---
    // For each element, we compute 4 bytes as: for each i in 0..4:
    //   byte = 2 * ((kgrid[k] >> (3*i)) & 0x7) + 1
    let mut grid: Vec<u32> = Vec::with_capacity(grid_size as usize);
    for &kg in kgrid.iter().take(grid_size as usize) {
        let mut bytes = [0u8; 4];
        for i in 0..4 {
            let l = (kg >> (3 * i)) & 0x7;
            bytes[i] = (2 * l + 1) as u8;
        }
        grid.push(u32::from_ne_bytes(bytes));
    }

    // --- Allocate and initialize the map ---
    // kmap: size = 4096, all initialized to -1.
    let mut kmap: Vec<i32> = vec![-1; kmap_size];

    // For each grid element, compute its index and store the grid index.
    for (j, &val) in grid.iter().enumerate() {
        let index = compute_index_from_grid_val(val);
        kmap[index] = j as i32;
    }

    // --- First pass: determine total space needed for neighbours ---
    let mut total_neighbors = 0;
    let mut num_not_in_map = 0;
    for i in 0..kmap_size {
        if kmap[i] >= 0 {
            continue;
        }
        num_not_in_map += 1;

        // Reconstruct the “position” from the map index.
        let mut pos = [0u8; 4];
        for k in 0..4 {
            let l = (i >> (3 * k)) & 0x7;
            pos[k] = (2 * l + 1) as u8;
        }
        // Build a vector of (distance, grid index) pairs.
        let mut dist_vec: Vec<(i32, usize)> = Vec::with_capacity(grid_size as usize);
        for (j, &grid_val) in grid.iter().enumerate() {
            let grid_bytes = grid_val.to_ne_bytes();
            let d = dist2(&grid_bytes, &pos);
            dist_vec.push((d, j));
        }
        // Sort the vector: first by distance, then by grid index.
        dist_vec.sort_by(|a, b| a.cmp(&b));
        // Count how many neighbors to include.
        let mut n = 0;
        let mut nhave = 1;
        let mut d_current = dist_vec[0].0;
        for &(d, _) in dist_vec.iter() {
            if d > d_current {
                if nhave == nwant {
                    break;
                }
                d_current = d;
                nhave += 1;
            }
            n += 1;
        }
        total_neighbors += n;
    }

    // Allocate neighbours vector with the total number of u16 values.
    // Note: the C code allocates (num_neighbors + num_not_in_map) elements.
    let total_nbrs_size = total_neighbors + num_not_in_map;
    let mut neighbours: Vec<u16> = Vec::with_capacity(total_nbrs_size);

    // --- Second pass: fill in the neighbours data and update kmap ---
    let mut nbr_counter = 0; // global counter in the neighbours vector
    for i in 0..kmap_size {
        if kmap[i] >= 0 {
            continue;
        }
        // Reconstruct the “position” from the map index.
        let mut pos = [0u8; 4];
        for k in 0..4 {
            let l = (i >> (3 * k)) & 0x7;
            pos[k] = (2 * l + 1) as u8;
        }
        // Build and sort the distances for all grid elements.
        let mut dist_vec: Vec<(i32, usize)> = Vec::with_capacity(grid_size as usize);
        for (j, &grid_val) in grid.iter().enumerate() {
            let grid_bytes = grid_val.to_ne_bytes();
            let d = dist2(&grid_bytes, &pos);
            dist_vec.push((d, j));
        }
        dist_vec.sort_by(|a, b| a.cmp(&b));

        // Store negative index in kmap to indicate start offset in the neighbours vector.
        kmap[i] = -((nbr_counter as i32) + 1);

        // Reserve a slot for the count of neighbours.
        neighbours.push(0); // placeholder; will update later
        nbr_counter += 1;

        // Now, add the neighbour indices.
        let mut n = 0;
        let mut nhave = 1;
        let mut d_current = dist_vec[0].0;
        for &(d, j) in dist_vec.iter() {
            if d > d_current {
                if nhave == nwant {
                    break;
                }
                d_current = d;
                nhave += 1;
            }
            // Store the grid index as u16.
            neighbours.push(j as u16);
            nbr_counter += 1;
            n += 1;
        }
        // Update the placeholder with the count of neighbours for this cell.
        neighbours[nbr_counter - n - 1] = n as u16;
    }

    // Finally, update the global IQ3_DATA entry.
    unsafe {
        IQ3_DATA[gindex].grid = Some(grid);
        IQ3_DATA[gindex].map = Some(kmap);
        IQ3_DATA[gindex].neighbours = Some(neighbours);
    }
}

pub unsafe fn iq3_find_best_neighbour(
    neighbours: *const u16,
    grid: *const u32,
    xval: *const f32,
    weight: *const f32,
    scale: f32,
    L: *mut i8,
) -> i32 {
    // neighbours[0] holds the number of neighbours.
    let num_neighbors = *neighbours as i32;
    assert!(num_neighbors > 0);
    let mut best_d2 = f32::MAX;
    let mut grid_index: i32 = -1;
    // j from 1 to num_neighbors (inclusive)
    for j in 1..=num_neighbors {
        // neighbours[j]
        let neigh = *neighbours.add(j as usize);
        // Compute pointer pg = (const int8_t*)(grid + neighbours[j])
        let pg = (grid.add(neigh as usize)) as *const i8;
        let mut d2 = 0f32;
        for i in 0..4 {
            // Note: pg[i] is read as i8 then converted to f32.
            let q = *pg.add(i) as f32;
            let diff = scale * q - *xval.add(i);
            d2 += *weight.add(i) * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            grid_index = neigh as i32;
        }
    }
    assert!(grid_index >= 0);
    let pg = (grid.add(grid_index as usize)) as *const i8;
    for i in 0..4 {
        // Here we assume that (pg[i]-1)/2 uses integer arithmetic.
        *L.add(i) = ((*pg.add(i)) - 1) / 2;
    }
    grid_index
}

pub unsafe fn quantize_row_iq3_xxs_impl(
    grid_size: i32,
    x: *const f32,
    y: *mut BlockIQ3xxs,
    n: i64,
    quant_weights: Option<&[f32]>,
) {
    iq3xs_init_impl(grid_size);

    // Assume iq3_data_index is defined elsewhere.
    let gindex = iq3_data_index(grid_size);

    // Assume iq3_data is a global array with fields: grid, map, neighbours.
    let kgrid_q3xs: *const u32 = IQ3_DATA[gindex].grid.as_ref().unwrap().as_ptr();
    let kmap_q3xs: *const i32 = IQ3_DATA[gindex].map.as_ref().unwrap().as_ptr();
    let kneighbors_q3xs: *const u16 = IQ3_DATA[gindex].neighbours.as_ref().unwrap().as_ptr();

    assert!(n % (QK_K as i64) == 0);

    let k_max_q: i32 = 8;
    let nbl = n / (QK_K as i64);

    // Variables to hold output pointers.
    let mut dh = &raw mut (*y).d;
    let mut qs = (*y).qs.as_mut_ptr();
    let block_size = mem::size_of::<BlockIQ3xxs>();
    let quant_size = block_size - mem::size_of::<f16>();

    // Allocate temporary arrays on the stack.
    let mut scales = [0f32; QK_K as usize / 32];
    let mut weight_arr = [0f32; 32];
    let mut xval_arr = [0f32; 32];
    let mut L_arr = [0i8; 32];
    let mut Laux_arr = [0i8; 32];
    let mut waux_arr = [0f32; 32];
    let mut is_on_grid = [false; 8];
    let mut is_on_grid_aux = [false; 8];
    let mut block_signs = [0u8; 8];
    let mut q3 = [0u8; 3 * (QK_K as usize / 8) + (QK_K as usize / 32)];

    // Calculate pointers into q3
    let scales_and_signs = q3[QK_K as usize / 4..].as_mut_ptr() as *mut u32;
    let qh = q3[3 * (QK_K as usize / 8)..].as_mut_ptr();

    // For each block of QK_K values:
    for ibl in 0..nbl as usize {
        // Set the first fp16 value to zero.
        *dh = f16::from_f32(0.0);
        ptr::write_bytes(
            q3.as_mut_ptr(),
            0,
            3 * (QK_K as usize / 8) + (QK_K as usize / 32),
        );

        let mut max_scale = 0f32;
        let xbl = x.add(QK_K as usize * ibl);
        let mut sumx2 = 0f32;
        for i in 0..(QK_K as usize) {
            let xi = *xbl.add(i);
            sumx2 += xi * xi;
        }
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        for ib in 0..(QK_K as usize / 32) {
            let xb = xbl.add(32 * ib);
            if let Some(quant_weights) = quant_weights {
                let qw = &quant_weights[QK_K as usize * ibl + 32 * ib..];
                for i in 0..32 {
                    weight_arr[i] = qw[i] * ((sigma2 + (*xb.add(i)) * (*xb.add(i))).sqrt());
                }
            } else {
                for i in 0..32 {
                    weight_arr[i] = *xb.add(i) * (*xb.add(i));
                }
            }
            for i in 0..32 {
                waux_arr[i] = weight_arr[i].sqrt();
            }
            for k in 0..4 {
                let mut nflip = 0;
                let mut s: u8 = 0;
                for i in 0..8 {
                    let val = *xb.add(8 * k + i);
                    if val >= 0.0 {
                        xval_arr[8 * k + i] = val;
                    } else {
                        xval_arr[8 * k + i] = -val;
                        nflip += 1;
                        s |= 1 << i;
                    }
                }
                if nflip % 2 != 0 {
                    let mut imin = 0;
                    let mut min_val = weight_arr[8 * k + imin]
                        * (*xb.add(8 * k + imin))
                        * (*xb.add(8 * k + imin));
                    for i in 1..8 {
                        let ax =
                            weight_arr[8 * k + i] * (*xb.add(8 * k + i)) * (*xb.add(8 * k + i));
                        if ax < min_val {
                            min_val = ax;
                            imin = i;
                        }
                    }
                    xval_arr[8 * k + imin] = -xval_arr[8 * k + imin];
                    s ^= 1 << imin;
                }
                block_signs[k] = s & 127;
            }
            let mut max_val = xval_arr[0];
            for i in 1..32 {
                max_val = if max_val > xval_arr[i] {
                    max_val
                } else {
                    xval_arr[i]
                };
            }
            if max_val < GROUP_MAX_EPS_IQ3_XXS {
                scales[ib] = 0.0;
                for i in 0..32 {
                    L_arr[i] = 0;
                }
                continue;
            }
            let mut best = 0f32;
            let mut scale = max_val / ((2 * k_max_q - 1) as f32);
            for is in -15..=15 {
                let id = ((2 * k_max_q - 1) as f32 + (is as f32) * 0.2) / max_val;
                let this_scale = 1.0 / id;
                for k in 0..8 {
                    for i in 0..4 {
                        // nearest_int and clamp must be defined elsewhere.
                        let l = nearest_int(0.5 * (id * xval_arr[4 * k + i] - 1.0));
                        Laux_arr[4 * k + i] = l.clamp(0, k_max_q - 1) as i8;
                    }
                    let mut u: u16 = 0;
                    for i in 0..4 {
                        u |= (Laux_arr[4 * k + i] as u16) << (3 * i);
                    }
                    let mut grid_index = *kmap_q3xs.add(u as usize);
                    is_on_grid_aux[k] = true;
                    if grid_index < 0 {
                        is_on_grid_aux[k] = false;
                        let neighbours =
                            kneighbors_q3xs.offset(-(*kmap_q3xs.add(u as usize) + 1) as isize);
                        grid_index = iq3_find_best_neighbour(
                            neighbours,
                            kgrid_q3xs,
                            xval_arr.as_ptr().add(4 * k),
                            waux_arr.as_ptr().add(4 * k),
                            this_scale,
                            Laux_arr.as_mut_ptr().add(4 * k),
                        );
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..32 {
                    let w = weight_arr[i];
                    let q = 2.0 * (Laux_arr[i] as f32) + 1.0;
                    sumqx += w * xval_arr[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for i in 0..32 {
                        L_arr[i] = Laux_arr[i];
                    }
                    for k in 0..8 {
                        is_on_grid[k] = is_on_grid_aux[k];
                    }
                }
            }
            let mut n_not_ongrid = 0;
            for k in 0..8 {
                if !is_on_grid[k] {
                    n_not_ongrid += 1;
                }
            }
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..8 {
                    if is_on_grid[k] {
                        continue;
                    }
                    let mut u: u16 = 0;
                    for i in 0..4 {
                        let mut l = nearest_int(0.5 * (id * xval_arr[4 * k + i] - 1.0));
                        l = l.clamp(0, k_max_q - 1);
                        u |= (l as u16) << (3 * i);
                    }
                    let mut grid_index = *kmap_q3xs.add(u as usize);
                    if grid_index < 0 {
                        let neighbours =
                            kneighbors_q3xs.offset(-(*kmap_q3xs.add(u as usize) + 1) as isize);
                        grid_index = iq3_find_best_neighbour(
                            neighbours,
                            kgrid_q3xs,
                            xval_arr.as_ptr().add(4 * k),
                            waux_arr.as_ptr().add(4 * k),
                            scale,
                            L_arr.as_mut_ptr().add(4 * k),
                        );
                    }
                    let pg = (kgrid_q3xs.add(grid_index as usize)) as *const i8;
                    for i in 0..4 {
                        L_arr[4 * k + i] = ((*pg.add(i)) - 1) / 2;
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..32 {
                    let w = weight_arr[i];
                    let q = 2.0 * (L_arr[i] as f32) + 1.0;
                    sumqx += w * xval_arr[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for k in 0..4 {
                    block_signs[k] = (!block_signs[k]) & 127;
                }
            }
            for k in 0..8 {
                let mut u: u16 = 0;
                for i in 0..4 {
                    u |= (L_arr[4 * k + i] as u16) << (3 * i);
                }
                let grid_index = *kmap_q3xs.add(u as usize);
                if grid_index < 0 {
                    println!("Oops: found point {} not on grid:", u);
                    for i in 0..4 {
                        print!(" {}", L_arr[4 * k + i]);
                    }
                    println!();
                    panic!("fatal error");
                }
                if grid_size == 256 {
                    q3[8 * ibl + k] = grid_index as u8;
                } else {
                    q3[8 * ibl + k] = (grid_index & 255) as u8;
                    *qh |= ((grid_index >> 8) as u8) << k;
                }
            }
            // Pack block_signs into scales_and_signs
            *scales_and_signs.add(ibl) = block_signs[0] as u32
                | ((block_signs[1] as u32) << 7)
                | ((block_signs[2] as u32) << 14)
                | ((block_signs[3] as u32) << 21);
            assert!(scale >= 0.0);
            scales[ibl] = scale;
            if scale > max_scale {
                max_scale = scale;
            }
        }

        if max_scale == 0.0 {
            ptr::write_bytes(qs, 0, quant_size as usize);
            dh = dh.add(block_size as usize / mem::size_of::<f16>());
            qs = qs.add(block_size as usize);
            continue;
        }
        let d = max_scale / 31.0;
        *dh = f16::from_f32(d * 1.0125);
        let id = 1.0 / d;
        for ib in 0..(QK_K as usize / 32) {
            let l = nearest_int(0.5 * (id * scales[ib] - 1.0));
            let l = l.clamp(0, 15);
            let prev = *scales_and_signs.add(ib);
            *scales_and_signs.add(ib) = prev | ((l as u32) << 28);
        }
        ptr::copy_nonoverlapping(q3.as_ptr(), qs, quant_size as usize);
        dh = dh.add(block_size as usize / mem::size_of::<f16>());
        qs = qs.add(block_size as usize);
    }
}
