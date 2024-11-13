#include <metal_stdlib>
using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

constant int THREADGROUP_SIZE = 2048;

template<typename T>
METAL_FUNC void argmin(
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    device uint *dst,
    uint id,
    uint tid,
    uint dst_id,
    uint block_dim,
    threadgroup T *shared_memory,
    threadgroup uint *shared_indices
) {
    bool notset = true;
    // Elements summed in this block range from dst_id * el_to_sum_per_block
    // to (dst_id + 1) * el_to_sum_per_block.
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = start_idx + el_to_sum_per_block;
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
        // TODO: Fast version for the contiguous case.
        size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
        if (notset || src[strided_i] < shared_memory[tid]) {
            shared_memory[tid] = src[strided_i];
            /* Assume that the reduction takes place over the last dimension which is contiguous. */
            shared_indices[tid] = idx % dims[num_dims - 1];
            notset = false;
        }
        idx += block_dim;
    }

    threadgroup_barrier(mem_flags::mem_none);
    // reduction in shared memory
    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s && shared_memory[tid + s] < shared_memory[tid]) {
            shared_indices[tid] = shared_indices[tid + s];
            shared_memory[tid] = shared_memory[tid + s];
        }  \
        threadgroup_barrier(mem_flags::mem_none);
    }
    if (tid == 0) {
    dst[dst_id] = shared_indices[0];
    }
}

#define ARGMIN(NAME, T, MAXVALUE) \
kernel void NAME( \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant size_t &el_to_sum_per_block, \
    device const T *src, \
    device uint *dst,  \
    uint id [[ thread_position_in_grid ]],  \
    uint tid [[ thread_index_in_threadgroup ]],  \
    uint dst_id [[ threadgroup_position_in_grid ]],  \
    uint block_dim [[ threads_per_threadgroup ]]  \
) {  \
    threadgroup T shared_memory[THREADGROUP_SIZE]; \
    threadgroup uint shared_indices[THREADGROUP_SIZE]; \
    shared_memory[tid] = MAXVALUE; \
    shared_indices[tid] = 0xFFFFFFFF; \
    argmin<T>(num_dims, dims, strides, el_to_sum_per_block, src, dst, id, tid, dst_id, block_dim, shared_memory, shared_indices); \
} \


template<typename T>
METAL_FUNC void argmax(
    constant size_t & num_dims,
    constant size_t * dims,
    constant size_t * strides,
    constant size_t & el_to_sum_per_block,
    device const T * src,
    device uint * dst,
    uint id,
    uint tid,
    uint dst_id,
    uint block_dim,
    threadgroup T * shared_memory,
    threadgroup uint * shared_indices
  ) {
    // Elements summed in this block range from dst_id * el_to_sum_per_block 
    // to (dst_id + 1) * el_to_sum_per_block.
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = start_idx + el_to_sum_per_block;
    size_t idx = start_idx + tid;
    bool notset = true;
    while (idx < stop_idx) {
        // TODO: Fast version for the contiguous case.
        size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
        if (notset || shared_memory[tid] < src[strided_i]) {
            shared_memory[tid] = src[strided_i];
            shared_indices[tid] = idx % dims[num_dims - 1];
            notset = false;
        }
        idx += block_dim;
    }

    threadgroup_barrier(mem_flags::mem_none);

    // reduction in shared memory
    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s && shared_memory[tid + s] > shared_memory[tid]) {
            shared_indices[tid] = shared_indices[tid + s];
            shared_memory[tid] = shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    // Thread 0 writes the result of the reduction
    if (tid == 0) {
        dst[dst_id] = shared_indices[0];
    }
  }

#define ARGMAX(NAME, T, MINVALUE) \
kernel void NAME( \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant size_t &el_to_sum_per_block, \
    device const T *src, \
    device uint *dst,  \
    uint id [[ thread_position_in_grid ]],  \
    uint tid [[ thread_index_in_threadgroup ]],  \
    uint dst_id [[ threadgroup_position_in_grid ]],  \
    uint block_dim [[ threads_per_threadgroup ]]  \
) {  \
   threadgroup T shared_memory[THREADGROUP_SIZE];  \
   threadgroup uint shared_indices[THREADGROUP_SIZE];  \
   shared_memory[tid] = MINVALUE;  \
   shared_indices[tid] = 0xFFFFFFFF; \
   argmax<T>(num_dims, dims, strides, el_to_sum_per_block, src, dst, id, tid, dst_id, block_dim, shared_memory, shared_indices);  \
} \

template<typename T>
METAL_FUNC void reduce(
    constant size_t & num_dims,
    constant size_t * dims,
    constant size_t * strides,
    constant size_t & el_to_sum_per_block,
    device const T * src,
    device T * dst,
    uint id,
    uint tid,
    uint dst_id,
    uint block_dim,
    threadgroup T * shared_memory,
    T (*fn)(T, T)
) {
    // Elements summed in this block range from dst_id * el_to_sum_per_block 
    // to (dst_id + 1) * el_to_sum_per_block.
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = start_idx + el_to_sum_per_block;
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
        // TODO: Fast version for the contiguous case.
        size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
        T x = shared_memory[tid];
        T y = src[strided_i];
        shared_memory[tid] = fn(x, y);
        idx += block_dim;
    }

    threadgroup_barrier(mem_flags::mem_none);

    // reduction in shared memory
    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            T x = shared_memory[tid];
            T y = shared_memory[tid + s];
            shared_memory[tid] = fn(x, y);
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    if (tid == 0) {
        dst[dst_id] = shared_memory[0];
    }
}

#define REDUCE(FN, NAME, T, START) \
METAL_FUNC T NAME##_##op(T x, T y) { return FN; } \
kernel void NAME( \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant size_t &el_to_sum_per_block, \
    device const T *src, \
    device T *dst, \
    uint id [[ thread_position_in_grid ]], \
    uint tid [[ thread_index_in_threadgroup ]], \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]] \
) { \
    threadgroup T shared_memory[THREADGROUP_SIZE]; \
    shared_memory[tid] = START; \
    reduce<T>(num_dims, dims, strides, el_to_sum_per_block, src, dst, id, tid, dst_id, block_dim, shared_memory, NAME##_##op); \
} \

template<typename T>
METAL_FUNC void softmax(
    constant size_t & src_numel,
    constant size_t & el_to_sum_per_block,
    device const T * src,
    device T * dst,
    uint id,
    uint tid,
    uint dst_id,
    uint block_dim,
    threadgroup float * shared_memory
) {
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    float tmp = -INFINITY;
    while (idx < stop_idx) {
        tmp = MAX(tmp, float(src[idx]));
        idx += block_dim;
    }
    shared_memory[tid] = tmp;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] = MAX(shared_memory[tid], shared_memory[tid + s]);\
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* wait for shared_memory[0] to be filled */
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float _max = shared_memory[0];

    /* prevent tid=0 from overwriting _max before other threads have written it */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shared_memory[tid] = 0;

    idx = start_idx + tid;
    while (idx < stop_idx) {
        const float val = exp(float(src[idx]) - _max);
        dst[idx] = T(val);
        shared_memory[tid] += val;
        idx += block_dim;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] += shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const T inv_acc = T(1.0 / shared_memory[0]);
    idx = start_idx + tid;
    while (idx < stop_idx) {
        dst[idx] *= inv_acc;
        idx += block_dim;
    }
}

#define SOFTMAX(NAME, T) \
kernel void NAME( \
    constant size_t &src_numel, \
    constant size_t &el_to_sum_per_block, \
    device const T *src, \
    device T *dst, \
    uint id [[ thread_position_in_grid ]], \
    uint tid [[ thread_index_in_threadgroup ]], \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]] \
) { \
    threadgroup float shared_memory[THREADGROUP_SIZE]; \
    shared_memory[tid] = -INFINITY; \
    softmax<T>(src_numel, el_to_sum_per_block, src, dst, id, tid, dst_id, block_dim, shared_memory); \
} \

template<typename T>
METAL_FUNC void rmsnorm(
    constant size_t & src_numel,
    constant size_t & el_to_sum_per_block,
    device const T * src,
    device T * dst,
    device const T * alpha,
    constant float & eps,
    uint id,
    uint tid,
    uint dst_id,
    uint block_dim,
    threadgroup float * shared_memory
) {
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    float tmp = 0;
    while (idx < stop_idx) {
        tmp = tmp + float(src[idx]) * float(src[idx]);
        idx += block_dim;
    }
    shared_memory[tid] = tmp;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] = shared_memory[tid] + shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* wait for shared_memory[0] to be filled */
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm = sqrt(shared_memory[0] / float(el_to_sum_per_block) + eps);
    float inv_norm = 1.0f / norm;
    idx = start_idx + tid;
    while (idx < stop_idx) {
        float val = float(src[idx]) * inv_norm;
        if (alpha != nullptr) {
            val *= float(alpha[idx - start_idx]);
        }
        dst[idx] = T(val);
        idx += block_dim;
    }
}

template<typename T>
METAL_FUNC void layernorm(
    constant size_t & src_numel,
    constant size_t & el_to_sum_per_block,
    device const T * src,
    device T * dst,
    device const T * alpha,
    device const T * beta,
    constant float & eps,
    uint id,
    uint tid,
    uint dst_id,
    uint block_dim,
    threadgroup float * shared_memory
) {
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    float tmp1 = 0;
    float tmp2 = 0;
    while (idx < stop_idx) {
        tmp1 += float(src[idx]);
        tmp2 += float(src[idx]) * float(src[idx]);
        idx += block_dim;
    }
    shared_memory[tid] = tmp1;
    shared_memory[tid + block_dim] = tmp2;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] = shared_memory[tid] + shared_memory[tid + s];
            shared_memory[block_dim + tid] = shared_memory[block_dim + tid] + shared_memory[block_dim + tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* wait for shared_memory[0] to be filled */
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = shared_memory[0] / float(el_to_sum_per_block);
    float var = shared_memory[block_dim] / float(el_to_sum_per_block) - mean * mean;
    float inv_norm = 1.0f / sqrt(var + eps);
    idx = start_idx + tid;
    while (idx < stop_idx) {
        float val = (float(src[idx]) - mean) * inv_norm;
        if (alpha != nullptr) {
            val *= float(alpha[idx - start_idx]);
        }
        if (beta != nullptr) {
            val += float(beta[idx - start_idx]);
        }
        dst[idx] = T(val);
        idx += block_dim;
    }
}

#define RMSNORM(NAME, T) \
kernel void NAME( \
    constant size_t &src_numel, \
    constant size_t &el_to_sum_per_block, \
    device const T *src, \
    device T *dst, \
    device const T *alpha, \
    constant float &eps, \
    uint id [[ thread_position_in_grid ]], \
    uint tid [[ thread_index_in_threadgroup ]], \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]] \
) { \
    threadgroup float shared_memory[THREADGROUP_SIZE]; \
    shared_memory[tid] = 0; \
    rmsnorm<T>(src_numel, el_to_sum_per_block, src, dst, alpha, eps, id, tid, dst_id, block_dim, shared_memory); \
} \

#define LAYERNORM(NAME, T) \
kernel void NAME( \
    constant size_t &src_numel, \
    constant size_t &el_to_sum_per_block, \
    device const T *src, \
    device T *dst, \
    device const T *alpha, \
    device const T *beta, \
    constant float &eps, \
    uint id [[ thread_position_in_grid ]], \
    uint tid [[ thread_index_in_threadgroup ]], \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]] \
) { \
    threadgroup float shared_memory[THREADGROUP_SIZE]; \
    shared_memory[tid] = 0; \
    layernorm<T>(src_numel, el_to_sum_per_block, src, dst, alpha, beta, eps, id, tid, dst_id, block_dim, shared_memory); \
} \

template<typename T>
METAL_FUNC void ropei(
    constant size_t &bh,
    constant size_t &td,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint tid
) {
    if (2 * tid >= bh * td) {
        return;
    }
    size_t rope_idx = tid % (td / 2);
    T c = cos[rope_idx];
    T s = sin[rope_idx];
    dst[2 * tid] = src[2 * tid] * c - src[2 * tid + 1] * s;
    dst[2 * tid + 1] = src[2 * tid] * s + src[2 * tid + 1] * c;
}

template<typename T>
METAL_FUNC void rope(
    constant size_t &bh,
    constant size_t &td,
    constant size_t &d,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx
) {
    if (2 * idx >= bh * td) {
        return;
    }
    size_t i_bh = idx / (td / 2);
    size_t i_td = idx - (td / 2) * i_bh;
    size_t i_t = i_td / (d / 2);
    size_t i_d = i_td - (d / 2) * i_t;
    size_t i1 = i_bh * td + i_t * d + i_d;
    size_t i2 = i1 + d / 2;
    size_t i_cs = i_t * (d / 2) + i_d;
    T c = cos[i_cs];
    T s = sin[i_cs];
    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template<typename T>
METAL_FUNC void rope_thd(
    constant size_t &b,
    constant size_t &t,
    constant size_t &h,
    constant size_t &d,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx
) {
    if (2 * idx >= b * t * h * d) {
        return;
    }
    const size_t i_bth = idx / (d / 2);
    const size_t i_d = idx - (d / 2) * i_bth;
    const size_t i_t = (i_bth / h) % t;
    const size_t i1 = i_bth * d + i_d;
    const size_t i2 = i1 + d / 2;
    const size_t i_cs = i_t * (d / 2) + i_d;
     T c = cos[i_cs];
    T s = sin[i_cs];
    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

#define ROPE(FN_NAME, FN_NAME_I, FN_NAME_THD, TYPENAME) \
kernel void FN_NAME_I( \
    constant size_t &bh, \
    constant size_t &td, \
    device const TYPENAME *src,  \
    device const TYPENAME *cos,  \
    device const TYPENAME *sin,  \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    ropei<TYPENAME>(bh, td, src, cos, sin, dst, tid); \
}\
kernel void FN_NAME( \
    constant size_t &bh, \
    constant size_t &td, \
    constant size_t &d, \
    device const TYPENAME *src,  \
    device const TYPENAME *cos,  \
    device const TYPENAME *sin,  \
    device TYPENAME *dst, \
    uint idx [[ thread_position_in_grid ]] \
) { \
    rope<TYPENAME>(bh, td, d, src, cos, sin, dst, idx); \
}\
kernel void FN_NAME_THD( \
    constant size_t &b, \
    constant size_t &t, \
    constant size_t &h, \
    constant size_t &d, \
    device const TYPENAME *src,  \
    device const TYPENAME *cos,  \
    device const TYPENAME *sin,  \
    device TYPENAME *dst, \
    uint idx [[ thread_position_in_grid ]] \
) { \
    rope_thd<TYPENAME>(b, t, h, d, src, cos, sin, dst, idx); \
}\


template<typename T>
kernel void attn_soft_max(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float * psrc0 = (device const float *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    device const     T * pmask = src1 != src0 ? (device const    T *) src1         + i01*ne00 : nullptr;
    device       float * pdst  = (device       float *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float slope = 1.0f;

    // parallel max
    float lmax = -INFINITY;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        lmax = MAX(lmax, psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        const float exp_psrc0 = exp((psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void attn_soft_max_4(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const T * psrc4 = (device const T *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;
    device const      T * pmask = src1 != src0 ? (device const     T *) src1         + i01*ne00/4 : nullptr;
    device       T * pdst4 = (device       T *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;

    float slope = 1.0f;

    // parallel max
    T lmax4 = -INFINITY;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + (T)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    T lsum4 = 0.0f;
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        const T exp_psrc4 = exp((psrc4[i00]*scale + (T)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        pdst4[i00] *= inv_sum;
    }
}


REDUCE(x + y, fast_sum_f32_strided, float, 0)
REDUCE(x + y, fast_sum_u32_strided, uint, 0)
REDUCE(x + y, fast_sum_f16_strided, half, 0)
REDUCE(x + y, fast_sum_u8_strided, uint8_t, 0)
REDUCE(x * y, fast_mul_f32_strided, float, 1)
REDUCE(x * y, fast_mul_u32_strided, uint, 1)
REDUCE(x * y, fast_mul_f16_strided, half, 1)
REDUCE(MAX(x, y), fast_max_f32_strided, float, -HUGE_VALF)
REDUCE(MAX(x, y), fast_max_u32_strided, uint, 0)
REDUCE(MAX(x, y), fast_max_f16_strided, half, -HUGE_VALH)
REDUCE(MAX(x, y), fast_max_u8_strided, uint8_t, 0)
REDUCE(MIN(x, y), fast_min_f32_strided, float, HUGE_VALF)
REDUCE(MIN(x, y), fast_min_u32_strided, uint, 0xFFFFFFFF)
REDUCE(MIN(x, y), fast_min_f16_strided, half, HUGE_VALH)
REDUCE(MIN(x, y), fast_min_u8_strided, uint8_t, 0xFF)
ARGMIN(fast_argmin_f32_strided, float, HUGE_VALF)
ARGMIN(fast_argmin_f16_strided, half, HUGE_VALH)
ARGMIN(fast_argmin_u32_strided, uint, 0xFFFFFFFF)
ARGMIN(fast_argmin_u8_strided, uint8_t, 0xFF)
ARGMAX(fast_argmax_f32_strided, float, -HUGE_VALF)
ARGMAX(fast_argmax_f16_strided, half, -HUGE_VALH)
ARGMAX(fast_argmax_u32_strided, uint, 0)
ARGMAX(fast_argmax_u8_strided, uint8_t, 0)

SOFTMAX(softmax_f32, float)
SOFTMAX(softmax_f16, half)
// Softmax for attention
typedef decltype(attn_soft_max<float>)    attn_soft_max_t;
typedef decltype(attn_soft_max_4<float4>) attn_soft_max_4_t;
template [[host_name("attn_soft_max_f16")]]   kernel attn_soft_max_t   attn_soft_max<half>;
template [[host_name("attn_soft_max_f32")]]   kernel attn_soft_max_t   attn_soft_max<float>;
template [[host_name("attn_soft_max_f16_4")]] kernel attn_soft_max_4_t attn_soft_max_4<half4>;
template [[host_name("attn_soft_max_f32_4")]] kernel attn_soft_max_4_t attn_soft_max_4<float4>;
RMSNORM(rmsnorm_f32, float)
RMSNORM(rmsnorm_f16, half)
LAYERNORM(layernorm_f32, float)
LAYERNORM(layernorm_f16, half)
ROPE(rope_f32, rope_i_f32, rope_thd_f32, float)
ROPE(rope_f16, rope_i_f16, rope_thd_f16, half)

#if __METAL_VERSION__ >= 220
REDUCE(x + y, fast_sum_i64_strided, int64_t, 0)
REDUCE(MIN(x, y), fast_min_i64_strided, int64_t, INT_MAX)
REDUCE(MAX(x, y), fast_max_i64_strided, int64_t, INT_MIN)
ARGMIN(fast_argmin_i64_strided, int64_t, INT_MAX)
ARGMAX(fast_argmax_i64_strided, int64_t, INT_MIN)
#endif

REDUCE(x + y, fast_sum_i32_strided, int32_t, 0)
REDUCE(MIN(x, y), fast_min_i32_strided, int32_t, INT_MAX)
REDUCE(MAX(x, y), fast_max_i32_strided, int32_t, INT_MIN)
ARGMIN(fast_argmin_i32_strided, int32_t, INT_MAX)
ARGMAX(fast_argmax_i32_strided, int32_t, INT_MIN)

REDUCE(x + y, fast_sum_i16_strided, int16_t, 0)
REDUCE(MIN(x, y), fast_min_i16_strided, int16_t, INT_MAX)
REDUCE(MAX(x, y), fast_max_i16_strided, int16_t, INT_MIN)
ARGMIN(fast_argmin_i16_strided, int16_t, INT_MAX)
ARGMAX(fast_argmax_i16_strided, int16_t, INT_MIN)

#if defined(__HAVE_BFLOAT__)
REDUCE(x + y, fast_sum_bf16, bfloat, 0)
REDUCE(x + y, fast_sum_bf16_strided, half, 0)
REDUCE(x * y, fast_mul_bf16, bfloat, 1)
REDUCE(x * y, fast_mul_bf16_strided, bfloat, 1)
REDUCE(MAX(x, y), fast_max_bf16, bfloat, -HUGE_VALBF)
REDUCE(MAX(x, y), fast_max_bf16_strided, bfloat, -HUGE_VALBF)
REDUCE(MIN(x, y), fast_min_bf16, bfloat, HUGE_VALBF)
REDUCE(MIN(x, y), fast_min_bf16_strided, bfloat, HUGE_VALBF)
ARGMIN(fast_argmin_bf16, bfloat, HUGE_VALBF)
ARGMAX(fast_argmax_bf16, bfloat, -HUGE_VALBF)
SOFTMAX(softmax_bf16, bfloat)
// // Softmax for attention
// template [[host_name("attn_soft_max_bf16")]]   kernel attn_soft_max_t   attn_soft_max<bfloat>;
// template [[host_name("attn_soft_max_bf16_4")]] kernel attn_soft_max_4_t attn_soft_max_4<bfloat4>;
RMSNORM(rmsnorm_bf16, bfloat)
LAYERNORM(layernorm_bf16, bfloat)
ROPE(rope_bf16, rope_i_bf16, rope_thd_bf16, bfloat)
#endif
