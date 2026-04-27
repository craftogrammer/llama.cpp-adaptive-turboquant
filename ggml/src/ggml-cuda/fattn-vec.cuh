#include "common.cuh"
#include "fattn-common.cuh"
#include "turbo-sink.cuh"

#include <cstdlib>

// Per-TU sink state — in fattn-vec.cuh so kernel and host code share the same TU copy.
// Only K sinks are used in the VEC kernel. V sinks were removed from the V accumulation
// loop because managed memory reads in the hot loop caused -3% short / -12% 32K regression.
// Uses __device__ + cudaMemcpyAsync (stream-ordered, graph-capturable).
// Previous __managed__ approach crashed on SM86 (page faults during graph replay).
//
// Multi-range support: TURBO_SINK_MAX_RANGES slots. Slot 0 is the static base
// [0, TURBO_SINK_SIZE); slots 1..MAX-1 are dynamic ranges registered by the
// server's <think>-detection hook. A range with width==0 is inactive and is
// skipped by the per-K-position check.
static __device__ const half * d_fattn_sink_K_bufs[TURBO_SINK_MAX_RANGES] = { nullptr };
static __device__ int64_t      d_fattn_sink_starts[TURBO_SINK_MAX_RANGES] = { 0 };
static __device__ int          d_fattn_sink_widths[TURBO_SINK_MAX_RANGES] = { 0 };
static __device__ int          d_fattn_sink_count = 0;
static __device__ int64_t      d_fattn_sink_ne0   = 0;

// Per-range hit counter, sampled at warp granularity (only thread 0 of each
// warp increments) so atomics don't dominate the FA hot path.
// Per-TU static — read back by the per-TU dispatcher into a process-global
// host accumulator (g_fattn_sink_hits_accum, defined in turbo-sink.cu) at
// the start of each FA call, gated by env TURBO_SINK_DIAG.
static __device__ unsigned long long d_fattn_sink_hits[TURBO_SINK_MAX_RANGES] = { 0 };

// Host-toggled gate for the per-hit warp-sampled atomic in
// fattn_sink_lookup_K_slow. Set to 1 once via cudaMemcpyToSymbol the first
// time the dispatcher sees TURBO_SINK_DIAG=1; stays 1 for the process
// lifetime (diag_hits is a one-time getenv-init static). When 0, the atomic
// is skipped entirely so DIAG-off cost is one global int load per warp-sampled
// hit instead of an atomic.
static __device__ int d_fattn_sink_diag_enabled = 0;

// Look up a fp16 K row across all active sink ranges. Returns nullptr if the
// position is not in any range.
//
// __noinline__: ptxas on Windows + nvcc 12.9 + sm_120 trips ACCESS_VIOLATION
// when this body is inlined into the mixed-pair FA TUs (turbo3_tcq×q8_0,
// turbo3_0×q8_0). The fully-unrolled version of the same logic also crashes.
// Same workaround pattern as vec_dot_fattn_vec_KQ_q4_0 / _turbo3_0 / _turbo3_tcq_decode.
//
// Cost: one function call per K-position attention check. For 64K-context
// decode this is ~4M calls/token × ~5 cycles/call ≈ ~10 ms/token, but only
// fires when at least one sink range is active (cheap early-out via
// d_fattn_sink_count == 0 fast-path is folded into the call site to keep
// the no-sink hot path identical to before this change).
static __device__ __noinline__ const half * fattn_sink_lookup_K_slow(int kv_pos) {
    for (int r = 0; r < TURBO_SINK_MAX_RANGES; ++r) {
        const int w = d_fattn_sink_widths[r];
        if (w > 0) {
            const int64_t s = d_fattn_sink_starts[r];
            if (kv_pos >= (int)s && kv_pos < (int)(s + w)) {
                const half * buf = d_fattn_sink_K_bufs[r];
                if (buf == nullptr) return nullptr;
                if ((threadIdx.x & 31) == 0 && d_fattn_sink_diag_enabled) {
                    atomicAdd(&d_fattn_sink_hits[r], 1ULL);
                }
                const int64_t row = (int64_t)kv_pos - s;
                return buf + row * d_fattn_sink_ne0;
            }
        }
    }
    return nullptr;
}

// Inline fast-path wrapper. The early-out keeps the no-sink hot path
// identical to the pre-multi-range cost (one int compare + branch). When
// ranges ARE active, dispatches to the __noinline__ slow path.
static __device__ __forceinline__ const half * fattn_sink_lookup_K(int kv_pos) {
    if (d_fattn_sink_count == 0) return nullptr;
    return fattn_sink_lookup_K_slow(kv_pos);
}

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

template<ggml_type type_K, int D, int nthreads>
static constexpr __host__ __device__ vec_dot_KQ_t get_vec_dot_KQ_fattn() {
    if constexpr (type_K == GGML_TYPE_TURBO3_0) {
        return nullptr;
    } else {
        return get_vec_dot_KQ<type_K, D, nthreads>();
    }
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
// `min_blocks_per_sm` is a template parameter so the dispatcher in
// ggml_cuda_flash_attn_ext_vec_case_impl can raise occupancy on (type_K, type_V)
// pairs that ptxas tolerates, without globally bumping minBlocks (which on
// Windows + nvcc 12.9 + sm_120 trips ACCESS_VIOLATION in fattn-vec-instance-
// f16-f16 and -q8_0-q8_0). The (turbo3_tcq, turbo3_tcq) same-type instance
// compiles cleanly at 4 and the higher occupancy gives noticeable long-context
// gains where the kernel is memory-latency bound. Default of 3 preserves
// behaviour for every other (type_K, type_V) instantiation.
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, int min_blocks_per_sm = 3>
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), min_blocks_per_sm)
static __global__ void flash_attn_ext_vec(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256 || D == 512)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    // ALL turbo types use nthreads_KQ=8 for better warp-level ILP at long context (S24B).
    // nthreads_KQ_q=32 gives 1 KQ dot per warp; 8 gives 4 interleaved dots → better latency hiding.
    // Dead End #19: nthreads_KQ=32 for turbo3 was -17% at 32K. turbo4/turbo1.5 had the same bug.
    constexpr bool K_is_unquantized = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16);
    constexpr bool V_is_unquantized = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16);
    constexpr bool K_is_turbo3_0    = (type_K == GGML_TYPE_TURBO3_0);
    constexpr bool K_is_turbo3_tcq  = (type_K == GGML_TYPE_TURBO3_TCQ);
    constexpr bool V_is_turbo3_0    = (type_V == GGML_TYPE_TURBO3_0);
    constexpr bool K_is_turbo2_0    = (type_K == GGML_TYPE_TURBO2_0);
    constexpr bool V_is_turbo2_0    = (type_V == GGML_TYPE_TURBO2_0);
    constexpr bool K_is_turbo       = (type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO2_0 ||
                                       type_K == GGML_TYPE_TURBO4_0 || type_K == GGML_TYPE_TURBO1_5 ||
                                       type_K == GGML_TYPE_TURBO3_TCQ || type_K == GGML_TYPE_TURBO2_TCQ);
    constexpr int nthreads_KQ = K_is_unquantized ? 128 / cpy_nb : (K_is_turbo3_tcq ? (D >= 256 ? 128 / cpy_nb : 64 / cpy_nb) : (K_is_turbo3_0 || K_is_turbo2_0 ? 16 / cpy_nb : (K_is_turbo ? 128 / cpy_nb : nthreads_KQ_q)));
    constexpr int nthreads_V  = V_is_unquantized ? 128 / cpy_nb : (V_is_turbo3_0 || V_is_turbo2_0 ? 8 : nthreads_V_q);

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = V_is_unquantized ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr bool K_is_tcq = (type_K == GGML_TYPE_TURBO3_TCQ || type_K == GGML_TYPE_TURBO2_TCQ);
    constexpr bool Q_q8_1 = !K_is_unquantized && !K_is_tcq;

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE

    // TCQ codebook in shared memory for K and V dequant.
    // Constant memory serializes when threads hit different 32B cache lines;
    // shared memory gives full 32-bank parallel access for random lookups.
    // When K and V use different TCQ types, load both codebooks separately.
    constexpr bool K_is_tcq3 = type_K == GGML_TYPE_TURBO3_TCQ;
    constexpr bool K_is_tcq2 = type_K == GGML_TYPE_TURBO2_TCQ;
    constexpr bool V_is_tcq3 = type_V == GGML_TYPE_TURBO3_TCQ;
    constexpr bool V_is_tcq2 = type_V == GGML_TYPE_TURBO2_TCQ;
    constexpr bool K_tcq_uses_smem_cb = (K_is_tcq3 || K_is_tcq2) && type_V != GGML_TYPE_Q8_0;
    constexpr int smem_cb_K_size = K_tcq_uses_smem_cb ? (K_is_tcq3 ? 512 : 256) : 0;
    constexpr int smem_cb_V_size = V_is_tcq3 ? 512 : (V_is_tcq2 ? 256 : 0);
    constexpr bool share_cb = (K_is_tcq3 && V_is_tcq3) || (K_is_tcq2 && V_is_tcq2);
    constexpr int smem_cb_total = share_cb ? smem_cb_K_size : (smem_cb_K_size + smem_cb_V_size);
    __shared__ float smem_codebook_buf[smem_cb_total > 0 ? smem_cb_total : 1];
#define FATTN_SMEM_CODEBOOK_K smem_codebook_buf
#define FATTN_SMEM_CODEBOOK_V (share_cb ? smem_codebook_buf : (smem_codebook_buf + smem_cb_K_size))
    if constexpr (smem_cb_K_size > 0) {
        const float * cb_K_src = K_is_tcq3 ? d_turbo3_tcq_codebook : d_turbo2_tcq_codebook;
        for (int i = tid; i < smem_cb_K_size; i += nthreads) {
            FATTN_SMEM_CODEBOOK_K[i] = cb_K_src[i];
        }
    }
    if constexpr (smem_cb_V_size > 0 && !share_cb) {
        const float * cb_V_src = V_is_tcq3 ? d_turbo3_tcq_codebook : d_turbo2_tcq_codebook;
        for (int i = tid; i < smem_cb_V_size; i += nthreads) {
            FATTN_SMEM_CODEBOOK_V[i] = cb_V_src[i];
        }
    }
    if constexpr (smem_cb_total > 0) {
        __syncthreads();
    }

    // Sparse V: skip V dequant for positions with negligible attention weights.
    // Keep the validated conservative threshold to avoid quality regressions.
    constexpr float sparse_v_threshold_f = 1e-6f;
#ifdef V_DOT2_F32_F16_AVAILABLE
    const     half  sparse_v_threshold_h = __float2half(sparse_v_threshold_f);
#endif

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // L2 prefetch: issue hints for the NEXT outer iteration's K and V blocks
        // while computing the current iteration. Hides GDDR latency (~300-400 cycles).
        // Available on SM50+, no correctness impact (hint only).
#if __CUDA_ARCH__ >= 500
        if (k_VKQ_0 + gridDim.y*nthreads < k_VKQ_max) {
            const char * K_next = K + gridDim.y*nthreads*nb11;
            const char * V_next = V + gridDim.y*nthreads*nb21;
            // Prefetch first few K and V cache lines of next chunk (128 bytes each)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(K_next));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(K_next + nb11));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(V_next));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(V_next + nb21));
        }
#endif

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum;
                // Turbo attention: sinks (fp16) check, then LUT or standard vec_dot
                if constexpr (type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO4_0 ||
                              type_K == GGML_TYPE_TURBO2_0 || type_K == GGML_TYPE_TURBO1_5) {
                    const int kv_pos = k_VKQ_0 + i_KQ;
                    const half * sink_row = fattn_sink_lookup_K(kv_pos);
                    if (sink_row != nullptr) {
                        const int kv_head = head / gqa_ratio;
                        const char * sink_K = (const char *)(sink_row + kv_head * D);
                        sum = vec_dot_fattn_vec_KQ_f16<D, nthreads_KQ>(sink_K, Q_reg[j], Q_i32[j], Q_ds[j]);
                    } else if constexpr (type_K == GGML_TYPE_TURBO3_0) {
                        sum = vec_dot_fattn_vec_KQ_turbo3_0_lean<D, nthreads_KQ>(
                            K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                    } else {
                        sum = get_vec_dot_KQ_fattn<type_K, D, nthreads_KQ>()(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                    }
                } else if constexpr (type_K == GGML_TYPE_TURBO3_TCQ) {
                    // Sink fast-path: any active sink range (base [0, TURBO_SINK_SIZE)
                    // OR a dynamic <think>-anchor range) uses captured fp16 K.
                    const int kv_pos = k_VKQ_0 + i_KQ;
                    const half * sink_row = fattn_sink_lookup_K(kv_pos);
                    if (sink_row != nullptr) {
                        const int kv_head = head / gqa_ratio;
                        const char * sink_K = (const char *)(sink_row + kv_head * D);
                        sum = vec_dot_fattn_vec_KQ_f16<D, nthreads_KQ>(sink_K, Q_reg[j], Q_i32[j], Q_ds[j]);
                    } else if constexpr (type_V == GGML_TYPE_TURBO3_TCQ) {
                        // Same-type TU: ptxas tolerates inlining the lean (unroll-1)
                        // body here, so call the __forceinline__ sibling and skip
                        // the per-K-position function-call overhead. Mixed-pair TUs
                        // (V=q8_0) fall through to the __noinline__ variant below.
                        sum = vec_dot_fattn_vec_KQ_turbo3_tcq_decode_inline<D, nthreads_KQ>(
                            K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                    } else {
                        sum = vec_dot_fattn_vec_KQ_turbo3_tcq_decode<D, nthreads_KQ>(
                            K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                    }
                } else if constexpr (type_K == GGML_TYPE_TURBO2_TCQ) {
                    const int kv_pos = k_VKQ_0 + i_KQ;
                    const half * sink_row = fattn_sink_lookup_K(kv_pos);
                    if (sink_row != nullptr) {
                        const int kv_head = head / gqa_ratio;
                        const char * sink_K = (const char *)(sink_row + kv_head * D);
                        sum = vec_dot_fattn_vec_KQ_f16<D, nthreads_KQ>(sink_K, Q_reg[j], Q_i32[j], Q_ds[j]);
                    } else {
                        if constexpr (type_V == GGML_TYPE_Q8_0) {
                            sum = vec_dot_fattn_vec_KQ_turbo2_tcq<D, nthreads_KQ>(
                                K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                        } else {
                            sum = vec_dot_fattn_vec_KQ_turbo2_tcq_cb<D, nthreads_KQ>(
                                K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j], FATTN_SMEM_CODEBOOK_K);
                        }
                    }
                } else {
                    sum = get_vec_dot_KQ_fattn<type_K, D, nthreads_KQ>()(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                }
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope*__half2float(maskh[j*ne11 + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = __expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = __expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif // GGML_USE_HIP

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }

            // Sparse V: skip V dequant if all attention weights for this position are negligible
            {
                bool dominated = true;
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    if (__hgt(__low2half(KQ_k[j]), sparse_v_threshold_h)) { dominated = false; break; }
                }
                if (dominated) { continue; }
            }

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    get_dequantize_V<type_V, float, V_rows_per_thread>()(V + k*nb21, tmp_f,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else if constexpr (type_V == GGML_TYPE_TURBO3_TCQ) {
                    if constexpr (type_K == GGML_TYPE_TURBO3_TCQ) {
                        // Same-type FA TU: ptxas tolerates the inlined V dequant body
                        // (the 2-bit V sibling is already __forceinline__ and compiles
                        // fine, so the 3-bit workaround was inherited rather than
                        // empirically required). Skip the per-V-element call overhead.
                        dequantize_V_turbo3_tcq_cb_inline<half, V_rows_per_thread>(V + k*nb21, tmp,
                            2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread, FATTN_SMEM_CODEBOOK_V);
                    } else {
                        dequantize_V_turbo3_tcq_cb<half, V_rows_per_thread>(V + k*nb21, tmp,
                            2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread, FATTN_SMEM_CODEBOOK_V);
                    }
                } else if constexpr (type_V == GGML_TYPE_TURBO2_TCQ) {
                    dequantize_V_turbo2_tcq_cb<half, V_rows_per_thread>(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread, FATTN_SMEM_CODEBOOK_V);
                } else {
                    get_dequantize_V<type_V, half, V_rows_per_thread>()(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }

            // Sparse V: skip V dequant if all attention weights for this position are negligible
            {
                bool dominated = true;
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    if (KQ_k[j] >= sparse_v_threshold_f) { dominated = false; break; }
                }
                if (dominated) { continue; }
            }

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_TURBO3_TCQ) {
                    if constexpr (type_K == GGML_TYPE_TURBO3_TCQ) {
                        // Same-type FA TU: inline V dequant body (see half-accum
                        // branch above for rationale).
                        dequantize_V_turbo3_tcq_cb_inline<float, V_rows_per_thread>(V + k*nb21, tmp,
                            2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread, FATTN_SMEM_CODEBOOK_V);
                    } else {
                        dequantize_V_turbo3_tcq_cb<float, V_rows_per_thread>(V + k*nb21, tmp,
                            2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread, FATTN_SMEM_CODEBOOK_V);
                    }
                } else if constexpr (type_V == GGML_TYPE_TURBO2_TCQ) {
                    dequantize_V_turbo2_tcq_cb<float, V_rows_per_thread>(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread, FATTN_SMEM_CODEBOOK_V);
                } else {
                    get_dequantize_V<type_V, float, V_rows_per_thread>()(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = __expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? __expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = __expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#undef FATTN_SMEM_CODEBOOK_K
#undef FATTN_SMEM_CODEBOOK_V
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Turbo attention sinks: set device memory state for K scoring in VEC kernel.
    // Uses __device__ variables updated via cudaGetSymbolAddress + cudaMemcpyAsync
    // (graph-capturable). Previous __managed__ approach crashed on SM86.
    if constexpr (type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO4_0 ||
                  type_K == GGML_TYPE_TURBO2_0 || type_K == GGML_TYPE_TURBO1_5 ||
                  type_K == GGML_TYPE_TURBO3_TCQ || type_K == GGML_TYPE_TURBO2_TCQ) {
        const ggml_tensor * K = dst->src[1];

        // Snapshot all currently-active sink ranges (base + dynamic) for THIS tensor.
        // The canonical ne0 comes from the buffer entry (allocated at write time);
        // it equals the row stride in halfs that the capture kernel wrote with.
        turbo_sink_range_view ranges[TURBO_SINK_MAX_RANGES];
        int64_t k_ne0_full = 0;
        const int n_ranges = turbo_sink_collect_active_ranges_K((void *)K->data, ranges, &k_ne0_full);

        // Always upload (even when n_ranges == 0) so a previously-active count
        // is correctly cleared on the device. The cost is ~50 bytes per FA call
        // and the cudaMemcpyAsync is graph-capturable.
        static void * d_addr_bufs   = nullptr;
        static void * d_addr_starts = nullptr;
        static void * d_addr_widths = nullptr;
        static void * d_addr_count  = nullptr;
        static void * d_addr_ne0    = nullptr;
        if (!d_addr_bufs) {
            CUDA_CHECK(cudaGetSymbolAddress(&d_addr_bufs,   d_fattn_sink_K_bufs));
            CUDA_CHECK(cudaGetSymbolAddress(&d_addr_starts, d_fattn_sink_starts));
            CUDA_CHECK(cudaGetSymbolAddress(&d_addr_widths, d_fattn_sink_widths));
            CUDA_CHECK(cudaGetSymbolAddress(&d_addr_count,  d_fattn_sink_count));
            CUDA_CHECK(cudaGetSymbolAddress(&d_addr_ne0,    d_fattn_sink_ne0));
        }

        // Aligned static staging — unaligned stack vars caused SM89 segfault
        // historically (preserved from the original impl).
        struct __align__(16) sink_upload_state {
            const half * bufs   [TURBO_SINK_MAX_RANGES];
            int64_t      starts [TURBO_SINK_MAX_RANGES];
            int          widths [TURBO_SINK_MAX_RANGES];
            int          count;
            int64_t      ne0;
        };
        static sink_upload_state ust;
        for (int r = 0; r < TURBO_SINK_MAX_RANGES; ++r) {
            if (r < n_ranges) {
                ust.bufs[r]   = ranges[r].K_buf;
                ust.starts[r] = ranges[r].start;
                ust.widths[r] = ranges[r].width;
            } else {
                ust.bufs[r]   = nullptr;
                ust.starts[r] = 0;
                ust.widths[r] = 0;
            }
        }
        ust.count = n_ranges;
        ust.ne0   = k_ne0_full;

        cudaStream_t stream = ctx.stream();
        CUDA_CHECK(cudaMemcpyAsync(d_addr_bufs,   ust.bufs,    sizeof(ust.bufs),    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_addr_starts, ust.starts,  sizeof(ust.starts),  cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_addr_widths, ust.widths,  sizeof(ust.widths),  cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_addr_count, &ust.count,   sizeof(ust.count),   cudaMemcpyHostToDevice, stream));

        // DIAGNOSTIC (env TURBO_SINK_DIAG=1): read back this TU's hit counter,
        // accumulate into the process-global host counter (defined in
        // turbo-sink.cu), then zero the device counter. One sync per FA call —
        // gated by env so release perf is unchanged.
        static const bool diag_hits = (std::getenv("TURBO_SINK_DIAG") != nullptr);
        if (diag_hits) {
            // Push the device-side gate to 1 once per TU lifetime so the
            // warp-sampled atomic in fattn_sink_lookup_K_slow becomes active.
            // Without this, h_hits below would always read back zero.
            static bool diag_flag_pushed = false;
            if (!diag_flag_pushed) {
                int one = 1;
                CUDA_CHECK(cudaMemcpyToSymbol(d_fattn_sink_diag_enabled, &one, sizeof(one), 0, cudaMemcpyHostToDevice));
                diag_flag_pushed = true;
            }
            extern unsigned long long g_fattn_sink_hits_accum[TURBO_SINK_MAX_RANGES];
            unsigned long long h_hits[TURBO_SINK_MAX_RANGES] = { 0 };
            CUDA_CHECK(cudaMemcpyFromSymbol(h_hits, d_fattn_sink_hits, sizeof(h_hits), 0, cudaMemcpyDeviceToHost));
            for (int i = 0; i < TURBO_SINK_MAX_RANGES; ++i) {
                g_fattn_sink_hits_accum[i] += h_hits[i];
            }
            unsigned long long zeros[TURBO_SINK_MAX_RANGES] = { 0 };
            CUDA_CHECK(cudaMemcpyToSymbol(d_fattn_sink_hits, zeros, sizeof(zeros), 0, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpyAsync(d_addr_ne0,   &ust.ne0,     sizeof(ust.ne0),     cudaMemcpyHostToDevice, stream));
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    // Raise FA-vec occupancy to ~33% on sm_120 for the same-type turbo3_tcq path
    // (vs 25% at the default minBlocks=3), giving the SM more in-flight warps
    // for the memory-latency-bound TCQ codebook lookups that dominate decode at
    // long context. Trade-off measured on Qwen3.6-27B IQ3 / RTX 5080:
    //   d=4K  : -2.3% TG (register pressure costs more than the extra warps help)
    //   d=16K : -2.9% TG
    //   d=32K : flat
    //   d=64K : +2.5% TG
    // Kept because user's workload includes 128K-context use-cases where the
    // long-context win compounds (and where decode is otherwise dropping to
    // ~11 t/s). All other (type_K, type_V) pairs keep minBlocks=3 because
    // f16-f16 / q8_0-q8_0 ptxas crashes at 4 on Windows + nvcc 12.9 + sm_120.
    fattn_kernel_t fattn_kernel;
    if constexpr (type_K == GGML_TYPE_TURBO3_TCQ && type_V == GGML_TYPE_TURBO3_TCQ) {
        fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap, 4>;
    } else {
        fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap, 3>;
    }
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case_no_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    GGML_ASSERT(logit_softcap == 0.0f);

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        return;
    }

    constexpr int cols_per_block = 2;
    constexpr bool use_logit_softcap = false;
    ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case_decode_no_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    GGML_ASSERT(logit_softcap == 0.0f);
    GGML_ASSERT(Q->ne[1] == 1);

    constexpr int cols_per_block = 1;
    constexpr bool use_logit_softcap = false;
    ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case_prefill_no_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    GGML_ASSERT(logit_softcap == 0.0f);
    GGML_ASSERT(Q->ne[1] != 1);

    constexpr int cols_per_block = 2;
    constexpr bool use_logit_softcap = false;
    ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_VEC_CASE_NO_SOFTCAP(D, type_K, type_V)                   \
    template void ggml_cuda_flash_attn_ext_vec_case_no_softcap              \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP(D, type_K, type_V)            \
    template void ggml_cuda_flash_attn_ext_vec_case_decode_no_softcap       \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP(D, type_K, type_V)           \
    template void ggml_cuda_flash_attn_ext_vec_case_prefill_no_softcap      \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)

// Macro for extern declarations with D=64, 128, 256, 512
#define EXTERN_DECL_FATTN_VEC_TURBO(type_K, type_V) \
    extern DECL_FATTN_VEC_CASE( 64, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE(128, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE(256, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE(512, type_K, type_V);

#define EXTERN_DECL_FATTN_VEC_TURBO_NO_SOFTCAP(type_K, type_V) \
    extern DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP( 64, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP( 64, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP(128, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP(128, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP(256, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP(256, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP(512, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP(512, type_K, type_V);

#define EXTERN_DECL_FATTN_VEC_TURBO_Q8_NO_SOFTCAP(type_K, type_V) \
    extern DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP( 64, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP( 64, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP(128, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP(128, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP(256, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP(256, type_K, type_V);

// Symmetric turbo types
EXTERN_DECL_FATTN_VEC_TURBO_NO_SOFTCAP(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
EXTERN_DECL_FATTN_VEC_TURBO_NO_SOFTCAP(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO1_5, GGML_TYPE_TURBO1_5)

// turbo × q8_0 cross-types
EXTERN_DECL_FATTN_VEC_TURBO_Q8_NO_SOFTCAP(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)
EXTERN_DECL_FATTN_VEC_TURBO_Q8_NO_SOFTCAP(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO1_5, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO1_5)

// turbo × turbo cross-types (all permutations)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO1_5)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO1_5)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO1_5)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO1_5, GGML_TYPE_TURBO4_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO1_5, GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO1_5, GGML_TYPE_TURBO2_0)

// turbo × f16 cross-types
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO4_0, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO3_0, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO2_0, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_TURBO1_5, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_F16, GGML_TYPE_TURBO4_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_F16, GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_F16, GGML_TYPE_TURBO2_0)
EXTERN_DECL_FATTN_VEC_TURBO(GGML_TYPE_F16, GGML_TYPE_TURBO1_5)

// TCQ types (D=128, 256 only — D=64 excluded because QK block size is 128)
#define EXTERN_DECL_FATTN_VEC_TCQ(type_K, type_V) \
    extern DECL_FATTN_VEC_CASE(128, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE(256, type_K, type_V);

#define EXTERN_DECL_FATTN_VEC_TCQ_NO_SOFTCAP(type_K, type_V) \
    extern DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP(128, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP(128, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP(256, type_K, type_V); \
    extern DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP(256, type_K, type_V);

#define EXTERN_DECL_FATTN_VEC_TCQ_DECODE_NO_SOFTCAP(type_K, type_V) \
    extern DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP(256, type_K, type_V);

#define EXTERN_DECL_FATTN_VEC_TCQ_PREFILL_NO_SOFTCAP(type_K, type_V) \
    extern DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP(256, type_K, type_V);

// Symmetric TCQ types
EXTERN_DECL_FATTN_VEC_TCQ_NO_SOFTCAP(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
EXTERN_DECL_FATTN_VEC_TCQ(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)

// TCQ cross-types (turbo3_tcq x turbo2_tcq)
EXTERN_DECL_FATTN_VEC_TCQ(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ)
EXTERN_DECL_FATTN_VEC_TCQ(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO3_TCQ)

// TCQ x q8_0 cross-types used by layer-adaptive TCQ.
EXTERN_DECL_FATTN_VEC_TCQ_DECODE_NO_SOFTCAP(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TCQ_PREFILL_NO_SOFTCAP(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TCQ_NO_SOFTCAP(GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_TCQ)
