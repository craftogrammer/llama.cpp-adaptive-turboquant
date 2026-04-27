#include "turbo-sink.cuh"
#include "turbo-quant.cuh"

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <algorithm>
#include <atomic>

// ─── Environment variables ────────────────────────────────────────────────────

int turbo_sink_size() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("TURBO_SINK_SIZE");
        cached = env ? atoi(env) : 0;
        if (cached > 0) {
            fprintf(stderr, "[turbo] attention sinks: first %d positions at fp16 precision\n", cached);
        }
    }
    return cached;
}

#define TURBO_THINK_SINK_WIDTH_MAX 64

int turbo_think_sink_width() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("TURBO_THINK_SINK_WIDTH");
        int w = env ? atoi(env) : TURBO_THINK_SINK_WIDTH_DEFAULT;
        if (w < 1) w = 1;
        if (w > TURBO_THINK_SINK_WIDTH_MAX) w = TURBO_THINK_SINK_WIDTH_MAX;
        cached = w;
    }
    return cached;
}

// ─── Host-side buffer management (base range, [0, sink_size)) ────────────────

struct sink_buf_entry {
    half * buf;
    int64_t ne0;
    int     cap;
};

static std::mutex g_sink_mutex;
static std::unordered_map<void *, sink_buf_entry> g_sink_K_bufs;
static std::unordered_map<void *, sink_buf_entry> g_sink_V_bufs;

static half * sink_get_or_alloc(std::unordered_map<void *, sink_buf_entry> & map,
                                void * key, int64_t ne0, int sink_size) {
    auto it = map.find(key);
    if (it != map.end() && it->second.ne0 == ne0 && it->second.cap >= sink_size) {
        return it->second.buf;
    }
    if (it != map.end() && it->second.buf) {
        cudaFree(it->second.buf);
    }
    half * buf = nullptr;
    cudaMalloc(&buf, ne0 * sink_size * sizeof(half));
    cudaMemset(buf, 0, ne0 * sink_size * sizeof(half));
    map[key] = {buf, ne0, sink_size};
    return buf;
}

half * turbo_sink_get_buf(void * tensor_data, int64_t ne0) {
    const int ss = turbo_sink_size();
    if (ss <= 0) return nullptr;
    std::lock_guard<std::mutex> lock(g_sink_mutex);
    return sink_get_or_alloc(g_sink_K_bufs, tensor_data, ne0, ss);
}

half * turbo_sink_get_V_buf(void * tensor_data, int64_t ne0) {
    const int ss = turbo_sink_size();
    if (ss <= 0) return nullptr;
    std::lock_guard<std::mutex> lock(g_sink_mutex);
    return sink_get_or_alloc(g_sink_V_bufs, tensor_data, ne0, ss);
}

half * turbo_sink_lookup_buf(void * tensor_data, int64_t * out_ne0) {
    const int ss = turbo_sink_size();
    if (ss <= 0) return nullptr;
    std::lock_guard<std::mutex> lock(g_sink_mutex);
    auto it = g_sink_K_bufs.find(tensor_data);
    if (it != g_sink_K_bufs.end() && it->second.buf) {
        if (out_ne0) *out_ne0 = it->second.ne0;
        return it->second.buf;
    }
    return nullptr;
}

// ─── Dynamic sink ranges (think-token-driven) ────────────────────────────────

struct dynamic_range {
    int64_t start;
    int     width;
};

// Single global vector of currently-registered dynamic ranges. Bound to the
// single-slot SHIP server config (--parallel 1). Multi-slot deployments would
// need a per-seq registry; documented at the public API.
static std::vector<dynamic_range>            g_dynamic_ranges;

// Revision counter: incremented on every register/clear. Read by the CUDA
// graph layer (graph_update_required) to detect registry mutations between
// graph captures and force recapture so the new range count flows into the
// per-FA cudaMemcpyAsync source buffer. Counters are also exposed to host
// callers via ggml_cuda_turbo_sink_get_anchor_revision.
static std::atomic<int64_t>                  g_anchor_revision{0};

// Diagnostic: how many times the CUDA graph layer observed a revision change
// and forced a graph recapture as a result. Plain non-atomic — incremented
// only from the host-side graph code (single-thread context).
static int64_t                               g_anchor_revision_recaptures = 0;

// Per (tensor_data, range_idx) buffer. range_idx in [1, TURBO_SINK_MAX_RANGES-1]
// (slot 0 is the static base in g_sink_K_bufs). Allocated lazily on first
// capture-time request for that range.
struct dynamic_buf_key {
    void * tensor_data;
    int    range_idx;
    bool operator==(const dynamic_buf_key & o) const {
        return tensor_data == o.tensor_data && range_idx == o.range_idx;
    }
};
struct dynamic_buf_key_hash {
    size_t operator()(const dynamic_buf_key & k) const {
        return std::hash<void*>{}(k.tensor_data) ^ (size_t)k.range_idx * 0x9E3779B97F4A7C15ull;
    }
};
static std::unordered_map<dynamic_buf_key, sink_buf_entry, dynamic_buf_key_hash> g_dynamic_K_bufs;

static half * dynamic_get_or_alloc_K(void * tensor_data, int range_idx, int64_t ne0, int width) {
    dynamic_buf_key key{tensor_data, range_idx};
    auto it = g_dynamic_K_bufs.find(key);
    if (it != g_dynamic_K_bufs.end() && it->second.ne0 == ne0 && it->second.cap >= width) {
        return it->second.buf;
    }
    if (it != g_dynamic_K_bufs.end() && it->second.buf) {
        cudaFree(it->second.buf);
    }
    half * buf = nullptr;
    cudaMalloc(&buf, ne0 * width * sizeof(half));
    cudaMemset(buf, 0, ne0 * width * sizeof(half));
    g_dynamic_K_bufs[key] = {buf, ne0, width};
    return buf;
}

extern "C" {

void ggml_cuda_turbo_register_thinking_anchor(int64_t pos, int width) {
    if (width <= 0) return;
    {
        // Default-OFF opt-in. The mechanism is disabled unless
        // TURBO_THINK_ANCHOR_ENABLE=1 is set. Legacy override
        // TURBO_THINK_ANCHOR_DISABLE=1 still wins (forces off even if enable=1)
        // so existing scripts continue to work. Cached on first access.
        static int gate = -1;  // 0 = disabled, 1 = enabled
        if (gate < 0) {
            const char * e_enable  = getenv("TURBO_THINK_ANCHOR_ENABLE");
            const char * e_disable = getenv("TURBO_THINK_ANCHOR_DISABLE");
            int enabled  = (e_enable  && atoi(e_enable)  != 0) ? 1 : 0;
            int disabled = (e_disable && atoi(e_disable) != 0) ? 1 : 0;
            gate = (enabled && !disabled) ? 1 : 0;
            fprintf(stderr,
                "[turbo] thinking-anchor gate: %s "
                "(TURBO_THINK_ANCHOR_ENABLE=%d, TURBO_THINK_ANCHOR_DISABLE=%d)\n",
                gate ? "ENABLED" : "disabled (default)", enabled, disabled);
        }
        if (!gate) return;
    }
    if (width > TURBO_THINK_SINK_WIDTH_MAX) width = TURBO_THINK_SINK_WIDTH_MAX;

    std::lock_guard<std::mutex> lock(g_sink_mutex);

    // The number of dynamic ranges is bounded by MAX_RANGES - 1 (slot 0 is
    // the static base). Drop new registrations once full; logged once.
    constexpr int kMaxDyn = TURBO_SINK_MAX_RANGES - 1;
    if ((int)g_dynamic_ranges.size() >= kMaxDyn) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr,
                "[turbo] thinking-anchor registry full (%d ranges), dropping additional triggers\n",
                kMaxDyn);
            warned = true;
        }
        return;
    }

    // Coalesce overlapping/touching ranges so two back-to-back <think> tags
    // (e.g. interleaved-thinking models) don't waste slots.
    for (auto & r : g_dynamic_ranges) {
        if (pos >= r.start && pos <= r.start + r.width) {
            const int64_t end = std::max<int64_t>(r.start + r.width, pos + width);
            r.width = (int)std::min<int64_t>(end - r.start, TURBO_THINK_SINK_WIDTH_MAX);
            return;
        }
    }

    g_dynamic_ranges.push_back({pos, width});
    g_anchor_revision.fetch_add(1, std::memory_order_release);
    fprintf(stderr,
        "[turbo] thinking-anchor registered: pos=%lld width=%d (active=%d, rev=%lld)\n",
        (long long)pos, width, (int)g_dynamic_ranges.size(),
        (long long)g_anchor_revision.load(std::memory_order_acquire));
}

void ggml_cuda_turbo_clear_thinking_anchors(void) {
    std::lock_guard<std::mutex> lock(g_sink_mutex);
    if (!g_dynamic_ranges.empty()) {
        g_anchor_revision.fetch_add(1, std::memory_order_release);
    }
    g_dynamic_ranges.clear();
    // Buffers held in g_dynamic_K_bufs are intentionally NOT freed here. They
    // get reused on the next registration for the same (tensor, range_idx);
    // a tensor data-pointer change naturally invalidates and re-allocs them
    // via dynamic_get_or_alloc_K. Avoiding cudaFree under the slot-reset path
    // keeps this hook cheap and CUDA-graph-safe.
}

int ggml_cuda_turbo_thinking_anchor_count(void) {
    std::lock_guard<std::mutex> lock(g_sink_mutex);
    return (int)g_dynamic_ranges.size();
}

// Read the current anchor-registry revision counter. Called by the CUDA
// graph layer (ggml_cuda_graph_update_required) to detect registry mutations
// and force graph recapture. Atomic acquire — pairs with release in
// register/clear.
int64_t ggml_cuda_turbo_sink_get_anchor_revision(void) {
    return g_anchor_revision.load(std::memory_order_acquire);
}

// Diagnostic: how many times the graph layer forced a recapture due to a
// revision change.
int64_t ggml_cuda_turbo_sink_get_recapture_count(void) {
    return g_anchor_revision_recaptures;
}

void ggml_cuda_turbo_sink_note_recapture(void) {
    ++g_anchor_revision_recaptures;
}

} // extern "C"

// Snapshot the currently-active ranges as views. *out_ne0 receives the
// canonical per-row stride (the value the buffers were allocated with).
int turbo_sink_collect_active_ranges_K(
    void * tensor_data,
    turbo_sink_range_view out[TURBO_SINK_MAX_RANGES],
    int64_t * out_ne0) {

    std::lock_guard<std::mutex> lock(g_sink_mutex);

    int n = 0;
    int64_t canonical_ne0 = 0;

    const int base_w = turbo_sink_size();
    if (base_w > 0) {
        auto it = g_sink_K_bufs.find(tensor_data);
        if (it != g_sink_K_bufs.end() && it->second.buf) {
            canonical_ne0 = it->second.ne0;
            out[n].start = 0;
            out[n].width = base_w;
            out[n].K_buf = it->second.buf;
            ++n;
        }
    }

    for (size_t i = 0; i < g_dynamic_ranges.size() && n < TURBO_SINK_MAX_RANGES; ++i) {
        const int range_idx = (int)i + 1;  // slot 0 is base
        dynamic_buf_key key{tensor_data, range_idx};
        auto it = g_dynamic_K_bufs.find(key);
        const half * buf = nullptr;
        if (it != g_dynamic_K_bufs.end() && it->second.buf) {
            buf = it->second.buf;
            if (canonical_ne0 == 0) canonical_ne0 = it->second.ne0;
        }
        out[n].start = g_dynamic_ranges[i].start;
        out[n].width = g_dynamic_ranges[i].width;
        out[n].K_buf = buf;
        ++n;
    }

    if (out_ne0) *out_ne0 = canonical_ne0;
    return n;
}

// ─── Diagnostic counters (per-process, sampled to keep atomics cheap) ────────

static __device__ unsigned long long d_turbo_sink_capture_writes = 0;

// Process-global host accumulator updated by the FA-vec dispatcher (per-TU
// d_fattn_sink_hits is read back and zeroed here under env TURBO_SINK_DIAG=1).
// Plain non-static C global — defined here, declared extern in fattn-vec.cuh,
// resolved at host link time. No CUDA RDC required.
unsigned long long g_fattn_sink_hits_accum[TURBO_SINK_MAX_RANGES] = { 0 };

extern "C" {

unsigned long long ggml_cuda_turbo_sink_get_capture_writes(void) {
    unsigned long long h = 0;
    cudaMemcpyFromSymbol(&h, d_turbo_sink_capture_writes, sizeof(h), 0, cudaMemcpyDeviceToHost);
    return h;
}

unsigned long long ggml_cuda_turbo_sink_get_fa_hits(int range_idx) {
    if (range_idx < 0 || range_idx >= TURBO_SINK_MAX_RANGES) return 0;
    return g_fattn_sink_hits_accum[range_idx];
}

void ggml_cuda_turbo_sink_reset_diagnostics(void) {
    unsigned long long zero = 0;
    cudaMemcpyToSymbol(d_turbo_sink_capture_writes, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice);
    for (int i = 0; i < TURBO_SINK_MAX_RANGES; ++i) g_fattn_sink_hits_accum[i] = 0;
}

} // extern "C"

// ─── Capture kernel: WHT-rotate + store fp16 for sink positions ───────────────
//
// Generalized to write into ANY range [range_start, range_start+range_width).
// Buffer offset = (dst_row - range_start) instead of dst_row.

template <typename idx_t, int GROUP_SIZE>
static __global__ void k_turbo_sink_capture(
    const float * __restrict__ src0,
    const idx_t * __restrict__ src1,
    half * __restrict__ sink_buf,
    const int64_t ne00,
    const int64_t ne01,
    const int64_t ne11,
    const int64_t s01,
    const int64_t s02,
    const int64_t s03,
    const int64_t ne12,
    const int64_t s10,
    const int64_t s11,
    const int64_t s12,
    const int64_t nb1,
    const int64_t nb2,
    const int64_t nb3,
    const int64_t range_start,
    const int     range_width) {

    const int j = threadIdx.x;
    const int64_t n_groups_per_row = ne00 / GROUP_SIZE;
    const int64_t g = blockIdx.x;
    const int64_t i_grp = g % n_groups_per_row;
    int64_t       tmp   = g / n_groups_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i10 = i01;
    const int64_t i11 = i01 % ne11;
    const int64_t i12 = i02;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    if (dst_row < range_start || dst_row >= range_start + range_width) return;
    const int64_t buf_row = dst_row - range_start;

    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;

    __shared__ float x[GROUP_SIZE];
    x[j] = src_row[i_grp * GROUP_SIZE + j];
    __syncthreads();

    if (d_innerq_active) {
        x[j] *= d_innerq_scale[j];
    }
    __syncthreads();

    // L2 norm
    constexpr int n_warps = GROUP_SIZE / 32;
    __shared__ float warp_accum[n_warps];
    float v2 = x[j] * x[j];
    for (int offset = 16; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    if (j % 32 == 0) warp_accum[j / 32] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float inv_norm = (s_norm_sq > 1e-20f) ? rsqrtf(s_norm_sq) : 0.0f;
    const float grp_norm = s_norm_sq * inv_norm;

    x[j] *= inv_norm;
    __syncthreads();

    // Forward WHT
    if (GROUP_SIZE == 128) { x[j] *= TURBO_WHT_SIGNS1[j]; }
    else                   { x[j] *= TURBO_WHT_SIGNS1_64[j]; }
    __syncthreads();

#define WHT_STAGE(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE(1) WHT_STAGE(2) WHT_STAGE(4) WHT_STAGE(8)
    WHT_STAGE(16) WHT_STAGE(32)
    if (GROUP_SIZE == 128) { WHT_STAGE(64) }
#undef WHT_STAGE

    constexpr float inv_sqrt_group = (GROUP_SIZE == 128) ? 0.08838834764831845f : 0.125f;
    if (GROUP_SIZE == 128) { x[j] = x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2[j]; }
    else                   { x[j] = x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2_64[j]; }
    __syncthreads();

    // Store WHT-rotated value at full fp16 precision into the per-range buffer
    const int64_t global_col = i_grp * GROUP_SIZE + j;
    sink_buf[buf_row * ne00 + global_col] = __float2half(x[j] * grp_norm);

    // Diagnostic: count once per capture-block (one row written) using thread 0.
    if (j == 0 && i_grp == 0) {
        atomicAdd(&d_turbo_sink_capture_writes, 1ULL);
    }
}

template<typename idx_t>
void turbo_sink_capture_turbo3_impl(
    const float * src0,
    const idx_t * src1,
    void * dst_data,
    int64_t ne00, int64_t ne01, int64_t ne11,
    int64_t s01, int64_t s02, int64_t s03,
    int64_t ne12, int64_t ne13,
    int64_t s10, int64_t s11, int64_t s12,
    int64_t nb1, int64_t nb2, int64_t nb3,
    int group_size,
    cudaStream_t stream) {

    const int ss        = turbo_sink_size();
    const int n_dyn_now = ggml_cuda_turbo_thinking_anchor_count();

    if (ss <= 0 && n_dyn_now == 0) return;

    const int64_t n_groups_per_row = ne00 / group_size;
    const int64_t ne_total = n_groups_per_row * ne01 * ne12 * ne13;

    auto launch = [&](half * buf, int64_t range_start, int range_width) {
        if (group_size == 128) {
            k_turbo_sink_capture<idx_t, 128><<<(int)ne_total, 128, 0, stream>>>(
                src0, src1, buf, ne00, ne01, ne11,
                s01, s02, s03, ne12, s10, s11, s12,
                nb1, nb2, nb3, range_start, range_width);
        } else {
            k_turbo_sink_capture<idx_t, 64><<<(int)ne_total, 64, 0, stream>>>(
                src0, src1, buf, ne00, ne01, ne11,
                s01, s02, s03, ne12, s10, s11, s12,
                nb1, nb2, nb3, range_start, range_width);
        }
    };

    // Base range [0, ss). Existing behavior.
    if (ss > 0) {
        half * buf = turbo_sink_get_buf(dst_data, ne00);
        if (buf) {
            launch(buf, /*range_start*/ 0, /*range_width*/ ss);
        }
    }

    // Dynamic ranges. Each gets its own buffer slot, allocated lazily here.
    if (n_dyn_now > 0) {
        // Snapshot under lock, then dispatch outside the critical section
        // (cudaMallocs above are already inside the helper's mutex acquire).
        struct dyn_dispatch { int64_t start; int width; half * buf; };
        dyn_dispatch dispatches[TURBO_SINK_MAX_RANGES - 1];
        int n_dispatches = 0;
        {
            std::lock_guard<std::mutex> lock(g_sink_mutex);
            for (size_t i = 0; i < g_dynamic_ranges.size() && i < (size_t)(TURBO_SINK_MAX_RANGES - 1); ++i) {
                const int range_idx = (int)i + 1;
                half * buf = dynamic_get_or_alloc_K(dst_data, range_idx, ne00, g_dynamic_ranges[i].width);
                dispatches[n_dispatches++] = {
                    g_dynamic_ranges[i].start,
                    g_dynamic_ranges[i].width,
                    buf,
                };
            }
        }
        for (int i = 0; i < n_dispatches; ++i) {
            if (dispatches[i].buf) {
                launch(dispatches[i].buf, dispatches[i].start, dispatches[i].width);
            }
        }
    }
}

// Explicit template instantiations
template void turbo_sink_capture_turbo3_impl<int32_t>(const float*, const int32_t*, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, cudaStream_t);
template void turbo_sink_capture_turbo3_impl<int64_t>(const float*, const int64_t*, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, cudaStream_t);

// ─── Not used yet — device state is per-TU in fattn-common.cuh ───────────────

void turbo_sink_set_device_state(
    const half * K_buf, const half * V_buf,
    int sink_size, int64_t ne0,
    cudaStream_t stream) {
    // Device state is set directly in fattn-common.cuh via static __device__ variables.
    // This function is a placeholder for future graph-compatible implementation.
    (void)K_buf; (void)V_buf; (void)sink_size; (void)ne0; (void)stream;
}
