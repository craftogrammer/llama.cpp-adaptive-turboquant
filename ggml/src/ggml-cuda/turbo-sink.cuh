#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Hard-cap on simultaneously-active fp16 sink ranges per tensor.
// Slot 0 is the static `[0, TURBO_SINK_SIZE)` base range; slots 1..MAX-1 are
// dynamic ranges registered at runtime by the server's reasoning-trigger hook.
// Bumping this raises a fixed per-FA-call upload cost (a few dozen bytes) and
// per-K-position attention check cost (one extra compare per slot, fully
// unrolled). 4 covers a single conversation's first three reasoning blocks at
// SHIP scale; raise if longer chats need more anchors.
#define TURBO_SINK_MAX_RANGES 4

// Default width of a dynamic anchor range, in KV positions. Override with
// env TURBO_THINK_SINK_WIDTH (clamped to [1, 64] internally).
#define TURBO_THINK_SINK_WIDTH_DEFAULT 8

int turbo_sink_size();
int turbo_think_sink_width();

half * turbo_sink_get_buf(void * tensor_data, int64_t ne0);
half * turbo_sink_get_V_buf(void * tensor_data, int64_t ne0);
half * turbo_sink_lookup_buf(void * tensor_data, int64_t * out_ne0);

template<typename idx_t>
void turbo_sink_capture_turbo3_impl(
    const float * src0, const idx_t * src1, void * dst_data,
    int64_t ne00, int64_t ne01, int64_t ne11,
    int64_t s01, int64_t s02, int64_t s03,
    int64_t ne12, int64_t ne13,
    int64_t s10, int64_t s11, int64_t s12,
    int64_t nb1, int64_t nb2, int64_t nb3,
    int group_size, cudaStream_t stream);

void turbo_sink_set_device_state(
    const half * K_buf, const half * V_buf,
    int sink_size, int64_t ne0,
    cudaStream_t stream);

// ─── Dynamic (think-token) sink ranges ────────────────────────────────────────
//
// Server-side trigger hook: when the chat sampler emits the model's reasoning
// start tag (e.g. "<think>"), call register_thinking_anchor(pos, width) where
// pos == slot.prompt.tokens.size() — the KV position the next sampled token's
// K/V will be written to. The next `width` writes per K cache tensor will be
// captured at fp16 alongside the TCQ encoding, and the FA-vec read path will
// route those positions through the fp16 fast-path on every subsequent decode.
//
// The trigger is cumulative: each call adds one range up to TURBO_SINK_MAX_RANGES-1
// dynamic ranges (range slot 0 is reserved for the static base [0, sink_size)).
// Once full, additional registrations are dropped (logged once). Cleared on
// slot reset / KV cache rotation via clear_thinking_anchors().
//
// Buffer storage is per (tensor, range_idx) — allocated lazily on first capture
// for that range. Lifetime: until clear_thinking_anchors() OR the underlying
// tensor's data pointer is replaced (we key on tensor_data and orphan stale
// entries; a fresh KV cache allocation evicts the old map entry).
//
// Thread-safety: all entry points lock g_sink_mutex internally. Only host-side
// state; per-FA-call upload to device is the existing cudaMemcpyAsync path.
//
// Public C trigger ABI lives in ggml/include/ggml-turbo-sink-trigger.h so that
// server-context.cpp can include it without CUDA dependencies.
#include "ggml-turbo-sink-trigger.h"

// Internal helper used by FA dispatch. Snapshots all active sink ranges
// (base + dynamic) for the given K-cache tensor data pointer. Returns the
// number of populated slots; *out_ne0 receives the canonical per-row stride
// (the value the buffers were allocated with at KV-write time).
//
// A range slot is included even if its K_buf is nullptr (e.g. dynamic range
// registered but not yet captured) — the FA-vec read path treats null bufs
// as inactive (NaN-safe). This guarantees device-side range count matches
// the host registry across multi-step decode where capture and read interleave.
struct turbo_sink_range_view {
    int64_t      start;
    int          width;
    const half * K_buf;
};
int turbo_sink_collect_active_ranges_K(
    void * tensor_data,
    turbo_sink_range_view out[TURBO_SINK_MAX_RANGES],
    int64_t * out_ne0);
