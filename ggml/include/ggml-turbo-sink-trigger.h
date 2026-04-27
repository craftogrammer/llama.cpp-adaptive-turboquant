#pragma once

// TurboQuant: server-callable trigger for the CUDA backend's dynamic fp16
// sink-range registry. Allows the server / sampler to mark KV-cache positions
// as anchor tokens (captured at fp16 alongside the TCQ encoding, served via
// the fp16 fast-path on every subsequent attention read).
//
// Exposed as plain-C dllexport from ggml-cuda.dll so the server can call
// without including CUDA headers. On non-CUDA backends the symbols simply
// don't exist and the server-side hook is compiled out via the
// GGML_TURBOQUANT_CUDA_LINKED define (set by tools/server/CMakeLists.txt
// when ggml-cuda is part of the build).
//
// Position semantics:
//   pos == the KV-cache slot index where the NEXT sampled token's K/V will
//   be written. For the typical reasoning trigger, this is `slot.prompt.tokens.size()`
//   captured at the moment the literal "<think>" tag first appears in the
//   model's accumulated generation.
//
// width == number of consecutive positions [pos, pos+width) to anchor at fp16.
//   Hard-clamped inside the registry to [1, 64].
//
// Coalescing: a registration whose [pos, pos+width) overlaps an existing
// dynamic range is merged in-place (no new range slot consumed). Up to
// TURBO_SINK_MAX_RANGES-1 = 3 disjoint dynamic ranges per session.
//
// Thread-safety: all entry points lock an internal mutex.

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API void ggml_cuda_turbo_register_thinking_anchor(int64_t pos, int width);
GGML_BACKEND_API void ggml_cuda_turbo_clear_thinking_anchors(void);
GGML_BACKEND_API int  ggml_cuda_turbo_thinking_anchor_count(void);

// Diagnostic counters (per-process). Reset via ggml_cuda_turbo_sink_reset_diagnostics().
// `range_idx` 0 = base [0, sink_size); 1..MAX-1 = dynamic ranges in registration order.
// Returns 0 if range_idx is out of bounds.
GGML_BACKEND_API unsigned long long ggml_cuda_turbo_sink_get_capture_writes(void);
GGML_BACKEND_API unsigned long long ggml_cuda_turbo_sink_get_fa_hits(int range_idx);
GGML_BACKEND_API void               ggml_cuda_turbo_sink_reset_diagnostics(void);

// Revision counter for the dynamic anchor registry. Read by the CUDA graph
// layer to detect mutations between captures and force graph recapture.
// Increments on every register/clear that mutates the registry.
GGML_BACKEND_API int64_t ggml_cuda_turbo_sink_get_anchor_revision(void);

// Diagnostic: how many times graph recapture has been forced by a revision
// change since process start (or since reset_diagnostics).
GGML_BACKEND_API int64_t ggml_cuda_turbo_sink_get_recapture_count(void);

// Called by the CUDA graph layer when it forces a recapture due to a
// revision change. Increments the recapture diagnostic counter.
GGML_BACKEND_API void    ggml_cuda_turbo_sink_note_recapture(void);

#ifdef  __cplusplus
}
#endif
