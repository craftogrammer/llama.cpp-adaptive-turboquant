// TurboQuant3 TCQ K + q8_0 V prefill-only CUDA flash attention vec kernel.
// Kept separate from the decode TU so optimized ptxas only sees the Qwen decode
// path; this prefill path can remain at -O0 if sm_120 ptxas still crashes on it.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE_PREFILL_NO_SOFTCAP(256, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0);
