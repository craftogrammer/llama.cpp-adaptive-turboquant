// Mixed KV: turbo3 K + q8_0 V, split by head dimension for sm_120 ptxas.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE_NO_SOFTCAP(64, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
