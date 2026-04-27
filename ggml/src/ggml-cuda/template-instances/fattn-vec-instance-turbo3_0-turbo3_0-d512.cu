// TurboQuant3 symmetric vec FA, split by head dimension to keep sm_120 ptxas optimized.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE_NO_SOFTCAP(512, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
