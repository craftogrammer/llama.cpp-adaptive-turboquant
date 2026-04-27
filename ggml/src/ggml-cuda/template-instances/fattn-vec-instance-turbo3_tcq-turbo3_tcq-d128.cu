// TurboQuant3 TCQ symmetric vec FA, split by head dimension for sm_120 ptxas.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE_NO_SOFTCAP(128, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ);
