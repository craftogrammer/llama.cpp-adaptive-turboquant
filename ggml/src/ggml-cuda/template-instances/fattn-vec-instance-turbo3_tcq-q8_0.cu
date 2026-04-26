// TurboQuant3 TCQ K + q8_0 V mixed pair CUDA flash attention vec kernel instantiation.
// Required by TURBO_LAYER_ADAPTIVE modes (e.g. mode 13) when V is promoted to q8_0
// while K stays at turbo3_tcq.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE_DECODE_NO_SOFTCAP(256, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0);
