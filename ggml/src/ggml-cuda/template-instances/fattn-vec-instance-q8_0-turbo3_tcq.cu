// q8_0 K + TurboQuant3 TCQ V mixed pair CUDA flash attention vec kernel instantiation.
// Symmetric to fattn-vec-instance-turbo3_tcq-q8_0.cu.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_TCQ);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_TCQ);
