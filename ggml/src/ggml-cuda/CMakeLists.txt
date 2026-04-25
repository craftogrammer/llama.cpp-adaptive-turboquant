cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found")

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # native == GPUs available at build time
        # 50     == Maxwell, lowest CUDA 12 standard
        # 60     == P100, FP16 CUDA intrinsics
        # 61     == Pascal, __dp4a instruction (per-byte integer dot product)
        # 70     == V100, FP16 tensor cores
        # 75     == Turing, int8 tensor cores
        # 80     == Ampere, asynchronous data loading, faster tensor core instructions
        # 86     == RTX 3000, needs CUDA v11.1
        # 89     == RTX 4000, needs CUDA v11.8
        # 120    == Blackwell, needs CUDA v12.8, FP4 tensor cores
        #
        # XX-virtual == compile CUDA code as PTX, do JIT compilation to binary code on first run
        # XX-real    == compile CUDA code as device code for this specific architecture
        # no suffix  == compile as both PTX and device code
        #
        # The default behavior for a non-native is to build virtual architectures as needed to cover all features needed
        #     for best performance and to also build real architectures for the most commonly used GPUs.
        if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES "native")
        else()
            if (CUDAToolkit_VERSION VERSION_LESS "13")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 50-virtual 61-virtual 70-virtual)
            endif ()

            list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual 80-virtual 86-real)

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real)
            endif()

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
                # The CUDA architecture 120f-virtual would in principle work for Blackwell support
                #     but the newly added "f" suffix conflicted with a preexising regex for validating CUDA architectures in CMake.
                # So either a recent CMake version or one with the backported fix is needed.
                # The following versions should work:
                #   - CMake >= v3.31.8 && CMake < v4.0.0
                #   - CMake >= v4.0.2
                # This is NOT documented in the CMake release notes,
                #     check Modules/Internal/CMakeCUDAArchitecturesValidate.cmake in the CMake git repository instead.
                # However, the architectures 120a-real and 121a-real should work with basically any CMake version and
                #     until the release of e.g. Rubin there is no benefit to shipping virtual architectures for Blackwell.
                list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 121a-real)
            endif()
        endif()
    endif()

    enable_language(CUDA)

    # TODO: Remove once CCCL 3.2 has been released and bundled with CUDA Toolkit
    if (GGML_CUDA_CUB_3DOT2)
        include(FetchContent)

        FetchContent_Declare(
            CCCL
            GIT_REPOSITORY https://github.com/nvidia/cccl.git
            GIT_TAG        v3.2.0
            GIT_SHALLOW    TRUE
        )

        FetchContent_MakeAvailable(CCCL)
    endif()

    # Preserve explicitly requested Blackwell targets. Desktop RTX 50-series GPUs
    # are sm_120, not sm_120a; forcing 120a can select unsupported arch-specific
    # instructions. Only default/native arch lists should contain 120a.
    foreach(ARCHS IN ITEMS CMAKE_CUDA_ARCHITECTURES CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(FIXED_ARCHS "")
        foreach(ARCH IN LISTS ${ARCHS})
            if (ARCH MATCHES "^12[0-9]$")
                string(REGEX REPLACE "^(12[0-9])((-real|-virtual)?)$" "\\1a\\2" FIXED_ARCH ${ARCH})
                message(STATUS "Replacing ${ARCH} in ${ARCHS} with ${FIXED_ARCH}")
                list(APPEND FIXED_ARCHS "${FIXED_ARCH}")
            else()
                list(APPEND FIXED_ARCHS "${ARCH}")
            endif()
        endforeach()
        set(${ARCHS} ${FIXED_ARCHS})
    endforeach()

    # If we try to compile a "native" build it will use the 12X architectures and fail.
    # So we should instead use the native architectures as determined by CMake after replacing 12X with 12Xa.
    # But if at the time of the build no GPUs are connected at all CMAKE_CUDA_ARCHITECTURES will contain garbage that we should not use.
    if (CMAKE_CUDA_ARCHITECTURES STREQUAL "native" AND CMAKE_CUDA_ARCHITECTURES_NATIVE MATCHES "^[0-9]+(a|f)?(-real|-virtual)?(;[0-9]+(a|f)?(-real|-virtual)?|;)*$")
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
    endif()
    message(STATUS "Using CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} CMAKE_CUDA_ARCHITECTURES_NATIVE=${CMAKE_CUDA_ARCHITECTURES_NATIVE}")

    file(GLOB   GGML_HEADERS_CUDA "*.cuh")
    list(APPEND GGML_HEADERS_CUDA "../../include/ggml-cuda.h")

    file(GLOB   GGML_SOURCES_CUDA "*.cu")
    file(GLOB   SRCS "template-instances/fattn-tile*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/fattn-mma*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmq*.cu")
    if (GGML_CUDA_NO_MXFP4)
        list(FILTER SRCS EXCLUDE REGEX "mmq-instance-mxfp4\\.cu$")
        add_compile_definitions(GGML_CUDA_NO_MXFP4)
    endif()
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmf*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})

    if (GGML_CUDA_FA_ALL_QUANTS)
        file(GLOB   SRCS "template-instances/fattn-vec*.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
    else()
        list(APPEND GGML_SOURCES_CUDA
            template-instances/fattn-vec-instance-f16-f16.cu
            template-instances/fattn-vec-instance-q4_0-q4_0.cu
            template-instances/fattn-vec-instance-q8_0-q8_0.cu
            # Turbo3 symmetric (primary target for --cache-type-k turbo3 --cache-type-v turbo3)
            template-instances/fattn-vec-instance-turbo3_0-turbo3_0.cu
            # Mixed turbo3_0 x q8_0: required for TURBO_LAYER_ADAPTIVE modes that
            # promote per-layer V (or K) to q8_0 while keeping the other side at turbo3.
            template-instances/fattn-vec-instance-turbo3_0-q8_0.cu
            template-instances/fattn-vec-instance-q8_0-turbo3_0.cu
            # TCQ variant: trellis-coded 3-bit, 3.25 bpv (+4% vs turbo3_0), potentially
            # better quality at similar size. Enables --cache-type-k/v turbo3_tcq.
            template-instances/fattn-vec-instance-turbo3_tcq-turbo3_tcq.cu
            # TCQ x q8_0 mixed pairs: enables LAYER_ADAPTIVE modes alongside TCQ KV.
            # Gemma 4 garbage output (spiritbuun issue #12) is head_dim=512 specific
            # (FWHT-128 mismatch); doesn't apply to Qwen3.6's head_dim=256.
            template-instances/fattn-vec-instance-turbo3_tcq-q8_0.cu
            template-instances/fattn-vec-instance-q8_0-turbo3_tcq.cu
            # turbo2_0: 2-bit scalar quant, ~7.53x compression. Per fork README, fastest
            # at every depth on every tested GPU. Boundary V auto-activates (mode 12) for
            # turbo2-V via llama-kv-cache.cpp. Worth trying for higher-context workloads.
            template-instances/fattn-vec-instance-turbo2_0-turbo2_0.cu
            # turbo2_0 x q8_0 mixed pairs: required for auto-mode-12 to actually fire
            # without aborting. Without these instances, llama-kv-cache.cpp:212's
            # boundary-V-to-q8_0 promotion crashes when dispatched. With them, V on
            # first/last attention layers gets the wider q8_0 representation, while
            # interior V layers stay at 2-bit turbo2.
            template-instances/fattn-vec-instance-turbo2_0-q8_0.cu
            template-instances/fattn-vec-instance-q8_0-turbo2_0.cu
            # EXCLUDED: bf16 template instances cause ptxas ACCESS_VIOLATION on SM120 with CUDA 12.8.2.
            # turbo4_0 also excluded — known NaN regression on Qwen3.5-9B (issue #60),
            # garbled output on SM120 (issue #56), SET_ROWS hardcoded for turbo3 packing
            # (issue #29). TCQ x q8_0 mixed pairs blocked per spiritbuun PR #1 (cross-type
            # crashes / garbage output, issue #12).
            # - fattn-vec-instance-bf16-*.cu (all 7 files)
            # - fattn-vec-instance-*-bf16.cu (all 6 files)
        )
    endif()

    # Windows ptxas ACCESS_VIOLATION workaround (SM120 / CUDA 12.9):
    # - q4_0 K/V helpers in fattn-common.cuh are marked __noinline__. That alone
    #   lets this file build cleanly at full -O3.
    # - turbo3_0 + turbo3_tcq TUs fall back to -O0 (heavy dequant bodies that
    #   ptxas's analyzer chokes on even with __noinline__ on helpers).
    # - Benchmarked: at -O0 on turbo3_0 TU we get ~36 t/s decode on Qwen3.6-27B
    #   at 140k context — matches original baseline, so no further tuning needed.
    set_source_files_properties(
        # fattn.cu itself is the vec FA dispatch table. With many template specializations
        # (turbo3, turbo2, TCQ, mixed pairs) the analyzer footprint exceeds Windows ptxas
        # capacity at -O3 -> ACCESS_VIOLATION. -O0 here is essentially free at runtime
        # (this TU is only routing logic, not the kernel hot path).
        fattn.cu
        template-instances/fattn-vec-instance-turbo3_0-turbo3_0.cu
        template-instances/fattn-vec-instance-turbo3_tcq-turbo3_tcq.cu
        # Asymmetry: K=turbo3_0 x V=q8_0 crashes ptxas (KQ-side analyzer weight),
        # but K=q8_0 x V=turbo3_0 builds fine at -O3. So only the turbo3_0-q8_0
        # direction needs the -O0 fallback; q8_0-turbo3_0 stays optimized.
        template-instances/fattn-vec-instance-turbo3_0-q8_0.cu
        # turbo2_0: preemptive -O0 — body is similar weight to turbo3_0 (4 centroid
        # lookups, bit-shift decode), so ptxas analyzer is likely to crash at -O3.
        template-instances/fattn-vec-instance-turbo2_0-turbo2_0.cu
        # TCQ x q8_0 mixed pairs: preemptive -O0 — TCQ K side carries the heavy
        # 9-bit codebook lookup, same analyzer pressure as turbo3_0 x q8_0 which
        # already needs -O0. q8_0 x turbo3_tcq direction may be safe at -O3 (q8_0 K
        # path is light), but include it preemptively; can drop if it builds clean.
        template-instances/fattn-vec-instance-turbo3_tcq-q8_0.cu
        template-instances/fattn-vec-instance-q8_0-turbo3_tcq.cu
        # turbo2_0 x q8_0 mixed pairs: preemptive -O0 — turbo2_0 K-side analyzer
        # weight matches turbo2_0-turbo2_0 (which is already -O0). The q8_0-turbo2_0
        # direction may be safe at -O3 (q8_0 K path is light, turbo3_0-q8_0's reverse
        # builds fine without it); include it preemptively, can drop if clean.
        template-instances/fattn-vec-instance-turbo2_0-q8_0.cu
        template-instances/fattn-vec-instance-q8_0-turbo2_0.cu
        PROPERTIES COMPILE_FLAGS "--ptxas-options=-O0")

    ggml_add_backend_library(ggml-cuda
                             ${GGML_HEADERS_CUDA}
                             ${GGML_SOURCES_CUDA}
                            )

    add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})

    if (GGML_CUDA_GRAPHS)
        add_compile_definitions(GGML_CUDA_USE_GRAPHS)
    endif()

    if (GGML_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_CUDA_FORCE_MMQ)
    endif()

    if (GGML_CUDA_FORCE_CUBLAS)
        add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
    endif()

    if (GGML_CUDA_NO_VMM)
        add_compile_definitions(GGML_CUDA_NO_VMM)
    endif()

    if (NOT GGML_CUDA_FA)
        add_compile_definitions(GGML_CUDA_NO_FA)
    endif()

    if (GGML_CUDA_NO_PEER_COPY)
        add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
    endif()

    if (GGML_STATIC)
        if (WIN32)
            # As of 12.3.1 CUDA Toolkit for Windows does not offer a static cublas library
            target_link_libraries(ggml-cuda PRIVATE CUDA::cudart_static CUDA::cublas)
        else ()
            if (GGML_CUDA_CUB_3DOT2)
                target_link_libraries(ggml-cuda PRIVATE  CCCL::CCCL)
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.1")
                target_link_libraries(ggml-cuda PRIVATE  CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            else()
                target_link_libraries(ggml-cuda PRIVATE  CUDA::cudart_static CUDA::cublas_static)
            endif()
        endif()
    else()
        if (GGML_CUDA_CUB_3DOT2)
            target_link_libraries(ggml-cuda PRIVATE  CCCL::CCCL)
        endif()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cudart CUDA::cublas)
    endif()

    if (GGML_CUDA_NO_VMM)
        # No VMM requested, no need to link directly with the cuda driver lib (libcuda.so)
    else()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cuda_driver)
    endif()

    set(CUDA_CXX_FLAGS "")

    set(CUDA_FLAGS -use_fast_math -extended-lambda)

    if (GGML_CUDA_DEBUG)
        list(APPEND CUDA_FLAGS -lineinfo)
        add_compile_definitions(GGML_CUDA_DEBUG)
    endif()

    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        # Options are:
        # - none (not recommended)
        # - speed (nvcc's default)
        # - balance
        # - size
        list(APPEND CUDA_FLAGS -compress-mode=${GGML_CUDA_COMPRESSION_MODE})
    endif()

    if (GGML_FATAL_WARNINGS)
        list(APPEND CUDA_FLAGS -Werror all-warnings)
    endif()

    if (GGML_ALL_WARNINGS AND NOT MSVC)
        set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
        if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
            list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
        endif()

        execute_process(
            COMMAND ${NVCC_CMD} -Xcompiler --version
            OUTPUT_VARIABLE CUDA_CCFULLVER
            ERROR_QUIET
        )

        if (NOT CUDA_CCFULLVER MATCHES clang)
            set(CUDA_CCID "GNU")
            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                OUTPUT_VARIABLE CUDA_CCVER
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()

        message(STATUS "CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

        ggml_get_flags(${CUDA_CCID} ${CUDA_CCVER})
        list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # This is passed to -Xcompiler later
    endif()

    if (NOT MSVC)
        list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
    else()
        # CCCL 3.2 onwards will require a cpp-standard-compliant preprocessor for MSVC
        # https://github.com/NVIDIA/cccl/pull/6827
        list(APPEND CUDA_CXX_FLAGS /Zc:preprocessor)
    endif()

    list(JOIN   CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument

    if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
        list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
    endif()

    target_compile_options(ggml-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
else()
    message(FATAL_ERROR "CUDA Toolkit not found")
endif()
