//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#ifdef WITH_LIKWID
#include <likwid.h>
#else

#define LIKWID_MARKER_INIT


#define LIKWID_MARKER_THREADINIT

#define LIKWID_MARKER_CLOSE

#define LIKWID_MARKER_START(opt)

#define LIKWID_MARKER_STOP(opt)

#endif

#ifdef WITH_NVTX
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#define NVTX_MARKER_INIT cudaProfilerStart()
#define NVTX_MARKER_CLOSE cudaProfilerStop()
#define NVTX_MARKER_START(opt) nvtxRangePushA(opt)
#define NVTX_MARKER_STOP(opt) nvtxRangePop()

#else

#define NVTX_MARKER_INIT

#define NVTX_MARKER_CLOSE

#define NVTX_MARKER_START(opt)

#define NVTX_MARKER_STOP(opt)

#endif

#define LSAN_DISABLE

#define LSAN_ENABLE

#if defined(__clang__) && defined(DEBUG)
#if __has_feature(address_sanitizer)
#include <sanitizer/lsan_interface.h>
#undef LSAN_DISABLE
#define LSAN_DISABLE __lsan_disable();
#undef LSAN_ENABLE
#define LSAN_ENABLE __lsan_enable();
#endif
#endif
