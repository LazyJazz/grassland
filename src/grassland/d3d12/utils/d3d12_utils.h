#pragma once

#include <d3d12.h>
#include <d3dcompiler.h>
#include <d3dx12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include <stdexcept>

#include "grassland/util/util.h"

namespace grassland::d3d12 {
using Microsoft::WRL::ComPtr;

#ifdef NDEBUG
constexpr bool kDefaultEnableDebugLayer = false;
#else
constexpr bool kDefaultEnableDebugLayer = true;
#endif

inline void ThrowIfFailed(HRESULT hr, const char *message) {
  if (FAILED(hr)) {
    // Output hr in unsigned hex form
    LAND_ERROR("{}: 0x{:X}", message, uint32_t(hr));
  }
}

}  // namespace grassland::d3d12
