#pragma once
#include "grassland/d3d12/core/device.h"

namespace grassland::d3d12 {

struct SwapChainSettings {
  HWND hwnd{nullptr};
  int frame_count{3};
};

class SwapChain {
 public:
  SwapChain(const DXGIFactory &factory, const SwapChainSettings &settings);
  ~SwapChain();

  [[nodiscard]] ComPtr<IDXGISwapChain3> Ptr() const;

 private:
  ComPtr<IDXGISwapChain3> swap_chain_;
  Device device_;
  SwapChainSettings settings_;
};
}  // namespace grassland::d3d12
