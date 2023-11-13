#pragma once
#include "grassland/d3d12/utils/d3d12_utils.h"

namespace grassland::d3d12 {

struct DXGIFactorySettings {
  bool enable_debug_layer = kDefaultEnableDebugLayer;
};

class DXGIFactory {
 public:
  explicit DXGIFactory(const DXGIFactorySettings &settings = {});
  ~DXGIFactory();

  [[nodiscard]] ComPtr<IDXGIFactory4> Ptr() const;

 private:
  ComPtr<IDXGIFactory4> factory_;
};
}  // namespace grassland::d3d12
