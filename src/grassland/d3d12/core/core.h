#pragma once
#include "grassland/d3d12/core/adaptor.h"
#include "grassland/d3d12/core/device.h"
#include "grassland/d3d12/core/dxgi_factory.h"
#include "grassland/d3d12/utils/d3d12_utils.h"

namespace grassland::d3d12 {

struct CoreSettings {
  HWND hwnd{nullptr};
  bool enable_debug_layer{kDefaultEnableDebugLayer};
  bool enable_ray_tracing{false};
  int device_index{-1};
};

class Core {
 public:
  explicit Core(const CoreSettings &settings);
  ~Core();

 private:
  std::unique_ptr<DXGIFactory> factory_;
};
}  // namespace grassland::d3d12
