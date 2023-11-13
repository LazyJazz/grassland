#pragma once
#include "grassland/d3d12/core/dxgi_factory.h"

namespace grassland::d3d12 {
class Adaptor {
 public:
  Adaptor();
  ~Adaptor();

  ComPtr<IDXGIAdapter1> Ptr() const;

 private:
  ComPtr<IDXGIAdapter1> adaptor_;
};
}  // namespace grassland::d3d12
