#pragma once
#include "grassland/d3d12/core/dxgi_factory.h"

namespace grassland::d3d12 {
class Device {
 public:
  Device();
  ~Device();

  ComPtr<ID3D12Device> Ptr() const;

 private:
  ComPtr<ID3D12Device> device_;
};
}  // namespace grassland::d3d12
