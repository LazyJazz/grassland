#pragma once
#include "grassland/d3d12/core/device.h"

namespace grassland::d3d12 {
class Fence {
 public:
  explicit Fence(const Device &device);
  ~Fence();

  [[nodiscard]] ComPtr<ID3D12Fence> Ptr() const;

 private:
  ComPtr<ID3D12Fence> fence_{};
};
}  // namespace grassland::d3d12
