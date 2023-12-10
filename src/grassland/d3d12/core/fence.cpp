#include "grassland/d3d12/core/fence.h"

namespace grassland::d3d12 {
Fence::Fence(const Device &device) {
  ThrowIfFailed(device.Ptr()->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                          IID_PPV_ARGS(&fence_)),
                "Failed to create fence");
}

Fence::~Fence() = default;

ComPtr<ID3D12Fence> Fence::Ptr() const {
  return fence_;
}

}  // namespace grassland::d3d12
