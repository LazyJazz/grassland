#pragma once
#include "grassland/d3d12/core/device.h"

namespace grassland::d3d12 {
class CommandAllocator {
 public:
  explicit CommandAllocator(
      const Device &device,
      D3D12_COMMAND_LIST_TYPE list_type = D3D12_COMMAND_LIST_TYPE_DIRECT);
  ~CommandAllocator();

  [[nodiscard]] ComPtr<ID3D12CommandAllocator> Ptr() const;

 private:
  ComPtr<ID3D12CommandAllocator> command_allocator_;
};
}  // namespace grassland::d3d12
