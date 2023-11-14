#pragma once
#include "grassland/d3d12/core/command_allocator.h"

namespace grassland::d3d12 {
class CommandQueue {
 public:
  explicit CommandQueue(
      const Device &device,
      D3D12_COMMAND_LIST_TYPE list_type = D3D12_COMMAND_LIST_TYPE_DIRECT);
  ~CommandQueue();

  [[nodiscard]] ComPtr<ID3D12CommandQueue> Ptr() const;

 private:
  ComPtr<ID3D12CommandQueue> command_queue_;
};
}  // namespace grassland::d3d12
