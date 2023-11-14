#pragma once
#include "grassland/d3d12/core/command_allocator.h"

namespace grassland::d3d12 {
class GraphicsCommandList {
 public:
  explicit GraphicsCommandList(
      const Device &device,
      const CommandAllocator &command_allocator,
      D3D12_COMMAND_LIST_TYPE list_type = D3D12_COMMAND_LIST_TYPE_DIRECT);
  ~GraphicsCommandList();

  [[nodiscard]] ComPtr<ID3D12GraphicsCommandList> Ptr() const;

 private:
  ComPtr<ID3D12GraphicsCommandList> command_list_;
};
}  // namespace grassland::d3d12
