#include "grassland/d3d12/core/graphics_command_list.h"

namespace grassland::d3d12 {
GraphicsCommandList::GraphicsCommandList(
    const Device &device,
    const CommandAllocator &command_allocator,
    D3D12_COMMAND_LIST_TYPE list_type)
    : command_list_{nullptr} {
  ThrowIfFailed(device.Ptr()->CreateCommandList(
                    0, list_type, command_allocator.Ptr().Get(), nullptr,
                    IID_PPV_ARGS(&command_list_)),
                "Failed to create command list");
}

GraphicsCommandList::~GraphicsCommandList() = default;

ComPtr<ID3D12GraphicsCommandList> GraphicsCommandList::Ptr() const {
  return command_list_;
}

}  // namespace grassland::d3d12
