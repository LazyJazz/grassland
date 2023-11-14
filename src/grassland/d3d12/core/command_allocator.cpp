#include "grassland/d3d12/core/command_allocator.h"

namespace grassland::d3d12 {

CommandAllocator::CommandAllocator(const Device &device,
                                   D3D12_COMMAND_LIST_TYPE list_type) {
  ThrowIfFailed(device.Ptr()->CreateCommandAllocator(
                    list_type, IID_PPV_ARGS(&command_allocator_)),
                "Failed to create command allocator");
}

CommandAllocator::~CommandAllocator() = default;

ComPtr<ID3D12CommandAllocator> CommandAllocator::Ptr() const {
  return command_allocator_;
}
}  // namespace grassland::d3d12
