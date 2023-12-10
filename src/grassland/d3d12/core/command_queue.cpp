#include "grassland/d3d12/core/command_queue.h"

namespace grassland::d3d12 {

CommandQueue::CommandQueue(const Device &device,
                           D3D12_COMMAND_LIST_TYPE list_type) {
  D3D12_COMMAND_QUEUE_DESC desc{};
  desc.Type = list_type;
  desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  ThrowIfFailed(
      device.Ptr()->CreateCommandQueue(&desc, IID_PPV_ARGS(&command_queue_)),
      "Failed to create command queue");
}

CommandQueue::~CommandQueue() = default;

ComPtr<ID3D12CommandQueue> CommandQueue::Ptr() const {
  return command_queue_;
}
}  // namespace grassland::d3d12
