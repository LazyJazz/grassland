#pragma once
#include "grassland/d3d12/core/adapter.h"
#include "grassland/d3d12/core/command_allocator.h"
#include "grassland/d3d12/core/command_queue.h"
#include "grassland/d3d12/core/device.h"
#include "grassland/d3d12/core/dxgi_factory.h"
#include "grassland/d3d12/core/fence.h"
#include "grassland/d3d12/core/graphics_command_list.h"
#include "grassland/d3d12/core/swap_chain.h"
#include "grassland/d3d12/utils/d3d12_utils.h"

namespace grassland::d3d12 {

struct CoreSettings {
  HWND hwnd{nullptr};
  bool enable_debug_layer{kDefaultEnableDebugLayer};
  bool enable_ray_tracing{false};
  int device_index{-1};
  int max_frames_in_flight{2};
};

class Core {
 public:
  explicit Core(const CoreSettings &settings);
  ~Core();

  class DXGIFactory *Factory() {
    return factory_.get();
  }
  class Device *Device() {
    return device_.get();
  }
  class CommandAllocator *CommandAllocator() {
    return command_allocator_.get();
  }
  class CommandQueue *CommandQueue() {
    return command_queue_.get();
  }
  class GraphicsCommandList *CommandList() {
    return command_list_[current_frame_].get();
  }
  class GraphicsCommandList *CommandList(int index) {
    return command_list_[index].get();
  }
  class SwapChain *SwapChain() {
    return swap_chain_.get();
  }
  int MaxFramesInFlight() const {
    return settings_.max_frames_in_flight;
  }
  // Get the current frame index
  uint32_t CurrentFrame() const {
    return current_frame_;
  }
  // Get current image index
  uint32_t ImageIndex() const {
    return image_index_;
  }

  void RebuildSwapChain(int width = 0,
                        int height = 0,
                        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN);

  void BeginFrame();
  void EndFrame();

  void MoveToNextFrame();
  void WaitForGPU();

 private:
  std::unique_ptr<DXGIFactory> factory_;
  std::unique_ptr<class Device> device_;
  std::unique_ptr<class CommandAllocator> command_allocator_;
  std::unique_ptr<class CommandQueue> command_queue_;
  std::vector<std::unique_ptr<class GraphicsCommandList>> command_list_;
  std::unique_ptr<class SwapChain> swap_chain_;

  std::unique_ptr<class Fence> fence_;
  std::vector<uint64_t> fence_values_;
  HANDLE fence_event_{};

  uint32_t current_frame_{0};
  uint32_t image_index_{0};

  CoreSettings settings_;
};
}  // namespace grassland::d3d12
