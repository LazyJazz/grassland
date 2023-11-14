#pragma once
#include "grassland/d3d12/core/adapter.h"
#include "grassland/d3d12/core/command_allocator.h"
#include "grassland/d3d12/core/command_queue.h"
#include "grassland/d3d12/core/device.h"
#include "grassland/d3d12/core/dxgi_factory.h"
#include "grassland/d3d12/core/graphics_command_list.h"
#include "grassland/d3d12/utils/d3d12_utils.h"

namespace grassland::d3d12 {

struct CoreSettings {
  HWND hwnd{nullptr};
  bool enable_debug_layer{kDefaultEnableDebugLayer};
  bool enable_ray_tracing{false};
  int device_index{-1};
  int max_frames_in_flight{3};
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

 private:
  std::unique_ptr<DXGIFactory> factory_;
  std::unique_ptr<class Device> device_;
  std::unique_ptr<class CommandAllocator> command_allocator_;
  std::unique_ptr<class CommandQueue> command_queue_;
  std::vector<std::unique_ptr<class GraphicsCommandList>> command_list_;

  uint32_t current_frame_{0};
  uint32_t image_index_{0};

  CoreSettings settings_;
};
}  // namespace grassland::d3d12
