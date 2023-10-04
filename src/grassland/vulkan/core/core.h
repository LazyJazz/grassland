#pragma once
#include "grassland/vulkan/core/command_buffer.h"
#include "grassland/vulkan/core/command_pool.h"
#include "grassland/vulkan/core/device.h"
#include "grassland/vulkan/core/fence.h"
#include "grassland/vulkan/core/instance.h"
#include "grassland/vulkan/core/physical_device.h"
#include "grassland/vulkan/core/semaphore.h"
#include "grassland/vulkan/core/single_time_commands.h"
#include "grassland/vulkan/core/surface.h"
#include "grassland/vulkan/core/swap_chain.h"

namespace grassland::vulkan {

struct CoreSettings {
  GLFWwindow *window{nullptr};
  bool enable_validation_layers{kDefaultEnableValidationLayers};
  bool enable_ray_tracing{false};
  int device_index{-1};
  int max_frames_in_flight{3};
};

class Core {
 public:
  explicit Core(const CoreSettings &settings = CoreSettings());
  ~Core();

  [[nodiscard]] class Surface *Surface() const {
    return surface_.get();
  }
  [[nodiscard]] class Instance *Instance() const {
    return instance_.get();
  }
  [[nodiscard]] class Device *Device() const {
    return device_.get();
  }
  [[nodiscard]] class CommandPool *CommandPool() const {
    return command_pool_.get();
  }
  [[nodiscard]] class SwapChain *SwapChain() const {
    return swap_chain_.get();
  }
  [[nodiscard]] int MaxFramesInFlight() const {
    return settings_.max_frames_in_flight;
  }
  // Get the current frame index
  [[nodiscard]] uint32_t CurrentFrame() const {
    return current_frame_;
  }
  // Get current image index
  [[nodiscard]] uint32_t ImageIndex() const {
    return image_index_;
  }
  // Get the current command buffer
  [[nodiscard]] class CommandBuffer *CommandBuffer() const {
    return command_buffers_[current_frame_].get();
  }

  void SingleTimeCommands(const std::function<void(VkCommandBuffer)> &function);

  // Begin Frame function and End Frame function
  void BeginFrame();
  void EndFrame();

  void RebuildSwapChain();

 private:
  CoreSettings settings_;
  std::unique_ptr<class Instance> instance_;
  std::unique_ptr<class Surface> surface_;
  std::unique_ptr<class Device> device_;
  std::unique_ptr<class CommandPool> command_pool_;
  std::vector<std::unique_ptr<class CommandBuffer>> command_buffers_;
  std::unique_ptr<class SwapChain> swap_chain_;
  uint32_t current_frame_{0};
  uint32_t image_index_{0};

  // Semaphores and fences
  std::vector<std::unique_ptr<class Semaphore>> image_available_semaphores_;
  std::vector<std::unique_ptr<class Semaphore>> render_finish_semaphores_;
  std::vector<std::unique_ptr<class Fence>> in_flight_fences_;
};

}  // namespace grassland::vulkan
