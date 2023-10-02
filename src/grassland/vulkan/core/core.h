#pragma once
#include "grassland/vulkan/core/command_buffer.h"
#include "grassland/vulkan/core/command_pool.h"
#include "grassland/vulkan/core/device.h"
#include "grassland/vulkan/core/instance.h"
#include "grassland/vulkan/core/physical_device.h"
#include "grassland/vulkan/core/swap_chain.h"

namespace grassland::vulkan {

struct CoreSettings {
  GLFWwindow *window{nullptr};
  bool enable_validation_layers{kDefaultEnableValidationLayers};
  bool enable_ray_tracing{false};
  int device_index{-1};
  int frames_in_flight{3};
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
  [[nodiscard]] int FramesInFlight() const {
    return settings_.frames_in_flight;
  }

 private:
  CoreSettings settings_;
  std::unique_ptr<class Instance> instance_;
  std::unique_ptr<class Surface> surface_;
  std::unique_ptr<class Device> device_;
  std::unique_ptr<class CommandPool> command_pool_;
  std::vector<std::unique_ptr<class CommandBuffer>> command_buffers_;
};

}  // namespace grassland::vulkan
