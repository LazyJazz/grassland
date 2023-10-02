#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {
Core::Core(const CoreSettings &settings) : settings_(settings) {
  InstanceSettings instance_settings;
  if (settings.window) {
    instance_settings.EnableSurfaceSupport();
  }
  if (settings.enable_validation_layers) {
    instance_settings.EnableValidationLayers();
  }
  instance_ = std::make_unique<class Instance>(instance_settings);

  // Find the optimal physical device
  std::vector<PhysicalDevice> physical_devices;
  instance_->EnumeratePhysicalDevices(physical_devices);
  PhysicalDevice *physical_device = nullptr;
  uint64_t device_score = 0;
  for (auto &device : physical_devices) {
    if (settings.enable_ray_tracing && !device.SupportRayTracing()) {
      continue;
    }
    uint64_t score = device.Evaluate();
    if (score > device_score) {
      physical_device = &device;
      device_score = score;
    }
  }

  if (!physical_device) {
    LAND_ERROR("No suitable physical device found");
  }

  if (settings.window) {
    // Create the surface
    surface_ =
        std::make_unique<class Surface>(instance_.get(), settings.window);
  }

  // Create the logical device
  device_ = std::make_unique<class Device>(instance_.get(), *physical_device,
                                           surface_.get(),
                                           settings.enable_ray_tracing);

  // Create the command pool
  command_pool_ = std::make_unique<class CommandPool>(device_.get());

  // Create command buffers
  command_buffers_.resize(settings.frames_in_flight);
  for (auto &command_buffer : command_buffers_) {
    command_buffer = std::make_unique<class CommandBuffer>(command_pool_.get());
  }

  // Create the swap chain
  if (settings.window) {
    swap_chain_ =
        std::make_unique<class SwapChain>(device_.get(), surface_.get());
  }
}

Core::~Core() {
  // Release all the resources in reverse order of creation
  swap_chain_.reset();
  command_buffers_.clear();
  command_pool_.reset();
  device_.reset();
  surface_.reset();
  instance_.reset();
}

void Core::SingleTimeCommands(
    const std::function<void(VkCommandBuffer)> &function) {
  grassland::vulkan::SingleTimeCommands(command_pool_.get(), function);
}
}  // namespace grassland::vulkan
