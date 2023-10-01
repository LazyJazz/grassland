#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {
Core::Core(const CoreSettings &settings) {
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
}

Core::~Core() {
  // Release all the resources in reverse order of creation
  device_.reset();
  surface_.reset();
  instance_.reset();
}
}  // namespace grassland::vulkan
