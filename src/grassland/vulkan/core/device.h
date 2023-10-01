#pragma once
#include <set>

#include "grassland/vulkan/core/device_procedures.h"
#include "grassland/vulkan/core/instance.h"
#include "grassland/vulkan/core/physical_device.h"
#include "grassland/vulkan/core/queue.h"
#include "grassland/vulkan/core/surface.h"

namespace grassland::vulkan {

struct DeviceSettings {
  PhysicalDevice physical_device;
  Surface *surface;
  bool enable_raytracing;
};

class Device {
 public:
  explicit Device(Instance *instance,
                  const PhysicalDevice &physical_device,
                  Surface *surface = nullptr,
                  bool enable_raytracing = false);
  Device(Instance *instance, const DeviceSettings &settings);
  ~Device();

  [[nodiscard]] VkDevice Handle() const;
  [[nodiscard]] Queue GraphicsQueue() const;
  [[nodiscard]] Queue PresentQueue() const;
  DeviceProcedures &Procedures() {
    return device_procedures_;
  }

 private:
  VkDevice device_{};
  Queue graphics_queue_{};
  Queue present_queue_{};
  DeviceProcedures device_procedures_{};
};
}  // namespace grassland::vulkan
