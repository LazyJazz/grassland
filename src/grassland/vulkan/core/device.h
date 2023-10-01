#pragma once
#include "grassland/vulkan/core/instance.h"

namespace grassland::vulkan {

class Device {
 public:
  explicit Device(Instance *instance, PhysicalDevice *physical_device);
  ~Device();

  VkDevice Handle() const;

 private:
  VkDevice device_{};
};
}  // namespace grassland::vulkan
