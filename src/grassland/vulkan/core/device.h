#pragma once
#include "grassland/vulkan/core/instance.h"
#include "grassland/vulkan/core/physical_device.h"

namespace grassland::vulkan {

struct DeviceSettings {
  PhysicalDevice physical_device;
  std::vector<std::string> extensions;
};

class Device {
 public:
  explicit Device();
  ~Device();

  VkDevice Handle() const;

 private:
  VkDevice device_{};
};
}  // namespace grassland::vulkan
