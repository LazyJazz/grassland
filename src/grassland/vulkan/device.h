#pragma once

#include "instance.h"
#include "physical_device.h"
#include "surface.h"

namespace grassland::vulkan {

class Device {
 public:
  GRASSLAND_CANNOT_COPY(Device)
  Device(PhysicalDevice physical_device,
         Surface *surface = nullptr,
         bool enable_validation_layer = kDefaultEnableValidationLayers,
         const std::vector<const char *> &extra_extensions =
             std::vector<const char *>{},
         void *extra_features = nullptr);

 private:
  GRASSLAND_VULKAN_HANDLE(VkDevice, device_)
  const PhysicalDevice physical_device_{};
  const Surface *surface_;

  uint32_t graphics_family_index_{};
  uint32_t compute_family_index_{};
  uint32_t present_family_index_{};

  VkQueue graphics_queue_{VK_NULL_HANDLE};
  VkQueue compute_queue_{VK_NULL_HANDLE};
  VkQueue present_queue_{VK_NULL_HANDLE};
};
}  // namespace grassland::vulkan
