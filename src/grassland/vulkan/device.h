#pragma once

#include "instance.h"
#include "physical_device.h"
#include "surface.h"

namespace grassland::vulkan {

class Device {
 public:
  GRASSLAND_CANNOT_COPY(Device)
  Device(class PhysicalDevice physical_device,
         class Surface *surface = nullptr,
         bool enable_validation_layer = kDefaultEnableValidationLayers,
         const std::vector<const char *> &extra_extensions =
             std::vector<const char *>{},
         void *extra_features = nullptr);
  ~Device();

  [[nodiscard]] class PhysicalDevice PhysicalDevice() const {
    return physical_device_;
  }

  [[nodiscard]] uint32_t GraphicsFamilyIndex() const {
    return graphics_family_index_;
  }
  [[nodiscard]] uint32_t ComputeFamilyIndex() const {
    return graphics_family_index_;
  }
  [[nodiscard]] uint32_t PresentFamilyIndex() const {
    return present_family_index_;
  }

  [[nodiscard]] VkQueue GraphicsQueue() const {
    return graphics_queue_;
  }
  [[nodiscard]] VkQueue ComputeQueue() const {
    return compute_queue_;
  }
  [[nodiscard]] VkQueue PresentQueue() const {
    return present_queue_;
  }

  [[nodiscard]] const class Surface *Surface() const {
    return surface_;
  }

  [[nodiscard]] const VkPhysicalDeviceMemoryProperties &
  PhysicalDeviceMemoryProperties() const {
    return memory_properties_;
  }

 private:
  GRASSLAND_VULKAN_HANDLE(VkDevice, device_)
  const class PhysicalDevice physical_device_ {};
  const class Surface *surface_;

  uint32_t graphics_family_index_{};
  uint32_t compute_family_index_{};
  uint32_t present_family_index_{};

  VkQueue graphics_queue_{VK_NULL_HANDLE};
  VkQueue compute_queue_{VK_NULL_HANDLE};
  VkQueue present_queue_{VK_NULL_HANDLE};

  VkPhysicalDeviceMemoryProperties memory_properties_{};
};
}  // namespace grassland::vulkan
