#pragma once
#include <grassland/vulkan/instance.h>
#include <grassland/vulkan/surface.h>
#include <grassland/vulkan/util.h>

#include <string>
#include <vector>

namespace grassland::vulkan {
class PhysicalDevice {
 public:
  explicit PhysicalDevice(VkPhysicalDevice handle);
  [[nodiscard]] std::string DeviceName() const;
  [[nodiscard]] int HasGraphicsFamily() const;
  int HasPresentFamily(Surface *surface) const;
  bool HasPresentationSupport() const;

 private:
  VK_HANDLE(VkPhysicalDevice)
};

std::vector<PhysicalDevice> GetPhysicalDevices(Instance *instance);
}  // namespace grassland::vulkan
