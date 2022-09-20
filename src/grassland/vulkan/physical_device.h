#pragma once
#include <grassland/vulkan/instance.h>
#include <grassland/vulkan/surface.h>
#include <grassland/vulkan/util.h>

#include <functional>
#include <string>
#include <vector>

namespace grassland::vulkan {
class PhysicalDevice {
 public:
  explicit PhysicalDevice(VkPhysicalDevice handle);
  [[nodiscard]] std::string DeviceName() const;
  [[nodiscard]] int HasGraphicsFamily() const;
  int HasPresentFamily(Surface *surface) const;
  [[nodiscard]] bool HasPresentationSupport() const;
  [[nodiscard]] bool IsDiscreteGPU() const;
  [[nodiscard]] bool HasGeometryShader() const;
  void PrintDeviceProperties() const;
  void PrintDeviceFeatures() const;

  [[nodiscard]] VkPhysicalDeviceFeatures GetFeatures() const;
  [[nodiscard]] VkPhysicalDeviceProperties GetProperties() const;

 private:
  VK_HANDLE(VkPhysicalDevice)
  VkPhysicalDeviceProperties properties_;
  VkPhysicalDeviceFeatures features_;
};

std::vector<PhysicalDevice> GetPhysicalDevices(Instance *instance);
PhysicalDevice PickPhysicalDevice(
    const std::vector<PhysicalDevice> &device_list,
    const std::function<int(PhysicalDevice physical_device)> &rate_function);
PhysicalDevice PickPhysicalDevice(
    Instance *instance,
    const std::function<int(PhysicalDevice)> &rate_function);

}  // namespace grassland::vulkan
