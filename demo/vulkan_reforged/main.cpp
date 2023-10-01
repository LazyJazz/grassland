#pragma once

#include "grassland/grassland.h"
#include "iostream"

using namespace grassland;

int main() {
  // Create an Instance, then enumerate all physical devices.
  vulkan::Instance instance;
  std::vector<vulkan::PhysicalDevice> physical_devices;
  instance.EnumeratePhysicalDevices(physical_devices);
  // Print physical device information
  for (const auto &physical_device : physical_devices) {
    std::cout << "Physical Device: "
              << physical_device.GetPhysicalDeviceProperties().deviceName
              << std::endl;
    // Show device type
    std::cout << "  Device Type: ";
    switch (physical_device.GetPhysicalDeviceProperties().deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_OTHER:
        std::cout << "Other" << std::endl;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        std::cout << "Integrated GPU" << std::endl;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        std::cout << "Discrete GPU" << std::endl;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        std::cout << "Virtual GPU" << std::endl;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:
        std::cout << "CPU" << std::endl;
        break;
      default:
        std::cout << "Unknown" << std::endl;
    }
    std::cout << "  Device Memory:"
              << grassland::util::SizeToString(
                     physical_device.GetDeviceLocalMemorySize())
              << std::endl;
    // Show raytracing and geometry shader support
    std::cout << "  Ray Tracing: "
              << (physical_device.IsRayTracingSupported() ? "Supported"
                                                          : "Not Supported")
              << std::endl;
    std::cout << "  Geometry Shader: "
              << (physical_device.IsGeometryShaderSupported() ? "Supported"
                                                              : "Not Supported")
              << std::endl;
  }

  vulkan::Device device(&instance, physical_devices[0]);
  return 0;
}
