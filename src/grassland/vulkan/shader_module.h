#pragma once
#include "device.h"

namespace grassland::vulkan {
class ShaderModule {
 public:
  GRASSLAND_CANNOT_COPY(ShaderModule)
  ShaderModule(const class Device &device);
  ~ShaderModule();

 private:
  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkShaderModule, shader_module_)
};
}  // namespace grassland::vulkan
