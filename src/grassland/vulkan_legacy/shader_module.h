#pragma once
#include <grassland/vulkan_legacy/device.h>

#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy {
class ShaderModule {
 public:
  explicit ShaderModule(Device *device, const char *spv_file_path);
  explicit ShaderModule(Device *device, const std::vector<uint8_t> &spv_data);
  ~ShaderModule();

 private:
  GRASSLAND_VULKAN_HANDLE(VkShaderModule)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan_legacy
