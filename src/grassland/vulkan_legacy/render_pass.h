#pragma once
#include <grassland/vulkan_legacy/device.h>
#include <grassland/vulkan_legacy/helper/attachment_parameters.h>

#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy {
class RenderPass {
 public:
  RenderPass(Device *device, VkFormat color_format);
  RenderPass(Device *device, VkFormat color_format, VkFormat depth_format);
  RenderPass(Device *device,
             const helper::AttachmentParameters &attachment_parameters);
  ~RenderPass();

 private:
  void ConstructorCommon(
      Device *device,
      const helper::AttachmentParameters &attachment_parameters);
  GRASSLAND_VULKAN_HANDLE(VkRenderPass)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan_legacy
