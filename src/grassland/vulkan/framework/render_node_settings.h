#pragma once
#include <grassland/vulkan/framework/core.h>

namespace grassland::vulkan::framework {
struct RenderNodeSettings {
  RenderNodeSettings();
  helper::DescriptorSetLayoutBindings bindings;
  helper::AttachmentParameters attachment_parameters;
  helper::ShaderStages shader_stages;
};
}  // namespace grassland::vulkan::framework
