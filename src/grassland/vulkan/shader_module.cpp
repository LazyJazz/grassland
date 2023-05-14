#include "shader_module.h"

#include "grassland/file/file.h"

namespace grassland::vulkan {
ShaderModule::ShaderModule(const class Device &device,
                           const std::string &filename)
    : ShaderModule(device, file::ReadFileBinary(filename.c_str())) {
}

ShaderModule::ShaderModule(const class Device &device,
                           const std::vector<uint8_t> &code)
    : device_(device) {
  VkShaderModuleCreateInfo shader_module_info{};
  shader_module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_module_info.codeSize = code.size();
  shader_module_info.pCode = reinterpret_cast<const uint32_t *>(code.data());

  GRASSLAND_VULKAN_CHECK(vkCreateShaderModule(
      device_.Handle(), &shader_module_info, nullptr, &shader_module_));
}

ShaderModule::~ShaderModule() {
  if (shader_module_ != VK_NULL_HANDLE) {
    vkDestroyShaderModule(device_.Handle(), shader_module_, nullptr);
  }
}

VkPipelineShaderStageCreateInfo ShaderModule::CreateShaderStage(
    VkShaderStageFlagBits stage) const {
  VkPipelineShaderStageCreateInfo stage_info{};
  stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_info.stage = stage;
  stage_info.module = shader_module_;
  stage_info.pName = "main";
  return stage_info;
}
}  // namespace grassland::vulkan
