#pragma once
#include "grassland/file/file.h"
#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {
class ShaderModule {
 public:
  ShaderModule(Core *core, const std::string &path);
  ShaderModule(Core *core, const std::vector<uint8_t> &spirv_code);
  ShaderModule(Core *core, const std::vector<uint32_t> &spirv_code);
  ~ShaderModule();

  [[nodiscard]] VkShaderModule Handle() const;

 private:
  Core *core_{};
  VkShaderModule shader_module_{};
};

std::vector<uint32_t> CompileGLSLToSPIRV(const std::string &glsl_code,
                                         VkShaderStageFlagBits shader_stage);
}  // namespace grassland::vulkan
