#include "grassland/vulkan/shaders/built_in_shaders.h"

#include "absl/strings/match.h"

namespace grassland::vulkan::built_in_shaders {

namespace {
#include "built_in_shaders.inl"
}

std::vector<uint32_t> GetShaderCompiledSpv(const std::string &name) {
  // Get shader code from built-in-shaders.inl
  std::string shader_code = GetShaderCode(name);

  VkShaderStageFlagBits shader_stage;
  // Get shader stage from name suffix, abseil::EndsWith is used here
  if (absl::EndsWith(name, ".vert")) {
    shader_stage = VK_SHADER_STAGE_VERTEX_BIT;
  } else if (absl::EndsWith(name, ".frag")) {
    shader_stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  } else if (absl::EndsWith(name, ".comp")) {
    shader_stage = VK_SHADER_STAGE_COMPUTE_BIT;
  } else if (absl::EndsWith(name, ".geom")) {
    shader_stage = VK_SHADER_STAGE_GEOMETRY_BIT;
  } else if (absl::EndsWith(name, ".tesc")) {
    shader_stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
  } else if (absl::EndsWith(name, ".tese")) {
    shader_stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
  } else if (absl::EndsWith(name, ".mesh")) {
    shader_stage = VK_SHADER_STAGE_MESH_BIT_NV;
  } else if (absl::EndsWith(name, ".task")) {
    shader_stage = VK_SHADER_STAGE_TASK_BIT_NV;
  }
  // Ray tracing stages
  else if (absl::EndsWith(name, ".rgen")) {
    shader_stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  } else if (absl::EndsWith(name, ".rint")) {
    shader_stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
  } else if (absl::EndsWith(name, ".rahit")) {
    shader_stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  } else if (absl::EndsWith(name, ".rchit")) {
    shader_stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  } else if (absl::EndsWith(name, ".rmiss")) {
    shader_stage = VK_SHADER_STAGE_MISS_BIT_KHR;
  } else if (absl::EndsWith(name, ".rcall")) {
    shader_stage = VK_SHADER_STAGE_CALLABLE_BIT_KHR;
  } else {
    throw std::runtime_error("Unknown shader stage");
  }

  // Compile the code into SPIR-V
  std::vector<uint32_t> spirv_code =
      CompileGLSLToSPIRV(shader_code, shader_stage);

  return spirv_code;
}

std::vector<std::string> ListAllBuiltInShaders() {
  std::vector<std::string> result;
  result.reserve(shader_list.size());
  for (auto &shader : shader_list) {
    result.push_back(shader.first);
  }
  return result;
}
}  // namespace grassland::vulkan::built_in_shaders
