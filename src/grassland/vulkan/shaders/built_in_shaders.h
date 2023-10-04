#include <map>

#include "grassland/vulkan/resources/shader_module.h"

namespace grassland::vulkan::built_in_shaders {
std::vector<std::string> ListAllBuiltInShaders();
std::vector<uint32_t> GetShaderCompiledSpv(const std::string &name);
}  // namespace grassland::vulkan::built_in_shaders
