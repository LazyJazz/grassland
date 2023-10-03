#include "grassland/vulkan/shaders/built_in_shaders.h"

namespace grassland::vulkan::built_in_shaders {

namespace {
#include "built_in_shaders.inl"
}

std::string GetShader(const std::string &name) {
  if (name == "example.vert") {
    return {reinterpret_cast<const char *>(example_vert), example_vert_len};
  }
  return "";
}
}  // namespace grassland::vulkan::built_in_shaders
