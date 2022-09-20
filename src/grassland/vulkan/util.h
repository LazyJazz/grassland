#pragma once

#include <vulkan/vulkan.h>

#include <vector>

#define VK_HANDLE(type)    \
  type handle_;            \
                           \
 public:                   \
  type GetHandle() const { \
    return handle_;        \
  }                        \
                           \
 private:

namespace grassland::vulkan {
const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif
}  // namespace grassland::vulkan
