#pragma once
#include <vulkan/vulkan.h>

#include "grassland/util/util.h"
#include "vector"

#define GRASSLAND_VULKAN_HANDLE(VulkanHandleType, object_name) \
 public:                                                       \
  VulkanHandleType Handle() const {                            \
    return object_name;                                        \
  }                                                            \
                                                               \
 private:                                                      \
  VulkanHandleType object_name{VK_NULL_HANDLE};

#define GRASSLAND_VULKAN_DEVICE_PTR \
  Device *device_{nullptr};         \
                                    \
 public:                            \
  Device *GetDevice() const {       \
    return device_;                 \
  }                                 \
                                    \
 private:

#define GRASSLAND_VULKAN_PROCEDURE_VAR(function_name) \
  PFN_##function_name function_name##_;

namespace grassland::vulkan {
const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

void CheckResult(VkResult result,
                 const char *file_name,
                 int line,
                 const char *code);

#ifdef NDEBUG
constexpr bool kDefaultEnableValidationLayers = false;
#else
constexpr bool kDefaultEnableValidationLayers = true;
#endif

#ifdef NDEBUG
#define GRASSLAND_VULKAN_CHECK(x) x
#else
#define GRASSLAND_VULKAN_CHECK(x) \
  ::grassland::vulkan::CheckResult(x, __FILE__, __LINE__, #x)
#endif
}  // namespace grassland::vulkan
