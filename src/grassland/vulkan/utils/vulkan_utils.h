#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <memory>
#include <string>
#include <vector>

#include "grassland/util/util.h"

#define GRASSLAND_VULKAN_HANDLE(type) \
  type handle_{};                     \
                                      \
 public:                              \
  type &GetHandle() {                 \
    return handle_;                   \
  }                                   \
  const type &GetHandle() const {     \
    return handle_;                   \
  }                                   \
                                      \
 private:

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
  PFN_##function_name function_name{};
namespace grassland::vulkan {
const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kDefaultEnableValidationLayers = false;
#else
constexpr bool kDefaultEnableValidationLayers = true;
#endif

std::string PCIVendorIDToName(uint32_t vendor_id);

std::string VkFormatToName(VkFormat format);
std::string VkColorSpaceToName(VkColorSpaceKHR color_space);
std::string VkPresentModeToName(VkPresentModeKHR present_mode);
}  // namespace grassland::vulkan
