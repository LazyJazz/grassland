#pragma once
#include "vulkan/vulkan.h"

#define GRASSLAND_VULKAN_HANDLE(VulkanHandleType, object_name) \
 public:                                                       \
  VulkanHandleType Handle() const {                            \
    return object_name;                                        \
  }                                                            \
                                                               \
 private:                                                      \
  VulkanHandleType object_name{VK_NULL_HANDLE};

namespace grassland::vulkan {}
