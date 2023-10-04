#pragma once
#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan {

struct InstanceSettings {
  std::vector<const char *> extensions;
  VkApplicationInfo app_info{};
  bool enable_validation_layers = false;
  InstanceSettings();
  void EnableValidationLayers();
  void EnableSurfaceSupport();
};

}  // namespace grassland::vulkan
