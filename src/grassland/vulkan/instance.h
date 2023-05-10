#pragma once
#include "GLFW/glfw3.h"
#include "grassland/util/util.h"
#include "grassland/vulkan/vulkan_util.h"

namespace grassland::vulkan {

struct InstanceSettings {
  bool glfw_surface{false};
  bool validation_layer{kDefaultEnableValidationLayers};
};

class Instance {
 public:
  GRASSLAND_CANNOT_COPY(Instance)
  explicit Instance(const InstanceSettings &settings = InstanceSettings{});

 private:
  GRASSLAND_VULKAN_HANDLE(VkInstance, instance_)
};
}  // namespace grassland::vulkan
