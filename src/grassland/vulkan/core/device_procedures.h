#pragma once
#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan {
class DeviceProcedures {
 public:
  explicit DeviceProcedures();
  void Initialize(VkDevice device, bool enable_raytracing_extensions);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdBuildAccelerationStructuresKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCreateAccelerationStructureKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetAccelerationStructureBuildSizesKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetBufferDeviceAddressKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkDestroyAccelerationStructureKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetAccelerationStructureDeviceAddressKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCreateRayTracingPipelinesKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetRayTracingShaderGroupHandlesKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdTraceRaysKHR);
};
}  // namespace grassland::vulkan
