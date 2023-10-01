#pragma once
#include "grassland/vulkan_legacy/device.h"

namespace grassland::vulkan_legacy {
class DeviceProcedures {
 public:
  explicit DeviceProcedures();
  static DeviceProcedures *GetStaticInstance();
  void SetDevice(Device *device);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdBuildAccelerationStructuresKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCreateAccelerationStructureKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetAccelerationStructureBuildSizesKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetBufferDeviceAddressKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkDestroyAccelerationStructureKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetAccelerationStructureDeviceAddressKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCreateRayTracingPipelinesKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkGetRayTracingShaderGroupHandlesKHR);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCmdTraceRaysKHR);

 private:
  GRASSLAND_VULKAN_DEVICE_PTR;
};
}  // namespace grassland::vulkan_legacy
