#include "grassland/vulkan/core/device_procedures.h"

namespace grassland::vulkan {

namespace {
template <class FuncTy>
FuncTy GetProcedure(VkDevice device, const char *function_name) {
  auto func =
      reinterpret_cast<FuncTy>(vkGetDeviceProcAddr(device, function_name));
  return func;
};
}  // namespace

#define GET_PROCEDURE(device, function_name)                               \
  function_name##_ = grassland::vulkan::GetProcedure<PFN_##function_name>( \
      device, #function_name)

DeviceProcedures::DeviceProcedures() = default;

void DeviceProcedures::GetFunctionPointers(VkDevice device) {
  GET_PROCEDURE(device, vkGetBufferDeviceAddressKHR);
  GET_PROCEDURE(device, vkGetAccelerationStructureBuildSizesKHR);
  GET_PROCEDURE(device, vkCreateAccelerationStructureKHR);
  GET_PROCEDURE(device, vkCmdBuildAccelerationStructuresKHR);
  GET_PROCEDURE(device, vkDestroyAccelerationStructureKHR);
  GET_PROCEDURE(device, vkGetAccelerationStructureDeviceAddressKHR);
  GET_PROCEDURE(device, vkCreateRayTracingPipelinesKHR);
  GET_PROCEDURE(device, vkGetRayTracingShaderGroupHandlesKHR);
  GET_PROCEDURE(device, vkCmdTraceRaysKHR);
}

VkDeviceAddress DeviceProcedures::vkGetBufferDeviceAddressKHR(
    VkDevice device,
    const VkBufferDeviceAddressInfo *pInfo) {
  return vkGetBufferDeviceAddressKHR_(device, pInfo);
}

void DeviceProcedures::vkGetAccelerationStructureBuildSizesKHR(
    VkDevice device,
    VkAccelerationStructureBuildTypeKHR buildType,
    const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
    const uint32_t *pMaxPrimitiveCounts,
    VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo) {
  return vkGetAccelerationStructureBuildSizesKHR_(
      device, buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo);
}

VkResult DeviceProcedures::vkCreateAccelerationStructureKHR(
    VkDevice device,
    const VkAccelerationStructureCreateInfoKHR *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkAccelerationStructureKHR *pAccelerationStructure) {
  return vkCreateAccelerationStructureKHR_(device, pCreateInfo, pAllocator,
                                           pAccelerationStructure);
}

void DeviceProcedures::vkCmdBuildAccelerationStructuresKHR(
    VkCommandBuffer commandBuffer,
    uint32_t infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
    const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos) {
  return vkCmdBuildAccelerationStructuresKHR_(commandBuffer, infoCount, pInfos,
                                              ppBuildRangeInfos);
}

void DeviceProcedures::vkDestroyAccelerationStructureKHR(
    VkDevice device,
    VkAccelerationStructureKHR accelerationStructure,
    const VkAllocationCallbacks *pAllocator) {
  return vkDestroyAccelerationStructureKHR_(device, accelerationStructure,
                                            pAllocator);
}

VkDeviceAddress DeviceProcedures::vkGetAccelerationStructureDeviceAddressKHR(
    VkDevice device,
    const VkAccelerationStructureDeviceAddressInfoKHR *pInfo) {
  return vkGetAccelerationStructureDeviceAddressKHR_(device, pInfo);
}

VkResult DeviceProcedures::vkCreateRayTracingPipelinesKHR(
    VkDevice device,
    VkDeferredOperationKHR deferredOperation,
    VkPipelineCache pipelineCache,
    uint32_t createInfoCount,
    const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
    const VkAllocationCallbacks *pAllocator,
    VkPipeline *pPipelines) {
  return vkCreateRayTracingPipelinesKHR_(device, deferredOperation,
                                         pipelineCache, createInfoCount,
                                         pCreateInfos, pAllocator, pPipelines);
}

VkResult DeviceProcedures::vkGetRayTracingShaderGroupHandlesKHR(
    VkDevice device,
    VkPipeline pipeline,
    uint32_t firstGroup,
    uint32_t groupCount,
    size_t dataSize,
    void *pData) {
  return vkGetRayTracingShaderGroupHandlesKHR_(device, pipeline, firstGroup,
                                               groupCount, dataSize, pData);
}

void DeviceProcedures::vkCmdTraceRaysKHR(
    VkCommandBuffer commandBuffer,
    const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
    const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
    uint32_t width,
    uint32_t height,
    uint32_t depth) {
  vkCmdTraceRaysKHR_(commandBuffer, pRaygenShaderBindingTable,
                     pMissShaderBindingTable, pHitShaderBindingTable,
                     pCallableShaderBindingTable, width, height, depth);
}

}  // namespace grassland::vulkan
