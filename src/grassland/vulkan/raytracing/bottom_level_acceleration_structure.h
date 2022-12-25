#pragma once
#include "grassland/vulkan/buffer.h"
#include "grassland/vulkan/command_pool.h"
#include "grassland/vulkan/device.h"

namespace grassland::vulkan::raytracing {
class BottomLevelAccelerationStructure {
 public:
  BottomLevelAccelerationStructure(Device *device,
                                   CommandPool *command_pool,
                                   void *vertices_data,
                                   uint32_t vertex_buffer_size,
                                   uint32_t *indices,
                                   uint32_t num_index,
                                   uint32_t stride);
  ~BottomLevelAccelerationStructure();
  [[nodiscard]] Buffer *GetBuffer() const;
  [[nodiscard]] VkDeviceAddress GetDeviceAddress() const;

 private:
  GRASSLAND_VULKAN_HANDLE(VkAccelerationStructureKHR);
  GRASSLAND_VULKAN_DEVICE_PTR;
  std::unique_ptr<Buffer> buffer_;
  VkDeviceAddress device_address_;
};
}  // namespace grassland::vulkan::raytracing
