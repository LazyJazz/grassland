#pragma once
#include "grassland/vulkan_legacy/buffer.h"
#include "grassland/vulkan_legacy/device.h"
#include "grassland/vulkan_legacy/raytracing/ray_tracing_pipeline.h"

namespace grassland::vulkan_legacy::raytracing {
class ShaderBindingTable {
 public:
  explicit ShaderBindingTable(RayTracingPipeline *ray_tracing_pipeline);
  [[nodiscard]] VkDeviceAddress GetRayGenDeviceAddress() const;
  [[nodiscard]] VkDeviceAddress GetMissDeviceAddress() const;
  [[nodiscard]] VkDeviceAddress GetClosestHitDeviceAddress() const;

 private:
  std::unique_ptr<Buffer> buffer_;
  VkDeviceAddress buffer_address_;
  VkDeviceAddress ray_gen_offset_;
  VkDeviceAddress miss_offset_;
  VkDeviceAddress closest_hit_offset_;
};
}  // namespace grassland::vulkan_legacy::raytracing
