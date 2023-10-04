#pragma once
#include <set>

#include "grassland/vulkan/core/device_procedures.h"
#include "grassland/vulkan/core/instance.h"
#include "grassland/vulkan/core/physical_device.h"
#include "grassland/vulkan/core/queue.h"
#include "grassland/vulkan/core/surface.h"

namespace grassland::vulkan {

struct DeviceSettings {
  PhysicalDevice physical_device;
  Surface *surface;
  bool enable_raytracing;
};

class Device {
 public:
  explicit Device(Instance *instance,
                  const class PhysicalDevice &physical_device,
                  Surface *surface = nullptr,
                  bool enable_raytracing = false);
  Device(Instance *instance, const DeviceSettings &settings);
  ~Device();

  [[nodiscard]] VkDevice Handle() const;
  [[nodiscard]] class PhysicalDevice PhysicalDevice() const;
  [[nodiscard]] Queue GraphicsQueue() const;
  [[nodiscard]] Queue PresentQueue() const;
  [[nodiscard]] Queue SingleTimeCommandQueue() const;
  [[nodiscard]] uint32_t GraphicsQueueFamilyIndex() const {
    return graphics_queue_.QueueFamilyIndex();
  }
  [[nodiscard]] uint32_t PresentQueueFamilyIndex() const {
    return present_queue_.QueueFamilyIndex();
  }
  DeviceProcedures &Procedures() {
    return device_procedures_;
  }

  [[nodiscard]] VmaAllocator Allocator() const {
    return allocator_;
  }

  void WaitIdle() const;

  void NameObject(VkImage image, const std::string &name);
  void NameObject(VkImageView image_view, const std::string &name);
  void NameObject(VkBuffer buffer, const std::string &name);
  void NameObject(VkDeviceMemory memory, const std::string &name);
  void NameObject(VkAccelerationStructureKHR acceleration_structure,
                  const std::string &name);
  void NameObject(VkPipeline pipeline, const std::string &name);
  void NameObject(VkPipelineLayout pipeline_layout, const std::string &name);
  void NameObject(VkShaderModule shader_module, const std::string &name);
  void NameObject(VkDescriptorSetLayout descriptor_set_layout,
                  const std::string &name);
  void NameObject(VkDescriptorPool descriptor_pool, const std::string &name);
  void NameObject(VkDescriptorSet descriptor_set, const std::string &name);
  void NameObject(VkCommandPool command_pool, const std::string &name);
  void NameObject(VkCommandBuffer command_buffer, const std::string &name);
  void NameObject(VkFramebuffer framebuffer, const std::string &name);
  void NameObject(VkRenderPass render_pass, const std::string &name);
  void NameObject(VkSampler sampler, const std::string &name);

 private:
  Instance *instance_{};
  class PhysicalDevice physical_device_;

  VkDevice device_{};
  Queue graphics_queue_{};
  Queue single_time_command_queue_{};
  Queue present_queue_{};
  DeviceProcedures device_procedures_{};
  VmaAllocator allocator_{};
};
}  // namespace grassland::vulkan
