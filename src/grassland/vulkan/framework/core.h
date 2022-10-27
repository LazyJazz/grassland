#pragma once
#include <grassland/vulkan/framework/core_settings.h>
#include <grassland/vulkan/vulkan.h>

namespace grassland::vulkan::framework {
class Core {
 public:
  explicit Core(const CoreSettings &core_settings);
  ~Core();
  [[nodiscard]] const CoreSettings &GetCoreSettings() const;
  [[nodiscard]] Instance *GetInstance() const;
  [[nodiscard]] PhysicalDevice *GetPhysicalDevice() const;
  [[nodiscard]] Device *GetDevice() const;
  [[nodiscard]] Surface *GetSurface() const;
  [[nodiscard]] Swapchain *GetSwapchain() const;
  [[nodiscard]] CommandPool *GetCommandPool() const;
  [[nodiscard]] GLFWwindow *GetWindow() const;

 private:
  CoreSettings core_settings_;

  std::unique_ptr<Instance> instance_;
  std::unique_ptr<PhysicalDevice> physical_device_;
  std::unique_ptr<Device> device_;

  GLFWwindow *window_;
  std::unique_ptr<Surface> surface_;
  std::unique_ptr<Swapchain> swapchain_;

  std::unique_ptr<CommandPool> command_pool_;
  std::vector<std::unique_ptr<CommandBuffer>> command_buffers_;

  std::vector<std::unique_ptr<Fence>> in_flight_fences_;
  std::vector<std::unique_ptr<Semaphore>> image_available_semaphores_;
  std::vector<std::unique_ptr<Semaphore>> render_finish_semaphores_;
};
}  // namespace grassland::vulkan::framework
