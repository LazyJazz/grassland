#pragma once
#include <grassland/vulkan/framework/core_settings.h>
#include <grassland/vulkan/vulkan.h>

namespace grassland::vulkan::framework {
class TextureImage;
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
  [[nodiscard]] CommandBuffer *GetCommandBuffer(int frame_index) const;
  [[nodiscard]] CommandBuffer *GetCommandBuffer() const;
  [[nodiscard]] int GetCurrentFrameIndex() const;

  void BeginCommandRecord();
  void EndCommandRecordAndSubmit();

  void Output(TextureImage *texture_image);

  void SetWindowSizeCallback(
      const std::function<void(int width, int height)> &window_size_callback);

 private:
  static void GLFWWindowSizeFunc(GLFWwindow *window, int width, int height);

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

  int frame_index_{0};
  uint32_t current_image_index{0};

  std::function<void(int width, int height)> custom_window_size_function_;
};
}  // namespace grassland::vulkan::framework
