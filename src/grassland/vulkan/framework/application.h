#pragma once
#include <grassland/vulkan/vulkan.h>

namespace grassland::vulkan::framework {
constexpr int kMaxFramesInFlight = 3;

class Application {
 public:
  Application(int width, int height, const char *title);
  virtual ~Application();
  void Run();

 protected:
  void OutputImage(CommandBuffer *command_buffer, Image *image);

 private:
  void OnCreate();
  void OnInit();
  void OnLoop();
  void OnClose();
  void OnDestroy();

  void OnUpdate();
  void OnRender();

  void recreateSwapChain();
  void recordCommandBuffer(CommandBuffer *command_buffer);

  GLFWwindow *window_;
  std::unique_ptr<vulkan::Instance> instance_;
  std::unique_ptr<vulkan::Surface> surface_;
  std::unique_ptr<vulkan::PhysicalDevice> physical_device_;
  std::unique_ptr<vulkan::Device> device_;
  std::unique_ptr<vulkan::Queue> present_queue_;
  std::unique_ptr<vulkan::Swapchain> swapchain_;
  std::unique_ptr<vulkan::RenderPass> render_pass_;

  std::unique_ptr<vulkan::DescriptorSetLayout> descriptor_set_layout_;

  std::unique_ptr<vulkan::PipelineLayout> pipeline_layout_;
  std::unique_ptr<vulkan::Pipeline> graphics_pipeline_;
  std::vector<std::unique_ptr<vulkan::Framebuffer>> framebuffers_;
  std::unique_ptr<vulkan::CommandPool> command_pool_;
  std::vector<std::unique_ptr<vulkan::CommandBuffer>> command_buffers_;

  std::vector<std::unique_ptr<vulkan::Semaphore>> image_available_semaphores_;
  std::vector<std::unique_ptr<vulkan::Semaphore>> render_finished_semaphores_;
  std::vector<std::unique_ptr<vulkan::Fence>> in_flight_fence_;

  std::unique_ptr<vulkan::Buffer> index_buffer_;
  std::unique_ptr<vulkan::Buffer> vertex_buffer_;

  std::vector<std::unique_ptr<vulkan::Buffer>> uniform_buffers_;
  std::unique_ptr<vulkan::DescriptorPool> descriptor_pool_;
  std::unique_ptr<vulkan::DescriptorSets> descriptor_sets_;

  std::unique_ptr<vulkan::Image> image_;
  std::unique_ptr<vulkan::ImageView> image_view_;
  std::unique_ptr<vulkan::Sampler> sampler_;

  std::unique_ptr<vulkan::Image> depth_buffer_image_;
  std::unique_ptr<vulkan::ImageView> depth_buffer_image_view_;
  std::unique_ptr<vulkan::Image> color_attachment_image_;
  std::unique_ptr<vulkan::ImageView> color_attachment_image_view_;

  bool framebufferResized{false};
  int currentFrame{0};
  uint32_t imageIndex{0};
};

}  // namespace grassland::vulkan::framework
