#pragma once
#include <memory>

#include "GLFW/glfw3.h"
#include "grassland/grassland.h"

using namespace grassland;

constexpr int kMaxFramesInFlight = 3;

class App {
 public:
  App(int width, int height, const char *title);
  ~App();
  void Run();

 private:
  void OnCreate();
  void OnInit();
  void OnLoop();
  void OnClose();
  void OnDestroy();

  void OnUpdate();
  void OnRender();

  void recreateSwapChain();
  void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

  GLFWwindow *window_;
  std::unique_ptr<vulkan_legacy::Instance> instance_;
  std::unique_ptr<vulkan_legacy::Surface> surface_;
  std::unique_ptr<vulkan_legacy::PhysicalDevice> physical_device_;
  std::unique_ptr<vulkan_legacy::Device> device_;
  std::unique_ptr<vulkan_legacy::Queue> present_queue_;
  std::unique_ptr<vulkan_legacy::Swapchain> swapchain_;
  std::unique_ptr<vulkan_legacy::RenderPass> render_pass_;

  std::unique_ptr<vulkan_legacy::DescriptorSetLayout> descriptor_set_layout_;

  std::unique_ptr<vulkan_legacy::PipelineLayout> pipeline_layout_;
  std::unique_ptr<vulkan_legacy::Pipeline> graphics_pipeline_;
  std::vector<std::unique_ptr<vulkan_legacy::Framebuffer>> framebuffers_;
  std::unique_ptr<vulkan_legacy::CommandPool> command_pool_;
  std::unique_ptr<vulkan_legacy::CommandBuffers> command_buffers_;

  std::vector<std::unique_ptr<vulkan_legacy::Semaphore>>
      image_available_semaphores_;
  std::vector<std::unique_ptr<vulkan_legacy::Semaphore>>
      render_finished_semaphores_;
  std::vector<std::unique_ptr<vulkan_legacy::Fence>> in_flight_fence_;

  std::unique_ptr<vulkan_legacy::Buffer> index_buffer_;
  std::unique_ptr<vulkan_legacy::Buffer> vertex_buffer_;

  std::vector<std::unique_ptr<vulkan_legacy::Buffer>> uniform_buffers_;
  std::unique_ptr<vulkan_legacy::DescriptorPool> descriptor_pool_;
  std::unique_ptr<vulkan_legacy::DescriptorSets> descriptor_sets_;

  bool framebufferResized{false};
  int currentFrame{0};
};
