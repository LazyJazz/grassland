#pragma once
#include <GLFW/glfw3.h>
#include <grassland/vulkan/vulkan.h>

#include <memory>

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
  std::unique_ptr<vulkan::Instance> instance_;
  std::unique_ptr<vulkan::Surface> surface_;
  std::unique_ptr<vulkan::PhysicalDevice> physical_device_;
  std::unique_ptr<vulkan::Device> device_;
  std::unique_ptr<vulkan::Queue> graphics_queue_;
  std::unique_ptr<vulkan::Queue> present_queue_;
  std::unique_ptr<vulkan::SwapChain> swap_chain_;
  std::unique_ptr<vulkan::RenderPass> render_pass_;
  std::unique_ptr<vulkan::PipelineLayout> pipeline_layout_;
  std::unique_ptr<vulkan::Pipeline> pipeline_graphics_;
  std::vector<std::unique_ptr<vulkan::FrameBuffer>> frame_buffers_;
  std::unique_ptr<vulkan::CommandPool> command_pool_;
  std::unique_ptr<vulkan::CommandBuffers> command_buffers_;

  std::vector<std::unique_ptr<vulkan::Semaphore>> image_available_semaphores_;
  std::vector<std::unique_ptr<vulkan::Semaphore>> render_finished_semaphores_;
  std::vector<std::unique_ptr<vulkan::Fence>> in_flight_fence_;

  std::unique_ptr<vulkan::Buffer> index_buffer;
  std::unique_ptr<vulkan::Buffer> vertex_buffer;

  bool framebufferResized{false};
};
