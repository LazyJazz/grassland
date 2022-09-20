#pragma once
#include <GLFW/glfw3.h>
#include <grassland/vulkan/vulkan.h>

#include <memory>

using namespace grassland;

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
};
