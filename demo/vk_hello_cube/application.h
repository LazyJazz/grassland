#pragma once
#include "grassland/grassland.h"

using namespace grassland;

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

class Application {
 public:
  Application(const std::string &name, int width, int height, bool headless);
  void Run();

 private:
  void OnUpdate();
  void OnRender();
  void OnClose();
  void OnInit();

  GLFWwindow *window_{};
  std::unique_ptr<vulkan::Core> core_;
  std::unique_ptr<vulkan::Buffer> vertex_buffer_;
  std::unique_ptr<vulkan::Buffer> index_buffer_;

  std::vector<std::unique_ptr<vulkan::Buffer>> uniform_buffers_;
  std::unique_ptr<vulkan::Buffer> host_uniform_buffers_;

  std::unique_ptr<vulkan::ShaderModule> vertex_shader_;
  std::unique_ptr<vulkan::ShaderModule> fragment_shader_;

  std::unique_ptr<vulkan::DescriptorSetLayout> descriptor_set_layout_;
  std::unique_ptr<vulkan::DescriptorPool> descriptor_pool_;
  std::vector<std::unique_ptr<vulkan::DescriptorSet>> descriptor_sets_;

  std::unique_ptr<vulkan::RenderPass> render_pass_;
  std::unique_ptr<vulkan::PipelineLayout> pipeline_layout_;

  std::unique_ptr<vulkan::Pipeline> pipeline_;

  std::unique_ptr<vulkan::Image> framebuffer_image_;
  std::unique_ptr<vulkan::Framebuffer> framebuffer_;

  bool application_should_close_{};
  std::string name_;
};
