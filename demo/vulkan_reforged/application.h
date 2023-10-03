#pragma once
#include "grassland/grassland.h"

using namespace grassland;

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

  std::unique_ptr<vulkan::ShaderModule> vertex_shader_;
  std::unique_ptr<vulkan::ShaderModule> fragment_shader_;

  bool application_should_close_{};
};
