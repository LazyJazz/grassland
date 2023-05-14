#pragma once
#include "grassland/grassland.h"

using namespace grassland;

class Application {
 public:
  Application(const char *title, int width, int height);
  ~Application();
  void Run();

 private:
  void OnInit();
  void OnLoop();
  void OnClose();
  GLFWwindow *window_;
  std::unique_ptr<vulkan::Instance> instance_;
  std::unique_ptr<vulkan::Device> device_;
  std::unique_ptr<vulkan::Surface> surface_;
  std::unique_ptr<vulkan::Swapchain> swapchain_;
};
