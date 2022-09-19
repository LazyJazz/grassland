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
};
