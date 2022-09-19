#pragma once
#include <GLFW/glfw3.h>
#include <grassland/vulkan/vulkan.h>

#include <memory>

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
  std::unique_ptr<grassland::vulkan::Instance> instance_;
};
