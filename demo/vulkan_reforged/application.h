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
  bool application_should_close_{};
};
