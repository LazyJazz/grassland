#pragma once
#include "grassland/grassland.h"

class GuiExample {
 public:
  GuiExample();
  void Run();

 private:
  void OnInit();
  void OnLoop();
  void OnClose();
  std::unique_ptr<grassland::vulkan::framework::Core> core_;
  std::unique_ptr<grassland::vulkan::framework::TextureImage> paint_buffer_;
  std::unique_ptr<grassland::vulkan::gui::Manager> manager_;
  std::unique_ptr<grassland::vulkan::gui::Model> test_model_;
};
