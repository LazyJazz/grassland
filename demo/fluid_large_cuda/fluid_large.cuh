#pragma once
#include "grassland/grassland.h"
using namespace grassland;

class FluidLarge {
 public:
  FluidLarge(const char * title, int width, int height);
  void Run();
 private:
  void OnInit();
  void OnLoop();
  void OnClose();
  std::unique_ptr<vulkan::framework::Core> core_;
  std::unique_ptr<vulkan::framework::TextureImage> color_frame_;
  std::unique_ptr<vulkan::framework::TextureImage> depth_frame_;
  std::unique_ptr<vulkan::framework::TextureImage> stencil_frame_;
};
