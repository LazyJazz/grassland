#pragma once
#include "grassland/grassland.h"

class RayTracingApp {
 public:
  RayTracingApp(uint32_t width, uint32_t height);
  void Run();

 private:
  void OnInit();
  void OnLoop();
  void OnClose();

  void OnUpdate();
  void OnRender();

  std::unique_ptr<grassland::vulkan::framework::Core> core_;
  std::unique_ptr<grassland::vulkan::framework::TextureImage> frame_texture_;
  std::unique_ptr<
      grassland::vulkan::raytracing::BottomLevelAccelerationStructure>
      bottom_level_acceleration_structure_;
  std::unique_ptr<grassland::vulkan::raytracing::TopLevelAccelerationStructure>
      top_level_acceleration_structure_;
};
