#pragma once
#include "grassland/grassland.h"

struct CameraObject {
  glm::mat4 screen_to_camera;
  glm::mat4 camera_to_world;
};

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
  std::unique_ptr<grassland::vulkan::framework::StaticBuffer<CameraObject>>
      camera_object_buffer_;
  std::unique_ptr<grassland::vulkan::DescriptorSetLayout>
      descriptor_set_layout_;
  std::unique_ptr<grassland::vulkan::PipelineLayout> pipeline_layout_;
  std::unique_ptr<grassland::vulkan::DescriptorPool> descriptor_pool_;
  std::unique_ptr<grassland::vulkan::DescriptorSet> descriptor_set_;
  std::unique_ptr<grassland::vulkan::raytracing::RayTracingPipeline>
      ray_tracing_pipeline_;
  std::unique_ptr<grassland::vulkan::raytracing::ShaderBindingTable>
      shader_binding_table_;
};
