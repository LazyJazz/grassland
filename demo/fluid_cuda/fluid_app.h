#pragma once
#include "grassland/grassland.h"

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
};

struct RenderInfo {
  glm::vec3 position;
  glm::vec3 color;
};

struct FluidAppSettings {
  uint32_t num_particle{65536};
};

class FluidApp {
 public:
  FluidApp(const FluidAppSettings &settings = FluidAppSettings());
  void Run();

 private:
  void OnInit();
  void OnLoop();
  void OnClose();

  void OnUpdate();
  void OnRender();

  FluidAppSettings settings_{};
  std::unique_ptr<grassland::vulkan::framework::Core> core_;
  std::unique_ptr<grassland::vulkan::framework::RenderNode> render_node_;
  std::unique_ptr<grassland::vulkan::framework::StaticBuffer<Vertex>>
      vertex_buffer_;
  std::unique_ptr<grassland::vulkan::framework::StaticBuffer<uint32_t>>
      index_buffer_;
  std::unique_ptr<grassland::vulkan::framework::DynamicBuffer<RenderInfo>>
      particle_buffer_;
  std::unique_ptr<grassland::vulkan::framework::TextureImage> frame_image_;
};
