#pragma once
#include "grassland/grassland.h"
#include "physic_solver.cuh"
using namespace grassland;

struct GlobalUniformObject {
  glm::mat4 world;
  glm::mat4 camera;
};

struct InstanceInfo {
  glm::vec3 offset;
  float size;
  glm::vec4 color;
};

struct Vertex {
  glm::vec2 position;
};

class FluidLarge {
 public:
  FluidLarge(const char * title, int width, int height, const PhysicSettings& physic_settings);
  void Run();
 private:
  void OnInit();
  void OnLoop();
  void OnRender();
  void OnUpdate();
  void OnClose();
  std::unique_ptr<vulkan::framework::Core> core_;
  std::unique_ptr<vulkan::framework::TextureImage> color_frame_;
  std::unique_ptr<vulkan::framework::TextureImage> depth_frame_;
  std::unique_ptr<vulkan::framework::TextureImage> stencil_frame_;

  std::unique_ptr<vulkan::framework::StaticBuffer<GlobalUniformObject>> global_uniform_buffer_;
  std::unique_ptr<vulkan::framework::StaticBuffer<InstanceInfo>> instance_info_buffer_;
  std::unique_ptr<vulkan::framework::StaticBuffer<Vertex>> vertex_buffer_;
  std::unique_ptr<vulkan::framework::StaticBuffer<uint32_t>> index_buffer_;

  std::vector<InstanceInfo> instance_infos_;
  std::unique_ptr<vulkan::framework::RenderNode> render_node_;
  std::unique_ptr<PhysicSolver> physic_solver_;
};
