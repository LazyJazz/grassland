#pragma once
#include "cuda_runtime.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "grassland/grassland.h"
#include "nbody_cuda.h"
#include "params.h"
#include "random"

using namespace grassland;

struct GlobalUniformObject {
  glm::mat4 world_to_screen;
  glm::mat4 camera_to_world;
  float particle_size;
};

class NBody {
 public:
  explicit NBody(int n_particles = NUM_PARTICLE);
  void Run();

 private:
  void OnInit();
  void OnUpdate();
  void OnRender();
  void OnClose();

  void BuildRenderNode();

  void UpdateParticles();
  void UpdateImGui();

  float RandomFloat();
  glm::vec3 RandomOnSphere();
  glm::vec3 RandomInSphere();

  std::unique_ptr<vulkan::framework::Core> core_;
  std::unique_ptr<vulkan::framework::RenderNode> render_node_;
  std::unique_ptr<vulkan::framework::DynamicBuffer<GlobalUniformObject>>
      global_uniform_object_;
  std::unique_ptr<vulkan::framework::DynamicBuffer<glm::vec4>>
      particle_positions_;
  std::unique_ptr<vulkan::framework::StaticBuffer<glm::vec2>> vertex_buffer_;
  std::unique_ptr<vulkan::framework::StaticBuffer<uint32_t>> index_buffer_;
  std::unique_ptr<vulkan::framework::TextureImage> frame_buffer_;
  std::vector<glm::vec4> positions_;
  std::vector<glm::vec4> velocities_;
  int n_particles_{4096};
  std::mt19937 random_device_{};
  glm::mat4 rotation{1.0f};
};
