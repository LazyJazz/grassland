#pragma once
#include "glm/glm.hpp"
#include "grassland/grassland.h"
#include "vector"

using namespace grassland;

#define GRID_SIZE_X 20
#define GRID_SIZE_Y 40
#define DELTA_X 1.0f
#define RHO_AIR 0.01f
#define RHO_LIQ 1.0f

#define TYPE_AIR 0
#define TYPE_LIQ 1

#define GRAVITY \
  glm::vec2 {   \
    0.0f, -9.8f \
  }

struct Particle {
  glm::vec2 position;
  glm::vec2 velocity;
  int type;
};

struct RenderObjectInfo {
  glm::mat4 transform;
  glm::vec4 color;
};

class Fluid2D {
 public:
  explicit Fluid2D(int n_particle);
  void Run();

 private:
  void OnInit();
  void OnLoop();
  void OnClose();

  void OnUpdate();
  void OnRender();

  void InitParticles();

  void DrawCircle(glm::vec2 origin, float radius, glm::vec4 color);
  void ComposeScene();
  void UpdateRenderAssets();

  void SolvePressure();
  void SolvePressureImpactToDivergence(const float *pressure,
                                       float *delta_divergence);
  void SolveParticleDynamics();

  static bool InsideContainer(glm::vec2 pos);

  std::unique_ptr<vulkan::framework::Core> core;
  std::unique_ptr<vulkan::framework::RenderNode> render_node;
  std::unique_ptr<vulkan::framework::RenderNode> bkground_render_node;
  std::unique_ptr<vulkan::framework::TextureImage> framebuffer;
  std::unique_ptr<vulkan::framework::TextureImage> texture_image;
  std::unique_ptr<vulkan::Sampler> sampler;
  std::unique_ptr<vulkan::framework::StaticBuffer<glm::vec2>> vertex_buffer;
  std::unique_ptr<vulkan::framework::StaticBuffer<uint32_t>> index_buffer;
  std::unique_ptr<vulkan::framework::StaticBuffer<glm::vec2>>
      rectangle_vertex_buffer;
  std::unique_ptr<vulkan::framework::StaticBuffer<uint32_t>>
      rectangle_index_buffer;
  std::unique_ptr<vulkan::framework::DynamicBuffer<RenderObjectInfo>>
      render_objects;
  std::unique_ptr<vulkan::framework::StaticBuffer<glm::mat4>>
      global_transform_buffer;
  std::vector<RenderObjectInfo> render_object_infos;

  std::vector<Particle> particles;

  float delta_t = 0.01f;
  float u_grid[2][GRID_SIZE_X + 1][GRID_SIZE_Y];
  float v_grid[2][GRID_SIZE_X][GRID_SIZE_Y + 1];
  float u_weight_grid[2][GRID_SIZE_X + 1][GRID_SIZE_Y];
  float v_weight_grid[2][GRID_SIZE_X][GRID_SIZE_Y + 1];
  float level_set[GRID_SIZE_X + 1][GRID_SIZE_Y + 1];
  float divergence[GRID_SIZE_X * GRID_SIZE_Y];
  float pressure[GRID_SIZE_X * GRID_SIZE_Y];
  float buffer[GRID_SIZE_X * GRID_SIZE_Y];
  float r_vec[GRID_SIZE_X * GRID_SIZE_Y];
  float p_vec[GRID_SIZE_X * GRID_SIZE_Y];
  float Ap_vec[GRID_SIZE_X * GRID_SIZE_Y];

  glm::mat3 u_grid_transform{1.0f / DELTA_X, 0.0f, 0.0f,  0.0f, 1.0f / DELTA_X,
                             0.0f,           0.0f, -0.5f, 1.0f};
  glm::mat3 v_grid_transform{1.0f / DELTA_X, 0.0f,  0.0f, 0.0f, 1.0f / DELTA_X,
                             0.0f,           -0.5f, 0.0f, 1.0f};
  glm::mat3 c_grid_transform{1.0f / DELTA_X, 0.0f, 0.0f, 0.0f, 1.0f / DELTA_X,
                             0.0f,           0.0f, 0.0f, 1.0f};

  glm::vec4 bkground_image[1024][512];
};
