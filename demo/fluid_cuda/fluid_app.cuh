#pragma once
#include "camera.h"
#include "grassland/grassland.h"
#include "grid.h"
#include "params.h"

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
};

struct RenderInfo {
  glm::mat4 model;
  glm::vec4 color;
};

struct FluidAppSettings {
  uint32_t num_particle{NUM_PARTICLE};
};

struct Particle {
  glm::vec3 position;
  glm::vec3 velocity;
  int type;
};

struct MACGridContent {
  float vel[2]{};
  float weight[2]{};
  float rho{};
  friend std::ostream &operator<<(std::ostream &os,
                                  const MACGridContent &content) {
    os << content.rho;
    return os;
  }
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

  void DrawObject(int model_id, glm::mat4 model, glm::vec4 color);
  void DrawLine(const glm::vec3 &v0,
                const glm::vec3 &v1,
                float thickness,
                const glm::vec4 &color = glm::vec4{1.0f, 1.0f, 1.0f, 1.0f});
  void DrawSphere(const glm::vec3 &origin,
                  float radius,
                  const glm::vec4 &color);
  void DrawObjects();
  void UpdateCamera();
  void UpdateDynamicInfos();

  void UpdatePhysicalSystem();
  void CalcPressureImpactToDivergence(
      const thrust::device_vector<float> &pressure,
      thrust::device_vector<float> &delta_divergence);
  void SolvePressure();

  void RegisterSphere();
  void RegisterCylinder();
  void UpdateImGui();

  int RegisterModel(const std::vector<Vertex> &vertices,
                    const std::vector<uint32_t> &indices);

  void InitParticles();
  void PlotMatrix();

  void OutputXYZFile();

  FluidAppSettings settings_{};
  std::unique_ptr<grassland::vulkan_legacy::framework::Core> core_;
  std::unique_ptr<grassland::vulkan_legacy::framework::RenderNode> render_node_;
  std::vector<std::pair<
      std::unique_ptr<
          grassland::vulkan_legacy::framework::StaticBuffer<Vertex>>,
      std::unique_ptr<
          grassland::vulkan_legacy::framework::StaticBuffer<uint32_t>>>>
      object_models_;
  std::unique_ptr<
      grassland::vulkan_legacy::framework::DynamicBuffer<RenderInfo>>
      render_info_buffer_;
  std::unique_ptr<
      grassland::vulkan_legacy::framework::DynamicBuffer<CameraObject>>
      camera_object_buffer_;
  std::unique_ptr<grassland::vulkan_legacy::framework::TextureImage>
      frame_image_;
  std::unique_ptr<grassland::vulkan_legacy::framework::TextureImage>
      depth_buffer_;

  std::vector<int> render_objects_;
  std::vector<RenderInfo> render_infos_;

  Camera camera_{};
  int sphere_model_id_{};
  int cylinder_model_id_{};

  /* Fluid Content */
  float delta_t_{1e-2f};
  std::vector<Particle> particles_;
  Grid<MACGridContent> u_field_;
  Grid<MACGridContent> v_field_;
  Grid<MACGridContent> w_field_;
  Grid<float> u_border_coe_;
  Grid<float> v_border_coe_;
  Grid<float> w_border_coe_;
  Grid<float> level_set_;
  thrust::device_vector<float> pressure_{GRID_SIZE_X * GRID_SIZE_Y *
                                         GRID_SIZE_Z};
  thrust::device_vector<float> divergence_{GRID_SIZE_X * GRID_SIZE_Y *
                                           GRID_SIZE_Z};
  thrust::device_vector<float> buffer_{GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z};
  thrust::device_vector<float> r_vec_{GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z};
  thrust::device_vector<float> p_vec_{GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z};
  thrust::device_vector<float> Ap_vec_{GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z};

  glm::vec4 air_particle_color{1.0f, 0.5f, 0.5f, 1.0f};
  glm::vec4 liq_particle_color{0.5f, 0.5f, 1.0f, 1.0f};
  float air_particle_size{0.04};
  float liq_particle_size{0.1};
  bool show_escaped_particles{false};
  glm::vec3 gravity{GRAVITY};
  bool pause_{false};
};
