#pragma once
#include "camera.h"
#include "grassland/grassland.h"

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
};

struct RenderInfo {
  glm::mat4 model;
  glm::vec4 color;
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

  void DrawObject(int model_id, glm::mat4 model, glm::vec4 color);
  void Line(const glm::vec3 &v0,
            const glm::vec3 &v1,
            float thickness,
            const glm::vec4 &color = glm::vec4{1.0f, 1.0f, 1.0f, 1.0f});
  void Sphere(const glm::vec3 &origin, float radius, const glm::vec4 &color);
  void DrawObjects();
  void UpdateCamera();
  void UpdateDynamicInfos();

  void RegisterSphere();
  void RegisterCylinder();

  int RegisterModel(const std::vector<Vertex> &vertices,
                    const std::vector<uint32_t> &indices);

  FluidAppSettings settings_{};
  std::unique_ptr<grassland::vulkan::framework::Core> core_;
  std::unique_ptr<grassland::vulkan::framework::RenderNode> render_node_;
  std::vector<std::pair<
      std::unique_ptr<grassland::vulkan::framework::StaticBuffer<Vertex>>,
      std::unique_ptr<grassland::vulkan::framework::StaticBuffer<uint32_t>>>>
      object_models_;
  std::unique_ptr<grassland::vulkan::framework::DynamicBuffer<RenderInfo>>
      render_info_buffer_;
  std::unique_ptr<grassland::vulkan::framework::DynamicBuffer<CameraObject>>
      camera_object_buffer_;
  std::unique_ptr<grassland::vulkan::framework::TextureImage> frame_image_;
  std::unique_ptr<grassland::vulkan::framework::TextureImage> depth_buffer_;

  std::vector<int> render_objects_;
  std::vector<RenderInfo> render_infos_;

  Camera camera_{};
  int sphere_model_id_{};
  int cylinder_model_id_{};
};
