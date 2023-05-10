#pragma once
#include "grassland/font/font.h"
#include "grassland/vulkan_legacy/framework/framework.h"
#include "grassland/vulkan_legacy/vulkan.h"

struct UniformBufferObject {
  glm::mat4 trans_mat;
  glm::vec4 color;
};

class FontViewer {
 public:
  explicit FontViewer(grassland::font::Mesh font_mesh);
  void Run();

 private:
  void OnInit();
  void OnLoop();
  void OnClose();
  grassland::vulkan_legacy::framework::CoreSettings core_settings_;
  std::unique_ptr<grassland::vulkan_legacy::framework::Core> core_;
  std::unique_ptr<grassland::vulkan_legacy::framework::RenderNode> render_node_;
  std::unique_ptr<grassland::vulkan_legacy::framework::StaticBuffer<glm::vec2>>
      vertex_buffer_;
  std::unique_ptr<grassland::vulkan_legacy::framework::StaticBuffer<uint32_t>>
      index_buffer_;
  std::unique_ptr<
      grassland::vulkan_legacy::framework::DynamicBuffer<UniformBufferObject>>
      uniform_buffer_;
  grassland::font::Mesh font_mesh_;
};
