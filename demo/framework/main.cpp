#include <grassland/util/util.h>
#include <grassland/vulkan/framework/framework.h>
#include <grassland/vulkan/vulkan.h>

#include "glm/glm.hpp"

using namespace grassland::vulkan::framework;

namespace {

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
    {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, 0.0f}},
    {{-1.0f, -1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {{-1.0f, 1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
    {{-1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}},
    {{1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},
    {{1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 1.0f}},
    {{1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, 0.0f}},
    {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}};

const std::vector<uint32_t> indices = {
    0b000, 0b010, 0b001, 0b011, 0b001, 0b010, 0b100, 0b101, 0b110,
    0b111, 0b110, 0b101, 0b000, 0b100, 0b010, 0b110, 0b010, 0b100,
    0b001, 0b011, 0b101, 0b111, 0b101, 0b011, 0b000, 0b001, 0b100,
    0b001, 0b101, 0b100, 0b010, 0b110, 0b011, 0b111, 0b011, 0b110};
}  // namespace

int main() {
  grassland::vulkan::framework::CoreSettings core_settings;
  core_settings.raytracing_pipeline_required = false;
  core_settings.window_title = "Hello, World!";
  Core core(core_settings);

  auto vertex_buffer =
      std::make_unique<StaticBuffer<Vertex>>(&core, vertices.size());
  vertex_buffer->Upload(vertices.data());

  auto index_buffer =
      std::make_unique<StaticBuffer<uint32_t>>(&core, indices.size());
  index_buffer->Upload(indices.data());

  auto uniform_buffer = std::make_unique<DynamicBuffer<UniformBufferObject>>(
      &core, size_t(1), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

  std::unique_ptr<RenderNode> render_node = std::make_unique<RenderNode>(&core);

  render_node->AddColorOutput(core.GetSwapchain()->GetFormat());
  render_node->EnableDepthTest();
  render_node->AddShader("../shaders/color_shader.vert.spv",
                         VK_SHADER_STAGE_VERTEX_BIT);
  render_node->AddShader("../shaders/color_shader.frag.spv",
                         VK_SHADER_STAGE_FRAGMENT_BIT);
  render_node->AddUniformBinding(uniform_buffer.get(),
                                 VK_SHADER_STAGE_VERTEX_BIT);
  render_node->VertexInput(
      {VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT});
  render_node->BuildRenderNode(core_settings.window_width,
                               core_settings.window_height);

  while (!glfwWindowShouldClose(core.GetWindow())) {
    glfwPollEvents();
  }
}
