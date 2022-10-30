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

const std::vector<uint16_t> indices = {
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

  std::unique_ptr<StaticBuffer<Vertex>> vertex_buffer =
      std::make_unique<StaticBuffer<Vertex>>(&core, vertices.size());
  vertex_buffer->Upload(vertices.data());
  std::vector<Vertex> another_vertex_buffer(vertices.size());
  vertex_buffer->Download(another_vertex_buffer.data());
  for (auto vertex : another_vertex_buffer) {
    LAND_INFO("({}, {}, {}) ({}, {}, {})", vertex.pos.x, vertex.pos.y,
              vertex.pos.z, vertex.color.r, vertex.color.g, vertex.color.b);
  }
  while (!glfwWindowShouldClose(core.GetWindow())) {
    glfwPollEvents();
  }
}
