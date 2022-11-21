#include "grassland/font/mesh.h"

namespace grassland::font {

Mesh::Mesh(const std::vector<glm::vec2> &vertices,
           const std::vector<uint32_t> &indices) {
  vertices_ = vertices;
  indices_ = indices;
}

std::vector<glm::vec2> &Mesh::GetVertices() {
  return vertices_;
}

const std::vector<glm::vec2> &Mesh::GetVertices() const {
  return vertices_;
}

std::vector<uint32_t> &Mesh::GetIndices() {
  return indices_;
}

const std::vector<uint32_t> &Mesh::GetIndices() const {
  return indices_;
}
}  // namespace grassland::font
