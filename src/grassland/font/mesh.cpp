#include "grassland/font/mesh.h"

#include "map"

namespace grassland::font {

Mesh::Mesh(const std::vector<glm::vec2> &vertices,
           const std::vector<uint32_t> &indices) {
  vertices_ = vertices;
  indices_ = indices;
}

Mesh::Mesh(const std::vector<glm::vec2> &triangle_vertices) {
  //  vertices_ = triangle_vertices;
  //  for (int i = 0; i < triangle_vertices.size(); i++) {
  //    indices_.push_back(i);
  //  }
  //  return;
  class Vec2Eq {
   public:
    bool operator()(const glm::vec2 &a, const glm::vec2 &b) const {
      return a.x < b.x || (a.x == b.x && a.y < b.y);
    }
  };
  std::map<glm::vec2, uint32_t, Vec2Eq> vec_index;
  auto vec_2_index = [&](const glm::vec2 &v) {
    if (!vec_index.count(v)) {
      uint32_t index = vertices_.size();
      vec_index.insert(std::make_pair(v, index));
      vertices_.push_back(v);
      return index;
    }
    return vec_index.at(v);
  };
  vertices_.clear();
  indices_.clear();
  for (auto &v : triangle_vertices) {
    indices_.push_back(vec_2_index(v));
  }
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
