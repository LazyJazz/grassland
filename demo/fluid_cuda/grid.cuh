#pragma once
#include "glm/glm.hpp"
#include "thrust/device_vector.h"
#include "thrust/device_new.h"
#include "thrust/device_delete.h"

template<class T>
class Grid {
 public:
  Grid() = default;
  Grid(glm::ivec3 size_grid, float delta_x_, glm::vec3 offset);
  ~Grid();
  [[nodiscard]] glm::mat4 MapMatrix() const;
 private:
  glm::ivec3 size_grid_{1, 1, 1};
  float delta_x_{1.0f};
  glm::vec3 offset_{0.0f};
  thrust::device_ptr<T> dev_mem_{};
};

template <class T>
glm::mat4 Grid<T>::MapMatrix() const {
  return {
      delta_x_, 0.0f, 0.0f, 0.0f,
      0.0f, delta_x_, 0.0f, 0.0f,
      0.0f, 0.0f, delta_x_, 0.0f,
      offset_.x, offset_.y, offset_.z, 1.0f
  };
}

template <class T>
Grid<T>::Grid(glm::ivec3 size_grid, float delta_x, glm::vec3 offset): size_grid_(size_grid), delta_x_(delta_x), offset_(offset) {
  dev_mem_ = thrust::device_new<T>(size_grid_.x * size_grid_.y * size_grid_.z);
}

template <class T>
Grid<T>::~Grid() {
  thrust::device_delete(dev_mem_, size_grid_.x * size_grid_.y * size_grid_.z);
}
