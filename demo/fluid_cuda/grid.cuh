#pragma once
#include "glm/glm.hpp"
#include "thrust/device_vector.h"
#include "thrust/device_new.h"
#include "thrust/device_delete.h"
#include "thrust/copy.h"

template<class T>
class Grid {
 public:
  Grid() = default;
  Grid(glm::ivec3 size_grid, float delta_x, glm::vec3 offset);
  Grid(const Grid &grid);
  ~Grid();
  __host__ __device__ [[nodiscard]] glm::mat4 MapMatrix() const;
  __host__ __device__ glm::ivec3 NearestGrid(const glm::vec3& world_pos);
  __host__ __device__ [[nodiscard]] glm::vec3 WorldToGrid(const glm::vec3& world_pos) const;
  __host__ __device__ [[nodiscard]] glm::vec3 GridToWorld(const glm::vec3& grid_pos) const;
  __host__ __device__ [[nodiscard]] int IndexConv(glm::ivec3 index) const;
  __host__ __device__ [[nodiscard]] int IndexConv(int x, int y, int z) const;
  __device__ T& operator() (int x, int y, int z);
  __device__ const T& operator() (int x, int y, int z) const;
  __device__ T& operator() (glm::ivec3 index);
  __device__ const T& operator() (glm::ivec3 index) const;
  __device__ T& operator[] (int index);
  __device__ const T& operator[] (int index) const;
 private:
  glm::ivec3 size_grid_{1, 1, 1};
  float delta_x_{1.0f};
  glm::vec3 offset_{0.0f};
  T* dev_mem_{};
};

template <class T>
const T& Grid<T>::operator[](int index) const {
  return dev_mem_[index];
}

template <class T>
__device__ T& Grid<T>::operator[](int index) {
  return dev_mem_[index];
}

template <class T>
__host__ __device__ int Grid<T>::IndexConv(int x, int y, int z) const {
  return IndexConv({x, y, z});
}

template <class T>
__host__ __device__ int Grid<T>::IndexConv(glm::ivec3 index) const {
  return (index.x * size_grid_.y + index.y) * size_grid_.z + index.z;
}

template <class T>
__device__ const T &Grid<T>::operator()(glm::ivec3 index) const {
  return dev_mem_[IndexConv(index)];
}

template <class T>
__device__ T &Grid<T>::operator()(glm::ivec3 index) {
  return dev_mem_[IndexConv(index)];
}

template <class T>
__device__ const T &Grid<T>::operator()(int x, int y, int z) const {
  return operator()(glm::ivec3{x, y, z});
}

template <class T>
__device__ T &Grid<T>::operator()(int x, int y, int z) {
  return operator()(glm::ivec3{x, y, z});
}

template <class T>
__host__ __device__ glm::vec3 Grid<T>::GridToWorld(
    const glm::vec3 &grid_pos) const {
  return (grid_pos + offset_) * delta_x_;
}

template <class T>
__host__ __device__ glm::vec3 Grid<T>::WorldToGrid(
    const glm::vec3 &world_pos) const {
  return world_pos / delta_x_ - offset_;
}

template <class T>
__host__ __device__ glm::ivec3 Grid<T>::NearestGrid(
    const glm::vec3 &world_pos) {
  return glm::ivec3(WorldToGrid(world_pos) + 0.5f);
}

template <class T>
Grid<T>::Grid(const Grid &grid): Grid(size_grid_, delta_x_, offset_) {
  thrust::copy(grid.dev_mem_, grid.dev_mem_ + size_grid_.x * size_grid_.y * size_grid_.z, dev_mem_);
}

template <class T>
__host__ __device__ glm::mat4 Grid<T>::MapMatrix() const {
  return {
      delta_x_, 0.0f, 0.0f, 0.0f,
      0.0f, delta_x_, 0.0f, 0.0f,
      0.0f, 0.0f, delta_x_, 0.0f,
      offset_.x, offset_.y, offset_.z, 1.0f
  };
}

template <class T>
Grid<T>::Grid(glm::ivec3 size_grid, float delta_x, glm::vec3 offset): size_grid_(size_grid), delta_x_(delta_x), offset_(offset) {
  cudaMalloc(reinterpret_cast<void**>(&dev_mem_), size_grid_.x * size_grid_.y * size_grid_.z * sizeof(T));
  cudaMemset(dev_mem_, 0, size_grid_.x * size_grid_.y * size_grid_.z * sizeof(T));
}

template <class T>
Grid<T>::~Grid() {
  cudaFree(dev_mem_);
}
