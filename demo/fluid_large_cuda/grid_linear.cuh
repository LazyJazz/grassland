#pragma once
#include "glm/glm.hpp"
#include "thrust/device_vector.h"
#include "util.cuh"

template <class ElementType>
class GridLinearHost;

template <class ElementType>
class GridLinear {
 public:
  GridLinear() = default;
  explicit GridLinear(const glm::ivec3 &range);
  struct DevRef {
    DevRef(GridLinear<ElementType> &grid);
    __device__ ElementType &operator()(const glm::ivec3 &index);
    __device__ const ElementType &operator()(const glm::ivec3 &index) const;
    __device__ ElementType &operator()(int x, int y, int z);
    __device__ const ElementType &operator()(int x, int y, int z) const;
    __device__ ElementType Sample(const glm::vec3 &index) const;
    [[nodiscard]] __device__ glm::ivec3 Range() const {
      return grid_range_;
    }
    [[nodiscard]] __device__ int Size() const {
      return grid_range_.x * grid_range_.y * grid_range_.z;
    }
    [[nodiscard]] int ToIndex(const glm::ivec3 &v) const {
      return RANGE_INDEX(v, grid_range_);
    }
    [[nodiscard]] int ToIndex(int x, int y, int z) const {
      return RANGE_INDEX_XYZ(x, y, z, grid_range_);
    }

   private:
    ElementType *ptr_elements_;
    glm::ivec3 grid_range_;
  };
  [[nodiscard]] glm::ivec3 Range() const {
    return grid_range_;
  }
  [[nodiscard]] size_t Size() const {
    return grid_range_.x * grid_range_.y * grid_range_.z;
  }
  void Clear();
  static size_t BufferSize(glm::ivec3 range);
  thrust::device_vector<ElementType> &Vector() {
    return elements_;
  }
  const thrust::device_vector<ElementType> &Vector() const {
    return elements_;
  }

 private:
  friend class GridLinearHost<ElementType>;
  thrust::device_vector<ElementType> elements_;
  glm::ivec3 grid_range_;
};

template <class ElementType>
__device__ ElementType
GridLinear<ElementType>::DevRef::Sample(const glm::vec3 &index) const {
  glm::ivec3 base{floor(index.x), floor(index.y), floor(index.z)};
  glm::vec3 residual{index - glm::vec3{base}};
  glm::ivec3 sample_index{};
  ElementType result{};
  for (int dx = 0; dx < 2; dx++) {
    float sx = (1.0f - dx) + (-1.0f + 2.0f * dx) * residual.x;
    sample_index.x = max(0, min(base.x + dx, grid_range_.x - 1));
    for (int dy = 0; dy < 2; dy++) {
      float sy = (1.0f - dy) + (-1.0f + 2.0f * dy) * residual.y;
      sample_index.y = max(0, min(base.y + dx, grid_range_.y - 1));
      for (int dz = 0; dz < 2; dz++) {
        float sz = (1.0f - dz) + (-1.0f + 2.0f * dz) * residual.z;
        sample_index.z = max(0, min(base.z + dx, grid_range_.z - 1));
        result =
            result + ptr_elements_[RANGE_INDEX(sample_index, grid_range_)] *
                         (sx * sy * sz);
      }
    }
  }
  return result;
}

template <class ElementType>
size_t GridLinear<ElementType>::BufferSize(glm::ivec3 range) {
  return range.x * range.y * range.z;
}

template <class ElementType>
__device__ ElementType &GridLinear<ElementType>::DevRef::operator()(
    const glm::ivec3 &index) {
  return operator()(index.x, index.y, index.z);
}

template <class ElementType>
__device__ const ElementType &GridLinear<ElementType>::DevRef::operator()(
    const glm::ivec3 &index) const {
  return operator()(index.x, index.y, index.z);
}

template <class ElementType>
void GridLinear<ElementType>::Clear() {
  thrust::fill(elements_.begin(), elements_.end(), ElementType{});
}

template <class ElementType>
__device__ const ElementType &
GridLinear<ElementType>::DevRef::operator()(int x, int y, int z) const {
  return ptr_elements_[RANGE_INDEX_XYZ(x, y, z, grid_range_)];
}

template <class ElementType>
__device__ ElementType &GridLinear<ElementType>::DevRef::operator()(int x,
                                                                    int y,
                                                                    int z) {
  return ptr_elements_[RANGE_INDEX_XYZ(x, y, z, grid_range_)];
}

template <class ElementType>
GridLinear<ElementType>::DevRef::DevRef(GridLinear<ElementType> &grid)
    : ptr_elements_(grid.elements_.data().get()),
      grid_range_(grid.grid_range_) {
}

template <class ElementType>
GridLinear<ElementType>::GridLinear(const glm::ivec3 &range)
    : elements_(range.x * range.y * range.z), grid_range_(range) {
}

template <class ElementType>
class GridLinearHost {
 public:
  GridLinearHost() = default;
  GridLinearHost(const glm::ivec3 &range);
  GridLinearHost(const GridLinear<ElementType> &grid_linear);
  ElementType &operator()(const glm::ivec3 &index);
  const ElementType &operator()(const glm::ivec3 &index) const;
  ElementType &operator()(int x, int y, int z);
  const ElementType &operator()(int x, int y, int z) const;
  [[nodiscard]] glm::ivec3 Range() const {
    return range_;
  }
  [[nodiscard]] size_t Size() const {
    return range_.x * range_.y * range_.z;
  }

 private:
  std::vector<ElementType> elements_;
  glm::ivec3 range_{};
};

template <class ElementType>
ElementType &GridLinearHost<ElementType>::operator()(const glm::ivec3 &index) {
  return operator()(index.x, index.y, index.z);
}

template <class ElementType>
const ElementType &GridLinearHost<ElementType>::operator()(
    const glm::ivec3 &index) const {
  return operator()(index.x, index.y, index.z);
}

template <class ElementType>
const ElementType &GridLinearHost<ElementType>::operator()(int x,
                                                           int y,
                                                           int z) const {
  return elements_[RANGE_INDEX_XYZ(x, y, z, range_)];
}

template <class ElementType>
ElementType &GridLinearHost<ElementType>::operator()(int x, int y, int z) {
  return elements_[RANGE_INDEX_XYZ(x, y, z, range_)];
}

template <class ElementType>
GridLinearHost<ElementType>::GridLinearHost(
    const GridLinear<ElementType> &grid_linear)
    : elements_(grid_linear.elements_.size()), range_(grid_linear.grid_range_) {
  thrust::copy(grid_linear.elements_.begin(), grid_linear.elements_.end(),
               elements_.begin());
}

template <class ElementType>
GridLinearHost<ElementType>::GridLinearHost(const glm::ivec3 &range)
    : elements_(range.x * range.y * range.z), range_(range) {
}
