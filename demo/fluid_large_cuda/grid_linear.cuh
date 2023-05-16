#pragma once
#include "util.cuh"
#include "thrust/device_vector.h"

template<class ElementType>
class GridLinear {
 public:
  GridLinear() = default;
  explicit GridLinear(const glm::ivec3 &range);
  struct DevRef {
    DevRef(const GridLinear<ElementType>& grid);
    __device__ ElementType& operator()(int x, int y, int z);
    __device__ const ElementType& operator()(int x, int y, int z) const;
   private:
    ElementType *ptr_elements_;
    glm::ivec3 grid_range_;
  };
 private:
  thrust::device_vector<ElementType> elements_;
  glm::ivec3 grid_range_;
};

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
GridLinear<ElementType>::DevRef::DevRef(const GridLinear<ElementType> &grid) {
  ptr_elements_ = grid.elements_.data().get();
  grid_range_ = grid.grid_range_;
}

template <class ElementType>
GridLinear<ElementType>::GridLinear(const glm::ivec3 &range): elements_(range.x * range.y * range.z), grid_range_(range) {
}
