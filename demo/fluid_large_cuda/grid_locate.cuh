#pragma once
#include "util.cuh"
#include "thrust/device_vector.h"

template<class ElementType>
class GridLocate {
 public:
  GridLocate() = default;
  struct Block {
    ElementType elements[8][8][8];
  };

  struct DevRef {
    DevRef(const GridLocate<ElementType>& grid);
    __device__ ElementType& operator() (int x, int y, int z);
    __device__ const ElementType& operator() (int x, int y, int z) const;
   private:
    Block *blocks_;
    glm::ivec3 cell_range_;
    glm::ivec3 block_range_{};
  };

  GridLocate(const glm::ivec3 &range);
 private:
  thrust::device_vector<Block> blocks_;
  glm::ivec3 cell_range_{};
  glm::ivec3 block_range_{};
};

template <class ElementType>
__device__ ElementType &GridLocate<ElementType>::DevRef::operator()(int x,
                                                                    int y,
                                                                    int z) {
  return blocks_[RANGE_INDEX_XYZ(x >> 3, y >> 3, z >> 3, block_range_)].elements[x&7][y&7][z&7];
}
template <class ElementType>
__device__ const ElementType &
GridLocate<ElementType>::DevRef::operator()(int x, int y, int z) const {
  return blocks_[RANGE_INDEX_XYZ(x >> 3, y >> 3, z >> 3, block_range_)].elements[x&7][y&7][z&7];
}

template <class ElementType>
GridLocate<ElementType>::DevRef::DevRef(const GridLocate<ElementType> &grid): blocks_(grid.blocks_.data().get()), cell_range_(grid.cell_range_), block_range_(grid.block_range_) {
}

template <class ElementType>
GridLocate<ElementType>::GridLocate(const glm::ivec3 &range) {
  cell_range_ = range;
  block_range_ = (range + 7) / 8;
  blocks_.resize(block_range_.x * block_range_.y * block_range_.z);
}
