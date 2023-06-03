#pragma once
#include "util.cuh"
#include "thrust/device_vector.h"

template<class ElementType>
class GridLocate {
 public:
  GridLocate() = default;
  struct Block {
    ElementType elements[BLOCK_DIM_SIZE_X][BLOCK_DIM_SIZE_Y][BLOCK_DIM_SIZE_Z];
  };

  struct DevRef {
    DevRef(GridLocate<ElementType>& grid);
    __device__ ElementType& operator() (int x, int y, int z);
    __device__ const ElementType& operator() (int x, int y, int z) const;
    [[nodiscard]] __device__ glm::ivec3 Range() const {return cell_range_;}
    [[nodiscard]] __device__ glm::ivec3 BlockRange() const {return block_range_;}
   private:
    Block *blocks_;
    glm::ivec3 cell_range_;
    glm::ivec3 block_range_{};
  };

  GridLocate(const glm::ivec3 &range);
  GridLocate(const GridLocate<ElementType>&) = delete;
  GridLocate(GridLocate<ElementType>&& grid) {
    blocks_ = grid.blocks_;
    cell_range_ = grid.cell_range_;
    block_range_ = grid.block_range_;
    grid.blocks_ = nullptr;
  }
  GridLocate<ElementType>& operator= (GridLocate<ElementType>&& grid) {
    if (blocks_) {
      cudaFree(blocks_);
    }
    blocks_ = grid.blocks_;
    cell_range_ = grid.cell_range_;
    block_range_ = grid.block_range_;
    grid.blocks_ = nullptr;
    return *this;
  }
  ~GridLocate();
  void Clear();
 private:
  Block *blocks_{nullptr};
  glm::ivec3 cell_range_{};
  glm::ivec3 block_range_{};
};

template <class ElementType>
GridLocate<ElementType>::GridLocate(const glm::ivec3 &range) {
  cell_range_ = range;
  block_range_ = (range + BLOCK_BIT_MASK_V3) / BLOCK_DIM_SIZE_V3;
  cudaMalloc(&blocks_, block_range_.x * block_range_.y * block_range_.z * sizeof(Block));
}


template <class ElementType>
GridLocate<ElementType>::~GridLocate() {
  if (blocks_) {
    cudaFree(blocks_);
  }
}

template <class ElementType>
void GridLocate<ElementType>::Clear() {
  // thrust::fill(blocks_, blocks_ + block_range_.x * block_range_.y * block_range_.z, Block{});
  cudaMemset(blocks_, 0, sizeof(Block) * block_range_.x * block_range_.y * block_range_.z);
}

template <class ElementType>
__device__ ElementType &GridLocate<ElementType>::DevRef::operator()(int x,
                                                                    int y,
                                                                    int z) {
  return blocks_[RANGE_INDEX_XYZ(x >> BLOCK_BIT_COUNT_X, y >> BLOCK_BIT_COUNT_Y, z >> BLOCK_BIT_COUNT_Z, block_range_)].elements[x&BLOCK_BIT_MASK_X][y&BLOCK_BIT_MASK_Y][z&BLOCK_BIT_MASK_Z];
}
template <class ElementType>
__device__ const ElementType &
GridLocate<ElementType>::DevRef::operator()(int x, int y, int z) const {
  return blocks_[RANGE_INDEX_XYZ(x >> BLOCK_BIT_COUNT_X, y >> BLOCK_BIT_COUNT_Y, z >> BLOCK_BIT_COUNT_Z, block_range_)].elements[x&BLOCK_BIT_MASK_X][y&BLOCK_BIT_MASK_Y][z&BLOCK_BIT_MASK_Z];
}

template <class ElementType>
GridLocate<ElementType>::DevRef::DevRef(GridLocate<ElementType> &grid): blocks_(grid.blocks_), cell_range_(grid.cell_range_), block_range_(grid.block_range_) {
}
