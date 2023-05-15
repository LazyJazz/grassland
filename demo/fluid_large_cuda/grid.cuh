#pragma once
#include "glm/glm.hpp"
#include "thrust/device_vector.h"

#define BLOCK_BIT 3
#define BLOCK_SIZE (1 << BLOCK_BIT)

template<class EleType>
struct GridBlock {
  EleType block[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];
};

template<class EleType>
class GridDevRef;

template<class EleType>
class Grid {
 public:
  Grid(glm::ivec3 grid_size);
 private:
  friend GridDevRef<EleType>;
  thrust::device_vector<GridBlock<EleType>> grid_blocks_;
  glm::ivec3 block_range_;
};

template<class EleType>
class GridDevRef {
 public:
  GridDevRef(const Grid<EleType>& grid);
  EleType& operator() (size_t x, size_t y, size_t z);
  const EleType& operator() (size_t x, size_t y, size_t z) const;
 private:
  GridBlock<EleType>* grid_blocks_;
  glm::ivec3 block_range_;
};

template <class EleType>
GridDevRef<EleType>::GridDevRef(const Grid<EleType> &grid) {
  grid_blocks_ = grid.grid_blocks_.data().get();
  block_range_ = grid.block_range_;
}

template <class EleType>
const EleType &GridDevRef<EleType>::operator()(size_t x,
                                               size_t y,
                                               size_t z) const {
  return grid_blocks_[block_range_.x * (x >> BLOCK_BIT) + block_range_.y * (y >> BLOCK_BIT) + block_range_.z * (z >> BLOCK_BIT)].block[x & (BLOCK_SIZE - 1)][y & (BLOCK_SIZE - 1)][z & (BLOCK_SIZE - 1)];
}

template <class EleType>
EleType &GridDevRef<EleType>::operator()(size_t x, size_t y, size_t z) {
  return grid_blocks_[block_range_.x * (x >> BLOCK_BIT) + block_range_.y * (y >> BLOCK_BIT) + block_range_.z * (z >> BLOCK_BIT)].block[x & (BLOCK_SIZE - 1)][y & (BLOCK_SIZE - 1)][z & (BLOCK_SIZE - 1)];
}
