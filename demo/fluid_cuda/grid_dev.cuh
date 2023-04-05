#include "grid.h"

template <class Ty>
struct GridDev {
 public:
  GridDev() = default;
  GridDev(Grid<Ty>& host_grid);
  __device__ Ty& operator()(int idx, int idy, int idz);
  __device__ const Ty& operator()(int idx, int idy, int idz) const;
  __device__ Ty& operator()(glm::ivec3 index);
  __device__ const Ty& operator()(glm::ivec3 index) const;
  Ty* buffer_{nullptr};
  int size_x_{};
  int size_y_{};
  int size_z_{};
};

template <class Ty>
__device__ const Ty &GridDev<Ty>::operator()(glm::ivec3 index) const {
  return operator()(index.x, index.y, index.z);
}

template <class Ty>
__device__ Ty &GridDev<Ty>::operator()(glm::ivec3 index) {
  return operator()(index.x, index.y, index.z);
}

template <class Ty>
__device__ const Ty &GridDev<Ty>::operator()(int idx, int idy, int idz) const {
  return buffer_[idx * size_y_ * size_z_ + idy * size_z_ + idz];
}

template <class Ty>
__device__ Ty &GridDev<Ty>::operator()(int idx, int idy, int idz) {
  return buffer_[idx * size_y_ * size_z_ + idy * size_z_ + idz];
}

template <class Ty>
GridDev<Ty>::GridDev(Grid<Ty> &host_grid) {
  size_x_ = host_grid.size_x_;
  size_y_ = host_grid.size_y_;
  size_z_ = host_grid.size_z_;
  buffer_ = host_grid.buffer_.data().get();
}
