#pragma once
#include "cuda_runtime.h"
#include "fstream"
#include "iomanip"
#include "thrust/device_vector.h"

template <class Ty>
struct GridDev;

template <class Ty>
class Grid {
 public:
  Grid() = default;
  Grid(int size_x, int size_y, int size_z) {
    buffer_.resize(size_x * size_y * size_z);
    size_x_ = size_x;
    size_y_ = size_y;
    size_z_ = size_z;
  }
  thrust::device_reference<Ty> &operator()(int idx, int idy, int idz);
  const thrust::device_reference<Ty> &operator()(int idx,
                                                 int idy,
                                                 int idz) const;
  void ClearData();
  void PlotCSV(int x);
  template <class RetTy>
  void PlotCSV(int x,
               std::function<RetTy(const thrust::device_reference<Ty> &)>
                   converter) const;
  int Size() const;

 private:
  thrust::device_vector<Ty> buffer_;
  int size_x_{};
  int size_y_{};
  int size_z_{};
  friend GridDev<Ty>;
};

template <class Ty>
int Grid<Ty>::Size() const {
  return size_x_ * size_y_ * size_z_;
}

template <class Ty>
void Grid<Ty>::ClearData() {
  thrust::fill(buffer_.begin(), buffer_.end(), Ty());
}

template <class Ty>
void Grid<Ty>::PlotCSV(int x) {
  std::ofstream file("grid.csv");
  file << std::fixed << std::setprecision(7);
  for (int y = 0; y < size_y_; y++) {
    for (int z = 0; z < size_z_; z++) {
      file << (*this)(x, y, z) << ",";
    }
    file << std::endl;
  }
  file.close();
  std::system("start grid.csv");
  std::system("pause");
}

template <class Ty>
const thrust::device_reference<Ty> &Grid<Ty>::operator()(int idx,
                                                         int idy,
                                                         int idz) const {
  return buffer_[idx * size_y_ * size_z_ + idy * size_z_ + idz];
}

template <class Ty>
thrust::device_reference<Ty> &Grid<Ty>::operator()(int idx, int idy, int idz) {
  return buffer_[idx * size_y_ * size_z_ + idy * size_z_ + idz];
}
