#pragma once
#include "thrust/device_vector.h"
#include "thrust/reduce.h"
#include "thrust/transform.h"

template <typename T>
struct SquareOp {
  __host__ __device__ T operator()(const T &x) const {
    return x * x;
  }
};

struct SaxpyFunctor {
  const float a;

  SaxpyFunctor(float _a) : a(_a) {
  }

  __host__ __device__ float operator()(const float &x, const float &y) const {
    return a * x + y;
  }
};

template <class MatrixType>
void ConjugateGradient(MatrixType matrix,
                       const thrust::device_vector<float> &b,
                       thrust::device_vector<float> &x) {
  thrust::device_vector<float> Ap_vec_(x.size());
  thrust::device_vector<float> buffer_(x.size());
  thrust::device_vector<float> r_vec_(x.size());
  thrust::device_vector<float> p_vec_(x.size());
  matrix(x, Ap_vec_);
  SquareOp<float> square_op;
  thrust::plus<float> plus_op;
  thrust::minus<float> minus_op;
  thrust::multiplies<float> multi_op;
  auto dot = [&](const thrust::device_vector<float> &a,
                 const thrust::device_vector<float> &b) {
    thrust::transform(a.begin(), a.end(), b.begin(), buffer_.begin(), multi_op);
    return thrust::reduce(buffer_.begin(), buffer_.end(), 0.0f, plus_op);
  };
  auto saxpy = [](float a, const thrust::device_vector<float> &x,
                  const thrust::device_vector<float> &y,
                  thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                      SaxpyFunctor(a));
  };
  auto add = [plus_op](const thrust::device_vector<float> &x,
                       const thrust::device_vector<float> &y,
                       thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), plus_op);
  };
  auto sub = [minus_op](const thrust::device_vector<float> &x,
                        const thrust::device_vector<float> &y,
                        thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), minus_op);
  };
  auto assign = [](const thrust::device_vector<float> &x,
                   thrust::device_vector<float> &y) {
    thrust::copy(x.begin(), x.end(), y.begin());
  };
  matrix(x, buffer_);
  sub(b, buffer_, r_vec_);
  assign(r_vec_, p_vec_);
  int cnt = 2;
  while (true) {
    float rk2 = dot(r_vec_, r_vec_);
    matrix(p_vec_, Ap_vec_);
    cnt++;
    float ak = rk2 / dot(p_vec_, Ap_vec_);
    saxpy(ak, p_vec_, x, x);
    saxpy(-ak, Ap_vec_, r_vec_, r_vec_);
    float new_rk2 = dot(r_vec_, r_vec_);
    // printf("%f\n", new_rk2);
    if (new_rk2 < r_vec_.size() * 1e-8f) {
      printf("%d\n", cnt);
      break;
    }
    float bk = new_rk2 / rk2;
    saxpy(bk, p_vec_, r_vec_, p_vec_);
  }
}
