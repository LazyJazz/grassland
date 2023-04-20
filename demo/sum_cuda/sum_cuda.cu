#include "cstdint"
#include "sum_cuda.h"

__global__ void CurandFloatKernel(float *data, uint64_t length) {
  uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < length) {
    curandState state;
    curand_init(blockIdx.x, threadIdx.x, 0, &state);
    data[id] = curand_uniform(&state);
  }
}
void CurandFloat(float *data, uint64_t length) {
  CurandFloatKernel<<<(length + 255ull) / 256ull, 256ull>>>(data, length);
}
