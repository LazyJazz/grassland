#include "cstdint"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "iostream"
#include "thrust/device_vector.h"

#define BlockSize 128
#define CALL_GRID(x) ((x) + BlockSize - 1) / BlockSize, BlockSize

// Hashing function
__device__ __host__ uint32_t WangHash(uint32_t seed) {
  seed = uint32_t(seed ^ uint32_t(61)) ^ uint32_t(seed >> uint32_t(16));
  seed *= uint32_t(9);
  seed = seed ^ (seed >> 4);
  seed *= uint32_t(0x27d4eb2d);
  seed = seed ^ (seed >> 15);
  return seed;
}

// Kernel function to set sequential memory
__global__ void SequentialMemorySet(float *data, int num_ele) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  data[id] = id;
}

// Kernel function to set random memory
__global__ void RandomMemorySet(float *data, int num_ele) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  id = WangHash(id);
  data[id & (num_ele - 1)] = id;
}

// Kernel function to add a scalar to each element in the data array
__global__ void VectorAddSet(float *data, int num_ele, float s) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  data[id] += s;
}

int main() {
  int tier = 30;
  std::cout << "Please input the tier of the memory size (2^tier): ";
  std::cin >> tier;
  int size = (1 << tier);

  // Initialize device memory
  thrust::device_vector<float> dev_mem(size);
  thrust::fill(dev_mem.begin(), dev_mem.end(), 1.0f);

  // Initialize cache trash
  thrust::device_vector<float> cache_trash(1 << 25);
  thrust::fill(cache_trash.begin(), cache_trash.end(), 2.0f);

  // Start CUDA event
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // Call kernel function
  VectorAddSet<<<CALL_GRID(size)>>>(dev_mem.data().get(), size, 1.0f);

  // Stop CUDA event and calculate elapsed time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  uint64_t mem_size = uint64_t(size) * sizeof(float);
  std::cout << "Memory size: ";
  if (mem_size >= (1ull << 30)) {
    std::cout << mem_size / (1ull << 30) << "GB" << std::endl;
  } else if (mem_size >= (1ull << 20)) {
    std::cout << mem_size / (1ull << 20) << "MB" << std::endl;
  } else if (mem_size >= (1ull << 10)) {
    std::cout << mem_size / (1ull << 10) << "KB" << std::endl;
  } else {
    std::cout << mem_size << "B" << std::endl;
  }
  std::cout << "Elapsed time: " << ms << "ms" << std::endl;
  std::cout << "Bandwidth: " << mem_size / ms / 1024 / 1024 << "GB/s"
            << std::endl;

  float sum = thrust::reduce(dev_mem.begin(), dev_mem.end());
  std::cout << "Sum: " << sum << std::endl;
  if (sum == size * 2.0f) {
    std::cout << "Result is correct!" << std::endl;
  } else {
    std::cout << "Result is wrong!" << std::endl;
  }
}
