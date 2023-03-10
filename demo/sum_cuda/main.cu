#include "sum_cuda.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/reduce.h"
#include "vector"
#include "chrono"

struct Square {
  __host__ __device__ float operator() (const float &x) const {
    return x * x;
  }
};

int main() {
  const uint64_t sum_size = 2000000;
  std::vector<float> numbers(sum_size, 0);
  thrust::device_vector<float> device_numbers(sum_size);

  auto start_time = std::chrono::steady_clock::now();
  CurandFloat(device_numbers.data().get(), uint64_t(sum_size));
  thrust::transform(device_numbers.begin(), device_numbers.end(), device_numbers.begin(), Square());
  std::cout << (std::chrono::steady_clock::now() - start_time) / std::chrono::milliseconds(1) << std::endl;

  start_time = std::chrono::steady_clock::now();
  thrust::copy(device_numbers.begin(), device_numbers.end(), numbers.begin());
  std::cout << (std::chrono::steady_clock::now() - start_time) / std::chrono::milliseconds(1) << std::endl;

  start_time = std::chrono::steady_clock::now();
  float sum = 0;
  const uint64_t block_size = 10000;

  for (uint64_t i = 0; i < sum_size; i += block_size) {
    float block_sum = 0;
      for (uint64_t j = 0; j < block_size && i + j < sum_size; j++) {
        block_sum += numbers[i + j];
      }
      sum += block_sum;
  }

  std::cout << (std::chrono::steady_clock::now() - start_time) / std::chrono::microseconds(1) << std::endl;
  start_time = std::chrono::steady_clock::now();

  auto reduce_result = thrust::reduce(device_numbers.begin(), device_numbers.end());

  std::cout << (std::chrono::steady_clock::now() - start_time) / std::chrono::microseconds(1) << std::endl;
  std::cout << reduce_result << std::endl;
  std::cout << sum << std::endl;
}
