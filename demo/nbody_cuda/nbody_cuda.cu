#include "nbody_cuda.h"

#define BLOCK_SIZE 256
#define GRID_SIZE ((NUM_PARTICLE + BLOCK_SIZE - 1) / BLOCK_SIZE)

__global__ void
UpdateKernel(const glm::vec4 *positions, glm::vec4 *positions_write, glm::vec4 *velocities, int n_particle) {
    uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= n_particle) {
        return;
    }
    auto pos = positions[id];
    auto vel = velocities[id];
    glm::vec4 accel{0.0f};
    extern __shared__ glm::vec4 shared_pos[];
    for (int j = 0; j < n_particle; j += blockDim.x) {
        if (j + threadIdx.x < n_particle) {
            shared_pos[threadIdx.x] = positions[threadIdx.x  + j];
        }
        __syncthreads();
        for (int i = 0; i < blockDim.x && i + j < n_particle; i++) {
            auto pos_j = shared_pos[i];
            auto diff = pos - pos_j;
            auto lsqr = glm::dot(diff, diff);
            auto l = rsqrt(lsqr);
            if (lsqr < DELTA_T * DELTA_T) {
                continue;
            }
            diff *= l * l * l;
            accel += -diff * DELTA_T * GRAVITY_COE;
        }
        __syncthreads();
    }
    vel += accel;
    pos += vel * DELTA_T;
    positions_write[id] = pos;
    velocities[id] = vel;
}

void UpdateStep(glm::vec4 *positions, glm::vec4 *velocities, int n_particles) {
    glm::vec4 *dev_positions;
    glm::vec4 *dev_velocities;
    glm::vec4 *dev_positions_write;
    cudaMalloc(&dev_positions, n_particles * sizeof(glm::vec4));
    cudaMalloc(&dev_velocities, n_particles * sizeof(glm::vec4));
    cudaMalloc(&dev_positions_write, n_particles * sizeof(glm::vec4));
    cudaMemcpy(dev_positions, positions, sizeof(glm::vec4) * n_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_velocities, velocities, sizeof(glm::vec4) * n_particles, cudaMemcpyHostToDevice);

    UpdateKernel<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(glm::vec4)>>>(
            dev_positions, dev_positions_write, dev_velocities, n_particles);
    cudaDeviceSynchronize();

    cudaMemcpy(positions, dev_positions_write, sizeof(glm::vec4) * n_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, dev_velocities, sizeof(glm::vec4) * n_particles, cudaMemcpyDeviceToHost);
    cudaFree(dev_velocities);
    cudaFree(dev_positions);
    cudaFree(dev_positions_write);

}
