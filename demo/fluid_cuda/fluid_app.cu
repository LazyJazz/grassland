#include "fluid_app.h"
#include "thrust/device_vector.h"
#include "curand.h"
#include "curand_kernel.h"
#include "glm/gtc/matrix_transform.hpp"
#include "grid.cuh"

#define BLOCK_SIZE 256
#define LAUNCH_SIZE(x) ((x + (BLOCK_SIZE - 1)) / BLOCK_SIZE), BLOCK_SIZE

__device__ glm::vec3 RandomOnSphere(curandState_t *state) {
  auto y = curand_uniform(state) * 2.0f - 1.0f;
  auto xz = sqrt(1 - y * y);
  auto theta = glm::pi<float>() * 2.0f * curand_uniform(state);
  return glm::vec3{sin(theta) * xz, y, cos(theta) * xz};
}

__device__ glm::vec3 RandomInSphere(curandState_t *state) {
  return RandomOnSphere(state) * pow(curand_uniform(state), 1.0f / 3.0f);
}

__global__ void InitRandomParticles(Particle *particles,int n_particles) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n_particles) {
    curandState_t state;
    curand_init(blockIdx.x, threadIdx.x, 0, &state);
    Particle particle;
    particle.position = RandomInSphere(&state) * 4.0f + glm::vec3{10.0f, 10.0f, 10.0f};
    particle.velocity = glm::vec3{0.0f};
    particles[id] = particle;
  }
}

void FluidApp::InitParticles() {
  const int n_particles = settings_.num_particle;
  thrust::device_vector<Particle> dev_particles(n_particles);
  InitRandomParticles<<<LAUNCH_SIZE(n_particles)>>>(dev_particles.data().get(), n_particles);
  particles_.resize(n_particles);
  thrust::copy(dev_particles.begin(), dev_particles.end(), particles_.begin());
}

__global__ void ApplyGravity(Particle *particles,int n_particles) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n_particles) {
    Particle particle = particles[id];
    particle.velocity += GRAVITY * DELTA_T;
    particles[id] = particle;
  }
}

__global__ void AdvectParticleKernel(Particle *particles,int n_particles) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n_particles) {
    Particle particle = particles[id];
    particle.position += particle.velocity * DELTA_T;
    particles[id] = particle;
  }
}

void FluidApp::UpdatePhysicalSystem() {
  const int n_particles = settings_.num_particle;
  thrust::device_vector<Particle> dev_particles = particles_;
  ApplyGravity<<<LAUNCH_SIZE(n_particles)>>>(dev_particles.data().get(), n_particles);
  AdvectParticleKernel<<<LAUNCH_SIZE(n_particles)>>>(dev_particles.data().get(), n_particles);
  thrust::copy(dev_particles.begin(), dev_particles.end(), particles_.begin());
}
