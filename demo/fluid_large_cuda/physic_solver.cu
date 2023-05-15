#include "curand_kernel.h"
#include "device_clock.cuh"
#include "fluid_large.cuh"
#include "physic_solver.cuh"
#include "util.cuh"

__global__ void InitParticleKernel(Particle *particles, int num_particle) {
  uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= num_particle)
    return;
  curandState rd;
  curand_init(blockIdx.x, threadIdx.x, 0, &rd);
  Particle particle{};
  particle.velocity = glm::vec3{};
  do {
    particle.position.x = curand_uniform(&rd);
    particle.position.y = curand_uniform(&rd);
    particle.position.z = curand_uniform(&rd);
    particle.position *= SceneRange();
  } while (!InsideFreeSpace(particle.position) ||
           (particle.type = AssignType(particle.position)) == -1);
  particles[id] = particle;
}

PhysicSolver::PhysicSolver(const PhysicSettings &physic_settings)
    : physic_settings_(physic_settings) {
  particles_.resize(physic_settings_.num_particle);
  InitParticleKernel<<<CALL_GRID(particles_.size())>>>(particles_.data().get(),
                                                       particles_.size());
}

__global__ void InstanceInfoComposeKernel(const Particle *particles,
                                          InstanceInfo *instance_infos,
                                          int num_particle) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particle)
    return;
  auto particle = particles[i];
  InstanceInfo info{};
  info.size = 1e-4f;
  info.offset = particle.position;
  info.color = particle.type ? glm::vec4{1.0f, 0.5f, 0.5f, 1.0f}
                             : glm::vec4{0.5f, 0.5f, 1.0f, 1.0f};
  instance_infos[i] = info;
}

void PhysicSolver::GetInstanceInfoArray(InstanceInfo *instance_infos) const {
  thrust::device_vector<InstanceInfo> dev_instance_infos(particles_.size());
  InstanceInfoComposeKernel<<<CALL_GRID(dev_instance_infos.size())>>>(
      particles_.data().get(), dev_instance_infos.data().get(),
      particles_.size());
  thrust::copy(dev_instance_infos.begin(), dev_instance_infos.end(),
               instance_infos);
}

__global__ void ApplyExternalForcesKernel(Particle *particles,
                                          glm::vec3 gravity_delta_t,
                                          int num_particle) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particle)
    return;
  auto particle = particles[i];
  particle.velocity += gravity_delta_t;
  particles[i] = particle;
}

__global__ void AdvectionKernel(Particle *particles,
                                float delta_t,
                                int num_particle) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particle)
    return;
  auto particle = particles[i];
  particle.position += delta_t * particle.velocity;
  particles[i] = particle;
}

void PhysicSolver::UpdateStep() {
  DeviceClock dev_clock;
  ApplyExternalForcesKernel<<<CALL_GRID(particles_.size())>>>(
      particles_.data().get(),
      physic_settings_.gravity * physic_settings_.delta_t, particles_.size());
  dev_clock.Record("Apply External Force");
  AdvectionKernel<<<CALL_GRID(particles_.size())>>>(
      particles_.data().get(), physic_settings_.delta_t, particles_.size());
  dev_clock.Record("Advection");
  dev_clock.Finish();
}

__device__ __host__ bool InsideFreeSpace(const glm::vec3 &position) {
  glm::vec3 range = SceneRange();
  return (glm::length(position - glm::vec3{range.x * 0.5f, range.y * 0.3f,
                                           range.z * 0.5f}) < range.x * 0.4f) ||
         (glm::length(position - glm::vec3{range.x * 0.5f, range.y * 0.7f,
                                           range.z * 0.5f}) < range.x * 0.4f) ||
         ((position.y > range.y * 0.3f && position.y < range.y * 0.7f) &&
          glm::length(glm::vec2{position.x, position.z} -
                      glm::vec2{range.x * 0.5f, range.z * 0.5f}) <
              range.x * 0.2f);
  ;
}

__device__ __host__ int AssignType(const glm::vec3 &position) {
  if (position.y < SceneRange().y * 0.5f) {
    return 0;
  } else {
    return 1;
  }
}

__device__ __host__ glm::vec3 SceneRange() {
  return {1.0f, 2.0f, 1.0f};
}
