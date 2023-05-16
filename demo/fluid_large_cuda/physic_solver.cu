#include "curand_kernel.h"
#include "device_clock.cuh"
#include "fluid_large.cuh"
#include "physic_solver.cuh"
#include "util.cuh"
#include "thrust/sort.h"

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
  cell_indices_.resize(physic_settings_.num_particle);
  block_indices_.resize(physic_settings_.num_particle);
  dev_instance_infos_.resize(physic_settings_.num_particle);
  InitParticleKernel<<<CALL_GRID(particles_.size())>>>(particles_.data().get(),
                                                       particles_.size());
  cell_range_ = SceneRange() / physic_settings_.delta_x;
  block_range_ = (cell_range_ + 7) / 8;
  printf("Cell [%d %d %d]\n", cell_range_.x, cell_range_.y, cell_range_.z);
  printf("Block [%d %d %d] * (8 8 8)\n", block_range_.x, block_range_.y,
         block_range_.z);

  level_sets_ = Grid<float>(cell_range_ + 1);
  u_grid_ = Grid<MACGridContent>(cell_range_ + glm::ivec3{1, 0, 0});
  v_grid_ = Grid<MACGridContent>(cell_range_ + glm::ivec3{0, 1, 0});
  w_grid_ = Grid<MACGridContent>(cell_range_ + glm::ivec3{0, 0, 1});
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
  thrust::copy(dev_instance_infos_.begin(), dev_instance_infos_.end(),
               instance_infos);
}

__global__ void ParticleLinearOperationsKernel(Particle *particles,
                                               int *cell_indices,
                                               int *block_indices,
                                               glm::vec3 gravity_delta_t,
                                               float delta_x,
                                               glm::ivec3 cell_range,
                                               glm::ivec3 block_range,
                                               size_t num_particle) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particle) {
    return;
  }
  auto particle = particles[i];

  glm::ivec3 cell_index_3 = particle.position / delta_x;
  glm::ivec3 block_index_3 = cell_index_3 >> 3;
  int cell_index = RANGE_INDEX(cell_index_3, cell_range);
  int block_index = RANGE_INDEX(block_index_3, block_range);
  if (!InSceneRange(particle.position)) {
    cell_index = -1;
    block_index = -1;
  }
  particle.velocity += gravity_delta_t;
//  if (i < 10) {
//    printf("index: %d %d %d %d %d %d\n", i, cell_index_3.x, cell_index_3.y, cell_index_3.z, cell_index, block_index);
//  }
  particles[i] = particle;
  cell_indices[i] = cell_index;
  block_indices[i] = block_index;
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

  ParticleLinearOperationsKernel<<<CALL_GRID(particles_.size())>>>(
      particles_.data().get(), cell_indices_.data().get(),
      block_indices_.data().get(),
      physic_settings_.gravity * physic_settings_.delta_t,
      physic_settings_.delta_x, cell_range_, block_range_, particles_.size());
  dev_clock.Record("Particle Linear Operations Kernel");

//  thrust::sort_by_key(cell_indices_.begin(), cell_indices_.end(), particles_.begin());
//  dev_clock.Record("Sort by Key");



  AdvectionKernel<<<CALL_GRID(particles_.size())>>>(
      particles_.data().get(), physic_settings_.delta_t, particles_.size());
  dev_clock.Record("Advection");

  InstanceInfoComposeKernel<<<CALL_GRID(dev_instance_infos_.size())>>>(
      particles_.data().get(), dev_instance_infos_.data().get(),
      particles_.size());
  dev_clock.Record("Compose Instance Info");


  dev_clock.Finish();

//  Particle host_particles[10];
//  int host_cell_indices[10];
//  int host_block_indices[10];
//  thrust::copy(cell_indices.begin(), cell_indices.begin() + 10, host_cell_indices);
//  thrust::copy(block_indices.begin(), block_indices.begin() + 10, host_block_indices);
//  thrust::copy(particles_.begin(), particles_.begin() + 10, host_particles);
//  for (int i = 0; i < 10; i++) {
//    glm::ivec3 cell_index = host_particles[i].position / physic_settings_.delta_x;
//    printf("%d %d %d %d %d\n", cell_index.x, cell_index.y, cell_index.z, host_cell_indices[i], host_block_indices[i]);
//  }

  // OutputXYZFile();
}

void PhysicSolver::OutputXYZFile() {
  std::vector<Particle> particles(particles_.size());
  thrust::copy(particles_.begin(), particles_.end(), particles.begin());
  static int round = 0;
  if (!round)
    std::system("mkdir data");
  std::ofstream file("data/" + std::to_string(round) + ".xyz",
                     std::ios::binary);
  int cnt = 0;
  for (auto &particle : particles) {
    if (particle.type == 0) {
      cnt++;
      if (cnt == 10) {
        file.write(reinterpret_cast<const char *>(&particle.position),
                   sizeof(particle.position));
        cnt = 0;
      }
    }
  }
  file.close();
  round++;
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

__device__ __host__ bool InSceneRange(const glm::vec3 &position) {
  if (position.x < 0.0f)
    return false;
  if (position.y < 0.0f)
    return false;
  if (position.z < 0.0f)
    return false;
  auto range = SceneRange();
  if (position.x >= range.x)
    return false;
  if (position.y >= range.y)
    return false;
  if (position.z >= range.z)
    return false;
  return true;
}
