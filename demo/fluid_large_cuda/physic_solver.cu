#include "curand_kernel.h"
#include "device_clock.cuh"
#include "fluid_large.cuh"
#include "physic_solver.cuh"
#include "thrust/sort.h"
#include "util.cuh"

template<class VectorType>
__device__ float KernelFunction(const VectorType& v) {
  auto len = glm::length(v);
  if (len < 0.5f) {
    return 0.75f - len * len;
  } else if (len < 1.5f) {
    return 0.5f * (1.5f - len) * (1.5f - len);
  }
  return 0.0f;
}

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
  block_range_ = (cell_range_ + BLOCK_BIT_MASK_V3) / BLOCK_DIM_SIZE_V3;
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
  glm::ivec3 block_index_3 = cell_index_3 >> BLOCK_BIT_COUNT_V3;
  int cell_index = RANGE_INDEX(cell_index_3, cell_range);
  int block_index = RANGE_INDEX(block_index_3, block_range);
  if (!InSceneRange(particle.position)) {
    cell_index = -1;
    block_index = -1;
  }
  particle.velocity += gravity_delta_t;
  //  if (i < 10) {
  //    printf("index: %d %d %d %d %d %d\n", i, cell_index_3.x, cell_index_3.y,
  //    cell_index_3.z, cell_index, block_index);
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

template<class ParticleType, class GridEleType, class OpType>
__device__ void NeighbourGridInteract(const ParticleType& particle, const glm::vec3& grid_pos, typename Grid<GridEleType>::DevRef& grid, OpType op) {
  glm::ivec3 nearest_index{
      floor(grid_pos.x),
      floor(grid_pos.y),
      floor(grid_pos.z)
  };

  auto cell_range = grid.Range();
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        auto current_index = nearest_index + glm::ivec3{dx, dy, dz};
        if (current_index.x < 0 || current_index.y < 0 || current_index.z < 0)
          continue;
        if (current_index.x >= cell_range.x || current_index.y >= cell_range.y  || current_index.z >= cell_range.z)
          continue ;
        auto weight = KernelFunction(grid_pos - glm::vec3{current_index} - 0.5f);
        if (weight > 1e-6f) {
          op(particle, grid(current_index.x, current_index.y, current_index.z), weight);
        }
      }
    }
  }
}

__device__ void AssignVelocity(thrust::pair<float, int> v_t, MACGridContent& content, float weight) {
  atomicAdd(&content.w[v_t.second], weight);
  atomicAdd(&content.v[v_t.second], v_t.first * weight);
}

__global__ void Particle2GridTransferKernel(
    const Particle *particles,
    int num_particle,
    Grid<float>::DevRef level_sets,
    Grid<MACGridContent>::DevRef u_grid,
    Grid<MACGridContent>::DevRef v_grid,
    Grid<MACGridContent>::DevRef w_grid,
    float delta_x,
    glm::ivec3 cell_range) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= num_particle) return;
  auto particle = particles[id];
  if (!InSceneRange(particle.position)) return;
  auto grid_pos = particle.position / delta_x;
  NeighbourGridInteract<thrust::pair<float, int>, MACGridContent>(thrust::make_pair(particle.velocity.x, particle.type), grid_pos + glm::vec3{0.5f, 0.0f, 0.0f}, u_grid, AssignVelocity);
  NeighbourGridInteract<thrust::pair<float, int>, MACGridContent>(thrust::make_pair(particle.velocity.y, particle.type), grid_pos + glm::vec3{0.0f, 0.5f, 0.0f}, v_grid, AssignVelocity);
  NeighbourGridInteract<thrust::pair<float, int>, MACGridContent>(thrust::make_pair(particle.velocity.z, particle.type), grid_pos + glm::vec3{0.0f, 0.0f, 0.5f}, w_grid, AssignVelocity);
}

__global__ void ProcessMACGridKernel(Grid<MACGridContent>::DevRef grid) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  auto cell_range = grid.Range();
  int num_cell = cell_range.x * cell_range.y * cell_range.z;
  if (id >= num_cell) return;
  glm::ivec3 cell_index{id / (cell_range.y * cell_range.z), (id / cell_range.z) % cell_range.y, id % cell_range.z};
  auto cell_content = grid(cell_index);
  if (cell_content.w[0] > 1e-6f) {
    cell_content.v[0] /= cell_content.w[0];
  } else {
    cell_content.v[0] = 0.0f;
  }
  if (cell_content.w[1] > 1e-6f) {
    cell_content.v[1] /= cell_content.w[1];
  } else {
    cell_content.v[1] = 0.0f;
  }
  grid(cell_index) = cell_content;
}

void PhysicSolver::UpdateStep() {
  DeviceClock dev_clock;

  level_sets_.Clear();
  u_grid_.Clear();
  v_grid_.Clear();
  w_grid_.Clear();
  dev_clock.Record("Clean data");

  ParticleLinearOperationsKernel<<<CALL_GRID(particles_.size())>>>(
      particles_.data().get(), cell_indices_.data().get(),
      block_indices_.data().get(),
      physic_settings_.gravity * physic_settings_.delta_t,
      physic_settings_.delta_x, cell_range_, block_range_, particles_.size());
  dev_clock.Record("Particle Linear Operations Kernel");

    thrust::stable_sort_by_key(cell_indices_.begin(), cell_indices_.end(),
                      particles_.begin());
    dev_clock.Record("Sort by Cells");

//      thrust::sort_by_key(block_indices_.begin(), block_indices_.end(),
//                          particles_.begin());
//      dev_clock.Record("Sort by Blocks");

  Particle2GridTransferKernel<<<CALL_GRID(particles_.size())>>>(particles_.data().get(), particles_.size(), level_sets_, u_grid_, v_grid_, w_grid_, physic_settings_.delta_x, cell_range_);
  dev_clock.Record("Trivial Grid2Particle");

    ProcessMACGridKernel<<<CALL_GRID(u_grid_.Size())>>>(u_grid_);
    ProcessMACGridKernel<<<CALL_GRID(v_grid_.Size())>>>(v_grid_);
    ProcessMACGridKernel<<<CALL_GRID(w_grid_.Size())>>>(w_grid_);
  dev_clock.Record("Process MAC Grid");

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
  //  thrust::copy(cell_indices.begin(), cell_indices.begin() + 10,
  //  host_cell_indices); thrust::copy(block_indices.begin(),
  //  block_indices.begin() + 10, host_block_indices);
  //  thrust::copy(particles_.begin(), particles_.begin() + 10, host_particles);
  //  for (int i = 0; i < 10; i++) {
  //    glm::ivec3 cell_index = host_particles[i].position /
  //    physic_settings_.delta_x; printf("%d %d %d %d %d\n", cell_index.x,
  //    cell_index.y, cell_index.z, host_cell_indices[i],
  //    host_block_indices[i]);
  //  }

  // OutputXYZFile();

    // GridLinearHost<MACGridContent> u_grid_host(u_grid_);
    GridLinearHost<MACGridContent> v_grid_host(v_grid_);
    // GridLinearHost<MACGridContent> w_grid_host(w_grid_);
    std::ofstream csv_file("grid.csv");
    for (int y = 0; y < v_grid_host.Range().y; y++) {
      for (int z = 0; z < v_grid_host.Range().z; z++) {
        csv_file << v_grid_host(v_grid_host.Range().x / 2, y, z).v[0] << ",";
      }
      csv_file << std::endl;
    }
    csv_file.close();
    std::system("start grid.csv");
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
