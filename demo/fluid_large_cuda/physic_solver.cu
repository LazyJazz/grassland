
#include "device_clock.cuh"
#include "fluid_large.cuh"
#include "physic_solver.cuh"
#include "thrust/sort.h"
#include "util.cuh"

template <class VectorType>
__device__ float KernelFunction(const VectorType &v) {
  auto len = glm::length(v);
  if (len < 0.5f) {
    return 0.75f - len * len;
  } else if (len < 1.5f) {
    return 0.5f * (1.5f - len) * (1.5f - len);
  }
  return 0.0f;
}

template <class VectorType>
__device__ glm::vec3 KernelFunctionGradient(const VectorType &v) {
  auto len = glm::length(v);
  if (len < 1e-6f) {
    return glm::vec3{0.0f};
  } else if (len < 0.5f) {
    return -2.0f * len * glm::normalize(v);
  } else if (len < 1.5f) {
    return -(1.5f - len) * glm::normalize(v);
  }
  return glm::vec3{0.0f};
}

__global__ void InitParticleKernel(Particle *particles,
                                   curandState_t *rand_states,
                                   int num_particle) {
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
  rand_states[id] = rd;
}

PhysicSolver::PhysicSolver(const PhysicSettings &physic_settings)
    : physic_settings_(physic_settings) {
  particles_.resize(physic_settings_.num_particle);
  cell_indices_.resize(physic_settings_.num_particle);
  dev_instance_infos_.resize(physic_settings_.num_particle);
  dev_rand_states_.resize(physic_settings_.num_particle);
  InitParticleKernel<<<CALL_GRID(particles_.size())>>>(
      particles_.data().get(), dev_rand_states_.data().get(),
      particles_.size());
  cell_range_ = SceneRange() / physic_settings_.delta_x;
  block_range_ = (cell_range_ + BLOCK_BIT_MASK_V3) / BLOCK_DIM_SIZE_V3;
  printf("Cell [%d %d %d]\n", cell_range_.x, cell_range_.y, cell_range_.z);
  printf("Block [%d %d %d] * (8 8 8)\n", block_range_.x, block_range_.y,
         block_range_.z);

  cell_index_lower_bound_.resize(Grid<int>::BufferSize(cell_range_) + 1);

  level_set_ = Grid<LevelSet_t>(cell_range_ + 1);
  level_set_gradient_ = Grid<LevelSetGradient_t>(cell_range_ + 1);
  u_grid_ = Grid<MACGridContent>(cell_range_ + glm::ivec3{1, 0, 0});
  v_grid_ = Grid<MACGridContent>(cell_range_ + glm::ivec3{0, 1, 0});
  w_grid_ = Grid<MACGridContent>(cell_range_ + glm::ivec3{0, 0, 1});

  divergence_ = Grid<float>(cell_range_);
  pressure_ = Grid<float>(cell_range_);
  cell_coe_ = Grid<CellCoe>(cell_range_);
}

__global__ void InstanceInfoComposeKernel(
    const Particle *particles,
    InstanceInfo *instance_infos,
    int num_particle,
    Grid<LevelSetGradient_t>::DevRef phi_gradient,
    float delta_x);

void PhysicSolver::GetInstanceInfoArray(InstanceInfo *instance_infos) const {
  thrust::copy(dev_instance_infos_.begin(), dev_instance_infos_.end(),
               instance_infos);
}

__global__ void ParticleLinearOperationsKernel(Particle *particles,
                                               int *cell_indices,
                                               glm::vec3 gravity_delta_t,
                                               float delta_x,
                                               glm::ivec3 cell_range,
                                               int num_particle) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particle) {
    return;
  }
  auto particle = particles[i];

  glm::ivec3 cell_index_3 = particle.position / delta_x;
  int cell_index = RANGE_INDEX(cell_index_3, cell_range);
  if (!InSceneRange(particle.position)) {
    cell_index = -1;
  }
  particle.velocity += gravity_delta_t;
  particles[i] = particle;
  cell_indices[i] = cell_index;
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

template <class ParticleType, class GridEleType, class OpType, class... Args>
__device__ void NeighbourGridInteract(ParticleType particle,
                                      const glm::vec3 &grid_coord,
                                      typename Grid<GridEleType>::DevRef &grid,
                                      OpType op,
                                      Args... args) {
  glm::ivec3 nearest_index{floor(grid_coord.x), floor(grid_coord.y),
                           floor(grid_coord.z)};

  auto cell_range = grid.Range();
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        auto current_index = nearest_index + glm::ivec3{dx, dy, dz};
        if (current_index.x < 0 || current_index.y < 0 || current_index.z < 0)
          continue;
        if (current_index.x >= cell_range.x ||
            current_index.y >= cell_range.y || current_index.z >= cell_range.z)
          continue;
        auto weight =
            KernelFunction(grid_coord - glm::vec3{current_index} - 0.5f);
        if (weight > 1e-6f) {
          op(particle, grid(current_index.x, current_index.y, current_index.z),
             weight, args...);
        }
      }
    }
  }
}

__device__ void AssignVelocity(thrust::pair<float, int> v_t,
                               MACGridContent &content,
                               float weight) {
  atomicAdd(&content.w[v_t.second], weight);
  atomicAdd(&content.v[v_t.second], v_t.first * weight);
}

__global__ void Particle2GridTransferKernel(const Particle *particles,
                                            int num_particle,
                                            Grid<MACGridContent>::DevRef u_grid,
                                            Grid<MACGridContent>::DevRef v_grid,
                                            Grid<MACGridContent>::DevRef w_grid,
                                            float delta_x,
                                            glm::ivec3 cell_range) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= num_particle)
    return;
  auto particle = particles[id];
  if (!InSceneRange(particle.position))
    return;
  auto grid_coord = particle.position / delta_x;
  NeighbourGridInteract<const thrust::pair<float, int> &, MACGridContent>(
      thrust::make_pair(particle.velocity.x, particle.type),
      grid_coord + glm::vec3{0.5f, 0.0f, 0.0f}, u_grid, AssignVelocity);
  NeighbourGridInteract<const thrust::pair<float, int> &, MACGridContent>(
      thrust::make_pair(particle.velocity.y, particle.type),
      grid_coord + glm::vec3{0.0f, 0.5f, 0.0f}, v_grid, AssignVelocity);
  NeighbourGridInteract<const thrust::pair<float, int> &, MACGridContent>(
      thrust::make_pair(particle.velocity.z, particle.type),
      grid_coord + glm::vec3{0.0f, 0.0f, 0.5f}, w_grid, AssignVelocity);
}

__global__ void Particle2GridTransferGatherKernel(
    Grid<MACGridContent>::DevRef u_grid,
    Grid<MACGridContent>::DevRef v_grid,
    Grid<MACGridContent>::DevRef w_grid,
    const Particle *particles,
    int *cell_index_lower_bound,
    glm::ivec3 cell_range,
    float delta_x) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  float phi[2]{0.25f, 0.25f};
  glm::vec3 phi_gradient[2]{};
  auto grid_range = cell_range + 1;
  {
    MACGridContent u_content{};
    MACGridContent v_content{};
    MACGridContent w_content{};
    glm::ivec3 cell_index{id / (grid_range.y * grid_range.z),
                          (id / grid_range.z) % grid_range.y,
                          id % grid_range.z};
    glm::vec3 grid_point_pos = glm::vec3{cell_index} + 0.5f;
    for (int dx = -2; dx < 2; dx++) {
      for (int dy = -2; dy < 2; dy++) {
        for (int dz = -2; dz < 2; dz++) {
          glm::ivec3 current_index = cell_index + glm::ivec3{dx, dy, dz};
          if (current_index.x < 0 || current_index.y < 0 || current_index.z < 0)
            continue;
          if (current_index.x >= cell_range.x ||
              current_index.y >= cell_range.y ||
              current_index.z >= cell_range.z)
            continue;
          for (int pid = cell_index_lower_bound[RANGE_INDEX(current_index,
                                                            cell_range)],
                   last = cell_index_lower_bound[RANGE_INDEX(current_index,
                                                             cell_range) +
                                                 1];
               pid < last; pid++) {
            Particle particle = particles[pid];
            float weight;
            weight =
                KernelFunction(particle.position / delta_x - grid_point_pos +
                               glm::vec3{0.5f, 0.0f, 0.0f});
            u_content.w[particle.type] += weight;
            u_content.v[particle.type] += weight * particle.velocity.x;

            weight =
                KernelFunction(particle.position / delta_x - grid_point_pos +
                               glm::vec3{0.0f, 0.5f, 0.0f});
            v_content.w[particle.type] += weight;
            v_content.v[particle.type] += weight * particle.velocity.y;

            weight =
                KernelFunction(particle.position / delta_x - grid_point_pos +
                               glm::vec3{0.0f, 0.0f, 0.5f});
            w_content.w[particle.type] += weight;
            w_content.v[particle.type] += weight * particle.velocity.z;
          }
        }
      }
    }
    if (cell_index.x < u_grid.Range().x && cell_index.y < u_grid.Range().y &&
        cell_index.z < u_grid.Range().z)
      u_grid(cell_index) = u_content;
    if (cell_index.x < v_grid.Range().x && cell_index.y < v_grid.Range().y &&
        cell_index.z < v_grid.Range().z)
      v_grid(cell_index) = v_content;
    if (cell_index.x < w_grid.Range().x && cell_index.y < w_grid.Range().y &&
        cell_index.z < w_grid.Range().z)
      w_grid(cell_index) = w_content;
  }
}

__global__ void LowerBoundKernel(int *array,
                                 int num_element,
                                 int *lower_bound,
                                 int max_val) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_val)
    return;
  int L = 0, R = num_element;
  while (L < R) {
    int m = (L + R) >> 1;
    if (array[m] < id)
      L = m + 1;
    else
      R = m;
  }
  lower_bound[id] = R;
}

__global__ void ConstructLevelSetKernel(Grid<LevelSet_t>::DevRef level_set,
                                        const int *cell_index_lower_bound,
                                        const Particle *particles,
                                        int num_particle,
                                        float delta_x) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  auto level_set_range = level_set.Range();
  auto cell_range = level_set_range - 1;
  float phi[2]{2.0f * delta_x, 2.0f * delta_x};
  if (id < level_set.Size()) {
    glm::ivec3 cell_index{id / (level_set_range.y * level_set_range.z),
                          (id / level_set_range.z) % level_set_range.y,
                          id % level_set_range.z};
    glm::vec3 grid_point_pos = glm::vec3{cell_index} * delta_x;
    for (int dx = -1; dx < 1; dx++) {
      for (int dy = -1; dy < 1; dy++) {
        for (int dz = -1; dz < 1; dz++) {
          glm::ivec3 current_index = cell_index + glm::ivec3{dx, dy, dz};
          if (current_index.x < 0 || current_index.y < 0 || current_index.z < 0)
            continue;
          if (current_index.x >= cell_range.x ||
              current_index.y >= cell_range.y ||
              current_index.z >= cell_range.z)
            continue;
          for (int pid = cell_index_lower_bound[RANGE_INDEX(current_index,
                                                            cell_range)],
                   last = cell_index_lower_bound[RANGE_INDEX(current_index,
                                                             cell_range) +
                                                 1];
               pid < last; pid++) {
            Particle particle = particles[pid];
            float dist = glm::length(particle.position - grid_point_pos);
            phi[particle.type] = min(dist, phi[particle.type]);
          }
        }
      }
    }
    LevelSet_t result{};
    result.phi[0] = phi[0];
    result.phi[1] = phi[1];
    level_set(cell_index) = result;
  }
}

__global__ void ConstructLevelSetWeightedKernel(
    Grid<LevelSet_t>::DevRef level_set,
    Grid<LevelSetGradient_t>::DevRef level_set_gradient,
    const int *cell_index_lower_bound,
    const Particle *particles,
    int num_particle,
    float delta_x) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  auto level_set_range = level_set.Range();
  auto cell_range = level_set_range - 1;
  float phi[2]{0.25f, 0.25f};
  glm::vec3 phi_gradient[2]{};
  if (id < level_set.Size()) {
    glm::ivec3 cell_index{id / (level_set_range.y * level_set_range.z),
                          (id / level_set_range.z) % level_set_range.y,
                          id % level_set_range.z};
    glm::vec3 grid_point_pos = glm::vec3{cell_index} * delta_x;
    for (int dx = -2; dx < 2; dx++) {
      for (int dy = -2; dy < 2; dy++) {
        for (int dz = -2; dz < 2; dz++) {
          glm::ivec3 current_index = cell_index + glm::ivec3{dx, dy, dz};
          if (current_index.x < 0 || current_index.y < 0 || current_index.z < 0)
            continue;
          if (current_index.x >= cell_range.x ||
              current_index.y >= cell_range.y ||
              current_index.z >= cell_range.z)
            continue;
          for (int pid = cell_index_lower_bound[RANGE_INDEX(current_index,
                                                            cell_range)],
                   last = cell_index_lower_bound[RANGE_INDEX(current_index,
                                                             cell_range) +
                                                 1];
               pid < last; pid++) {
            Particle particle = particles[pid];
            phi[particle.type] -=
                KernelFunction(particle.position - grid_point_pos);
            phi_gradient[particle.type] -=
                KernelFunctionGradient(particle.position - grid_point_pos);
          }
        }
      }
    }
    LevelSet_t result{};
    result.phi[0] = phi[0];
    result.phi[1] = phi[1];
    LevelSetGradient_t result_gradient{};
    result_gradient.phi_gradient[0] = phi_gradient[0];
    result_gradient.phi_gradient[1] = phi_gradient[1];
    level_set(cell_index) = result;
    level_set_gradient(cell_index) = result_gradient;
  }
}

__global__ void ConstructLevelSetNearestKernel(
    Grid<LevelSet_t>::DevRef level_set,
    Grid<LevelSetGradient_t>::DevRef level_set_gradient,
    const int *cell_index_lower_bound,
    const Particle *particles,
    int num_particle,
    float delta_x) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  auto level_set_range = level_set.Range();
  auto cell_range = level_set_range - 1;
  float phi[2]{delta_x * 2.0f, delta_x * 2.0f};
  glm::vec3 phi_gradient[2]{};
  if (id < level_set.Size()) {
    glm::ivec3 cell_index{id / (level_set_range.y * level_set_range.z),
                          (id / level_set_range.z) % level_set_range.y,
                          id % level_set_range.z};
    glm::vec3 grid_point_pos = glm::vec3{cell_index} * delta_x;
    for (int dx = -2; dx < 2; dx++) {
      for (int dy = -2; dy < 2; dy++) {
        for (int dz = -2; dz < 2; dz++) {
          glm::ivec3 current_index = cell_index + glm::ivec3{dx, dy, dz};
          if (current_index.x < 0 || current_index.y < 0 || current_index.z < 0)
            continue;
          if (current_index.x >= cell_range.x ||
              current_index.y >= cell_range.y ||
              current_index.z >= cell_range.z)
            continue;
          for (int pid = cell_index_lower_bound[RANGE_INDEX(current_index,
                                                            cell_range)],
                   last = cell_index_lower_bound[RANGE_INDEX(current_index,
                                                             cell_range) +
                                                 1];
               pid < last; pid++) {
            Particle particle = particles[pid];
            float local_phi = glm::length(grid_point_pos - particle.position) -
                              delta_x * 0.5f;
            glm::vec3 local_phi_gradient = grid_point_pos - particle.position;
            if (glm::length(local_phi_gradient) > 1e-6f) {
              local_phi_gradient = glm::normalize(local_phi_gradient);
            }

            if (local_phi < phi[particle.type]) {
              phi[particle.type] = local_phi;
              phi_gradient[particle.type] = local_phi_gradient;
            }
            //            if (local_phi < 0.0f) {
            //              if (phi[particle.type] < 0.0f) {
            //                phi[particle.type] += local_phi;
            //                phi_gradient[particle.type] += local_phi_gradient;
            //              } else {
            //                phi[particle.type] = local_phi;
            //                phi_gradient[particle.type] = local_phi_gradient;
            //              }
            //            } else {
            //            }
          }
        }
      }
    }
    LevelSet_t result{};
    result.phi[0] = phi[0];
    result.phi[1] = phi[1];
    LevelSetGradient_t result_gradient{};
    result_gradient.phi_gradient[0] = phi_gradient[0];
    result_gradient.phi_gradient[1] = phi_gradient[1];
    level_set(cell_index) = result;
    level_set_gradient(cell_index) = result_gradient;
  }
}

__global__ void ProcessMACGridKernel(
    Grid<MACGridContent>::DevRef grid,
    Grid<LevelSet_t>::DevRef level_set,
    Grid<LevelSetGradient_t>::DevRef level_set_gradient,
    glm::ivec3 t_axis,
    glm::ivec3 b_axis,
    float delta_x) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  auto grid_range = grid.Range();
  glm::ivec3 cell_index{id / (grid_range.y * grid_range.z),
                        (id / grid_range.z) % grid_range.y, id % grid_range.z};
  glm::ivec3 n_axis = glm::ivec3{1, 1, 1} - t_axis - b_axis;
  if (id < grid.Size()) {
    LevelSet_t set_value[2][2] = {
        {level_set(cell_index), level_set(cell_index + t_axis)},
        {level_set(cell_index + b_axis),
         level_set(cell_index + t_axis + b_axis)}};
    LevelSetGradient_t set_value_gradient =
        (level_set_gradient(cell_index) +
         level_set_gradient(cell_index + t_axis) +
         level_set_gradient(cell_index + b_axis) +
         level_set_gradient(cell_index + t_axis + b_axis)) *
        0.25f;
    auto gradient =
        set_value_gradient.phi_gradient[0] - set_value_gradient.phi_gradient[1];
    if (glm::length(gradient) > 1e-6f) {
      gradient = glm::normalize(gradient);
    }
    float ortho{glm::dot(gradient, glm::vec3{n_axis})};
    ortho *= ortho;

    glm::vec3 set_value_pos[2][2] = {
        {glm::vec3{cell_index} * delta_x,
         glm::vec3{cell_index + t_axis} * delta_x},
        {glm::vec3{cell_index + b_axis} * delta_x,
         glm::vec3{cell_index + b_axis + t_axis} * delta_x}};

    const int precision = 8;
    const float inv_precision = 1.0f / float(precision);
    int sample_cnt[2]{};
    for (int i = 0; i < precision; i++) {
      float ax = (float(i) + 0.5f) * inv_precision;
      for (int j = 0; j < precision; j++) {
        float bx = (float(j) + 0.5f) * inv_precision;
        glm::vec3 pos = set_value_pos[0][0] * (1.0f - ax) * (1.0f - bx) +
                        set_value_pos[0][1] * (1.0f - ax) * (bx) +
                        set_value_pos[1][0] * (ax) * (1.0f - bx) +
                        set_value_pos[1][1] * (ax) * (bx);
        if (InsideFreeSpace(pos)) {
          LevelSet_t value = set_value[0][0] * (1.0f - ax) * (1.0f - bx) +
                             set_value[0][1] * (1.0f - ax) * (bx) +
                             set_value[1][0] * (ax) * (1.0f - bx) +
                             set_value[1][1] * (ax) * (bx);
          if (value.phi[0] < value.phi[1]) {
            sample_cnt[0]++;
          } else {
            sample_cnt[1]++;
          }
        }
      }
    }
    MACGridContent content = grid(cell_index);
    if (content.w[0] > 1e-6f) {
      content.v[0] /= content.w[0];
    } else {
      content.v[0] = 0.0f;
    }
    if (content.w[1] > 1e-6f) {
      content.v[1] /= content.w[1];
    } else {
      content.v[1] = 0.0f;
    }
    content.w[0] = sample_cnt[0] * inv_precision * inv_precision;
    content.w[1] = sample_cnt[1] * inv_precision * inv_precision;
    content.ortho = 1.0f;
    grid(cell_index) = content;
  }
}

__global__ void PreparePoissonEquationKernel(
    Grid<float>::DevRef divergence,
    Grid<CellCoe>::DevRef cell_coe,
    const int *cell_index_lower_bound,
    Grid<MACGridContent>::DevRef u_grid,
    Grid<MACGridContent>::DevRef v_grid,
    Grid<MACGridContent>::DevRef w_grid,
    float delta_x,
    float delta_t) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  auto grid_range = divergence.Range();
  glm::ivec3 cell_index{id / (grid_range.y * grid_range.z),
                        (id / grid_range.z) % grid_range.y, id % grid_range.z};
  auto is_vacuum = [&](const glm::ivec3 &index) {
    return false;
    glm::vec3 pos = (glm::vec3{index} + 0.5f) * delta_x;
    if (!InSceneRange(pos))
      return true;
    return cell_index_lower_bound[RANGE_INDEX(index, grid_range)] ==
           cell_index_lower_bound[RANGE_INDEX(index, grid_range) + 1];
  };

  float rho[2]{RHO_AIR, RHO_LIQ};
  auto border_sensitivity = [&](const MACGridContent &content) {
    float mixed_rho = rho[0] * content.w[0] + rho[1] * content.w[1];
    if (content.w[0] + content.w[1] > 1e-6f) {
      mixed_rho /= content.w[0] + content.w[1];
    } else {
      mixed_rho = (rho[0] + rho[1]) * 0.5f;
    }
    return (content.w[0] * (delta_t / (delta_x * rho[0])) +
            content.w[1] * (delta_t / (delta_x * rho[1]))) *
               (1.0f - content.ortho) +
           (content.w[0] + content.w[1]) * (delta_t / (delta_x * mixed_rho)) *
               content.ortho;
  };

  if (id < divergence.Size()) {
    bool this_vacuum = is_vacuum(cell_index);
    float div = 0.0f;
    CellCoe coe{};
    if (!this_vacuum) {
      MACGridContent border_content{};
      glm::ivec3 neighbor_index{};
      bool neighbor_vacuum{false};
      float delta_phi;

      /* X axis */
      neighbor_index = cell_index - glm::ivec3{1, 0, 0};
      border_content = u_grid(cell_index);
      div += border_content.w[0] * border_content.v[0] +
             border_content.w[1] * border_content.v[1];
      delta_phi = border_sensitivity(border_content);
      coe.local += delta_phi;
      if (!is_vacuum(neighbor_index)) {
        coe.x[0] -= delta_phi;
      }

      neighbor_index = cell_index + glm::ivec3{1, 0, 0};
      border_content = u_grid(cell_index + glm::ivec3{1, 0, 0});
      div -= border_content.w[0] * border_content.v[0] +
             border_content.w[1] * border_content.v[1];
      delta_phi = border_sensitivity(border_content);
      coe.local += delta_phi;
      if (!is_vacuum(neighbor_index)) {
        coe.x[1] -= delta_phi;
      }

      /* Y axis */
      neighbor_index = cell_index - glm::ivec3{0, 1, 0};
      border_content = v_grid(cell_index);
      div += border_content.w[0] * border_content.v[0] +
             border_content.w[1] * border_content.v[1];
      delta_phi = border_sensitivity(border_content);
      coe.local += delta_phi;
      if (!is_vacuum(neighbor_index)) {
        coe.y[0] -= delta_phi;
      }

      neighbor_index = cell_index + glm::ivec3{0, 1, 0};
      border_content = v_grid(cell_index + glm::ivec3{0, 1, 0});
      div -= border_content.w[0] * border_content.v[0] +
             border_content.w[1] * border_content.v[1];
      delta_phi = border_sensitivity(border_content);
      coe.local += delta_phi;
      if (!is_vacuum(neighbor_index)) {
        coe.y[1] -= delta_phi;
      }

      /* Z axis */
      neighbor_index = cell_index - glm::ivec3{0, 0, 1};
      border_content = w_grid(cell_index);
      div += border_content.w[0] * border_content.v[0] +
             border_content.w[1] * border_content.v[1];
      delta_phi = border_sensitivity(border_content);
      coe.local += delta_phi;
      if (!is_vacuum(neighbor_index)) {
        coe.z[0] -= delta_phi;
      }

      neighbor_index = cell_index + glm::ivec3{0, 0, 1};
      border_content = w_grid(cell_index + glm::ivec3{0, 0, 1});
      div -= border_content.w[0] * border_content.v[0] +
             border_content.w[1] * border_content.v[1];
      delta_phi = border_sensitivity(border_content);
      coe.local += delta_phi;
      if (!is_vacuum(neighbor_index)) {
        coe.z[1] -= delta_phi;
      }
    }
    divergence(cell_index) = div;
    cell_coe(cell_index) = coe;
  }
}

__global__ void ApplyPressureKernel(const CellCoe *cell_coe,
                                    const float *pressure,
                                    float *delta_div,
                                    glm::ivec3 range) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_cell = range.x * range.y * range.z;
  if (id < num_cell) {
    float div = 0.0f;
    CellCoe coe = cell_coe[id];
    div += coe.local * pressure[id];
    if (id > 1)
      div += coe.z[0] * pressure[id - 1];
    if (id > range.z)
      div += coe.y[0] * pressure[id - range.z];
    if (id > range.z * range.y)
      div += coe.x[0] * pressure[id - range.z * range.y];
    if (id + 1 < num_cell)
      div += coe.z[1] * pressure[id + 1];
    if (id + range.z < num_cell)
      div += coe.y[1] * pressure[id + range.z];
    if (id + range.z * range.y < num_cell)
      div += coe.x[1] * pressure[id + range.z * range.y];
    delta_div[id] = div;
  }
}

__global__ void JacobiIterationKernel(const CellCoe *cell_coe,
                                      const float *divergence,
                                      const float *pressure,
                                      float *pressure_new,
                                      glm::ivec3 range) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_cell = range.x * range.y * range.z;
  if (id < num_cell) {
    float res = 0.0f;
    CellCoe coe = cell_coe[id];
    if (coe.local > 1e-6f) {
      // coe.local *= 1.0f + 3e-4f;
      if (id > 1)
        res += coe.z[0] * pressure[id - 1];
      if (id > range.z)
        res += coe.y[0] * pressure[id - range.z];
      if (id > range.z * range.y)
        res += coe.x[0] * pressure[id - range.z * range.y];
      if (id + 1 < num_cell)
        res += coe.z[1] * pressure[id + 1];
      if (id + range.z < num_cell)
        res += coe.y[1] * pressure[id + range.z];
      if (id + range.z * range.y < num_cell)
        res += coe.x[1] * pressure[id + range.z * range.y];
      res = (divergence[id] - res) / coe.local;
    }
    pressure_new[id] = res;
  }
}

__global__ void UpdateVelocityFieldKernel(Grid<MACGridContent>::DevRef grid,
                                          Grid<float>::DevRef pressure,
                                          glm::ivec3 n_axis,
                                          float delta_x,
                                          float delta_t) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  glm::ivec3 grid_range = grid.Range();
  glm::ivec3 cell_index{id / (grid_range.y * grid_range.z),
                        (id / grid_range.z) % grid_range.y, id % grid_range.z};
  const auto pressure_range = pressure.Range();
  if (id < grid.Size()) {
    float pressure_low = 0.0f;
    float pressure_high = 0.0f;
    if (cell_index.x < pressure_range.x && cell_index.y < pressure_range.y &&
        cell_index.z < pressure_range.z) {
      pressure_high = pressure(cell_index);
    }
    if (cell_index.x >= n_axis.x && cell_index.y >= n_axis.y &&
        cell_index.z >= n_axis.z) {
      pressure_low = pressure(cell_index - n_axis);
    }
    auto content = grid(cell_index);
    float block_weight = 0.0f;  // 1.0f - content.w[0] - content.w[1];

    float mixed_rho = RHO_AIR * content.w[0] + RHO_LIQ * content.w[1];
    if (content.w[0] + content.w[1] > 1e-6f) {
      mixed_rho /= content.w[0] + content.w[1];
    } else {
      mixed_rho = (RHO_AIR + RHO_LIQ) * 0.5f;
    }

    content.v[0] = content.v[0] * PIC_SCALE +
                   ((pressure_low - pressure_high) * delta_t /
                        (delta_x * RHO_AIR) * (1.0f - content.ortho) +
                    (pressure_low - pressure_high) * delta_t /
                        (delta_x * mixed_rho) * content.ortho);
    content.v[1] = content.v[1] * PIC_SCALE +
                   ((pressure_low - pressure_high) * delta_t /
                        (delta_x * RHO_LIQ) * (1.0f - content.ortho) +
                    (pressure_low - pressure_high) * delta_t /
                        (delta_x * mixed_rho) * content.ortho);
    grid(cell_index) = content;
  }
}

__device__ void GatherVelocity(thrust::pair<float, float> &v_wv,
                               const MACGridContent content,
                               float weight,
                               int type) {
  //  float wv = content.v[0] * content.w[0] + content.v[1] * content.w[1];
  //  if (content.w[0] + content.w[1] > 1e-6f) {
  //    wv /= content.w[0] + content.w[1];
  //  }

  //  v_wv.first += weight * (1.0f - content.w[type ^ 1]) * content.v[type];
  //  v_wv.second += weight * (1.0f - content.w[type ^ 1]);

  v_wv.first += weight * content.w[type] * content.v[type];
  v_wv.second += weight * content.w[type];
}

__global__ void Grid2ParticleTransferKernel(Particle *particles,
                                            int num_particle,
                                            Grid<MACGridContent>::DevRef u_grid,
                                            Grid<MACGridContent>::DevRef v_grid,
                                            Grid<MACGridContent>::DevRef w_grid,
                                            float delta_x,
                                            glm::ivec3 cell_range) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= num_particle)
    return;
  auto particle = particles[id];
  if (!InSceneRange(particle.position))
    return;
  auto grid_coord = particle.position / delta_x;
  thrust::pair<float, float> x_wx{};
  thrust::pair<float, float> y_wy{};
  thrust::pair<float, float> z_wz{};
  NeighbourGridInteract<thrust::pair<float, float> &, MACGridContent>(
      x_wx, grid_coord + glm::vec3{0.5f, 0.0f, 0.0f}, u_grid, GatherVelocity,
      particle.type);
  NeighbourGridInteract<thrust::pair<float, float> &, MACGridContent>(
      y_wy, grid_coord + glm::vec3{0.0f, 0.5f, 0.0f}, v_grid, GatherVelocity,
      particle.type);
  NeighbourGridInteract<thrust::pair<float, float> &, MACGridContent>(
      z_wz, grid_coord + glm::vec3{0.0f, 0.0f, 0.5f}, w_grid, GatherVelocity,
      particle.type);
  if (x_wx.second > 1e-6f) {
    x_wx.first /= x_wx.second;
  }
  if (y_wy.second > 1e-6f) {
    y_wy.first /= y_wy.second;
  }
  if (z_wz.second > 1e-6f) {
    z_wz.first /= z_wz.second;
  }
  particle.velocity = particle.velocity * (1.0f - PIC_SCALE) +
                      glm::vec3{x_wx.first, y_wy.first, z_wz.first};
  particles[id] = particle;
}

__device__ bool CorrectSide(const Particle &particle,
                            const LevelSet_t &level_set_value) {
  return level_set_value.phi[particle.type] <
         level_set_value.phi[particle.type ^ 1];
}

__global__ void ResetWrongParticleKernel(Particle *particles,
                                         curandState_t *rand_states,
                                         int num_particle,
                                         int *cell_indices,
                                         glm::ivec3 cell_range,
                                         Grid<LevelSet_t>::DevRef level_set,
                                         Grid<MACGridContent>::DevRef u_grid,
                                         Grid<MACGridContent>::DevRef v_grid,
                                         Grid<MACGridContent>::DevRef w_grid,
                                         float delta_x) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= num_particle)
    return;
  auto particle = particles[id];
  auto rd = rand_states[id];
  bool reset = false;
  auto grid_coord = particle.position / delta_x;
  while (!InsideFreeSpace(particle.position) ||
         !CorrectSide(particle, level_set.Sample(grid_coord))) {
    particle.position =
        SceneRange() * glm::vec3{curand_uniform(&rd), curand_uniform(&rd),
                                 curand_uniform(&rd)};
    grid_coord = particle.position / delta_x;
    reset = true;
  }

  if (reset) {
    particle.velocity =
        glm::vec3{u_grid.Sample(grid_coord - glm::vec3{0.0f, 0.5f, 0.5f})
                      .v[particle.type],
                  v_grid.Sample(grid_coord - glm::vec3{0.5f, 0.0f, 0.5f})
                      .v[particle.type],
                  w_grid.Sample(grid_coord - glm::vec3{0.5f, 0.5f, 0.0f})
                      .v[particle.type]};
    cell_indices[id] =
        RANGE_INDEX(glm::ivec3{particle.position / delta_x}, cell_range);
    rand_states[id] = rd;
    particles[id] = particle;
  }
}

void PhysicSolver::UpdateStep() {
  static int round = 0;
  printf("-- Round #%d --\n", round++);
  DeviceClock dev_clock;

  level_set_.Clear();
  u_grid_.Clear();
  v_grid_.Clear();
  w_grid_.Clear();

  dev_clock.Record("Clean data");

  ParticleLinearOperationsKernel<<<CALL_GRID(particles_.size())>>>(
      particles_.data().get(), cell_indices_.data().get(),
      physic_settings_.gravity * physic_settings_.delta_t,
      physic_settings_.delta_x, cell_range_, particles_.size());
  dev_clock.Record("Particle Linear Operations Kernel");

  thrust::sort_by_key(cell_indices_.begin(), cell_indices_.end(),
                      particles_.begin());
  dev_clock.Record("Sort by Cells");

  LowerBoundKernel<<<CALL_GRID(cell_index_lower_bound_.size())>>>(
      cell_indices_.data().get(), cell_indices_.size(),
      cell_index_lower_bound_.data().get(), cell_index_lower_bound_.size());
  dev_clock.Record("Cell Index Lower Bound");

  Particle2GridTransferKernel<<<CALL_GRID(particles_.size())>>>(
      particles_.data().get(), particles_.size(), u_grid_, v_grid_, w_grid_,
      physic_settings_.delta_x, cell_range_);
  dev_clock.Record("Scatter Grid2Particle");

  //  Particle2GridTransferGatherKernel<<<CALL_GRID(level_set_.Size())>>>(
  //      u_grid_, v_grid_, w_grid_, particles_.data().get(),
  //      cell_index_lower_bound_.data().get(), cell_range_,
  //      physic_settings_.delta_x);
  //  dev_clock.Record("Gather Grid2Particle");
  //
  //  ConstructLevelSetKernel<<<CALL_GRID(level_sets_.Size())>>>(
  //      level_sets_, cell_index_lower_bound_.data().get(),
  //      particles_.data().get(), particles_.size(), physic_settings_.delta_x);

  ConstructLevelSetNearestKernel<<<CALL_GRID(level_set_.Size())>>>(
      level_set_, level_set_gradient_, cell_index_lower_bound_.data().get(),
      particles_.data().get(), particles_.size(), physic_settings_.delta_x);
  dev_clock.Record("Construct Level Set");

  ProcessMACGridKernel<<<CALL_GRID(u_grid_.Size())>>>(
      u_grid_, level_set_, level_set_gradient_, glm::ivec3{0, 1, 0},
      glm::ivec3{0, 0, 1}, physic_settings_.delta_x);
  ProcessMACGridKernel<<<CALL_GRID(v_grid_.Size())>>>(
      v_grid_, level_set_, level_set_gradient_, glm::ivec3{0, 0, 1},
      glm::ivec3{1, 0, 0}, physic_settings_.delta_x);
  ProcessMACGridKernel<<<CALL_GRID(w_grid_.Size())>>>(
      w_grid_, level_set_, level_set_gradient_, glm::ivec3{1, 0, 0},
      glm::ivec3{0, 1, 0}, physic_settings_.delta_x);
  dev_clock.Record("Process MAC grid");

  ResetWrongParticleKernel<<<CALL_GRID(particles_.size())>>>(
      particles_.data().get(), dev_rand_states_.data().get(), particles_.size(),
      cell_indices_.data().get(), cell_range_, level_set_, u_grid_, v_grid_,
      w_grid_, physic_settings_.delta_x);
  dev_clock.Record("Reset Wrong Particles");

  PreparePoissonEquationKernel<<<CALL_GRID(divergence_.Size())>>>(
      divergence_, cell_coe_, cell_index_lower_bound_.data().get(), u_grid_,
      v_grid_, w_grid_, physic_settings_.delta_x, physic_settings_.delta_t);
  dev_clock.Record("Prepare Poisson Equation");

  //  ConjugateGradient(
  //      [this](const thrust::device_vector<float> &pressure,
  //             thrust::device_vector<float> &delta_gradient) {
  //        ApplyPressureKernel<<<CALL_GRID(pressure_.Size())>>>(
  //            cell_coe_.Vector().data().get(), pressure.data().get(),
  //            delta_gradient.data().get(), pressure_.Range());
  //      },
  //      divergence_.Vector(), pressure_.Vector());
  //  dev_clock.Record("Solve Poisson Equation (Conjugate Gradient)");
  //
  //  JacobiMethod(
  //      [this](const thrust::device_vector<float> &divergence,
  //             const thrust::device_vector<float> &pressure,
  //             thrust::device_vector<float> &pressure_new) {
  //        JacobiIterationKernel<<<CALL_GRID(pressure_.Size())>>>(
  //            cell_coe_.Vector().data().get(), divergence.data().get(),
  //            pressure.data().get(), pressure_new.data().get(),
  //            pressure_.Range());
  //      },
  //      divergence_.Vector(), pressure_.Vector());
  //  dev_clock.Record("Solve Poisson Equation (Jacobi Iteration)");

  MultiGrid(cell_coe_, divergence_, pressure_);
  MultiGrid(cell_coe_, divergence_, pressure_);
  dev_clock.Record("Solve Poisson Equation (Multi Grid)");

  UpdateVelocityFieldKernel<<<CALL_GRID(u_grid_.Size())>>>(
      u_grid_, pressure_, glm::ivec3{1, 0, 0}, physic_settings_.delta_x,
      physic_settings_.delta_t);
  UpdateVelocityFieldKernel<<<CALL_GRID(v_grid_.Size())>>>(
      v_grid_, pressure_, glm::ivec3{0, 1, 0}, physic_settings_.delta_x,
      physic_settings_.delta_t);
  UpdateVelocityFieldKernel<<<CALL_GRID(w_grid_.Size())>>>(
      w_grid_, pressure_, glm::ivec3{0, 0, 1}, physic_settings_.delta_x,
      physic_settings_.delta_t);
  dev_clock.Record("Update Velocity Field");

  Grid2ParticleTransferKernel<<<CALL_GRID(particles_.size())>>>(
      particles_.data().get(), particles_.size(), u_grid_, v_grid_, w_grid_,
      physic_settings_.delta_x, cell_range_);
  dev_clock.Record("Grid2Particle Transfer");

  AdvectionKernel<<<CALL_GRID(particles_.size())>>>(
      particles_.data().get(), physic_settings_.delta_t, particles_.size());
  dev_clock.Record("Advection");

  InstanceInfoComposeKernel<<<CALL_GRID(dev_instance_infos_.size())>>>(
      particles_.data().get(), dev_instance_infos_.data().get(),
      particles_.size(), level_set_gradient_, physic_settings_.delta_x);
  dev_clock.Record("Compose Instance Info");

  dev_clock.Finish();

  OutputXYZFile();
}

void PhysicSolver::OutputXYZFile() {
  std::vector<Particle> particles(particles_.size());
  thrust::copy(particles_.begin(), particles_.end(), particles.begin());
  static int round = 0;
  if (!round)
    std::system("mkdir data");
  if (round % 20 == 0) {
    std::ofstream file("data/" + std::to_string(round) + FILE_SUFFIX ".xyz",
                       std::ios::binary);
    int cnt = 0;
    for (auto &particle : particles) {
      if (particle.type == 1) {
        cnt++;
        if (cnt == 1) {
          file.write(reinterpret_cast<const char *>(&particle.position),
                     sizeof(particle.position));
          cnt = 0;
        }
      }
    }
    file.close();
  }
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
  glm::vec3 range = SceneRange();
  if (glm::length(position - glm::vec3{range.x * 0.5f, range.y * 0.7f,
                                       range.z * 0.5f}) < range.x * 0.4f &&
      position.y < range.y * 0.7f) {
    return 1;
  } else {
    return 0;
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

__global__ void InstanceInfoComposeKernel(
    const Particle *particles,
    InstanceInfo *instance_infos,
    int num_particle,
    Grid<LevelSetGradient_t>::DevRef phi_gradient,
    float delta_x) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particle)
    return;
  auto particle = particles[i];
  InstanceInfo info{};
  info.size = PARTICLE_SIZE;
  info.offset = particle.position;
  info.color = particle.type ? glm::vec4{0.5f, 0.5f, 1.0f, 1.0f}
                             : glm::vec4{1.0f, 0.5f, 0.5f, 1.0f};
  //  info.color = glm::vec4{
  //      glm::vec3{0.5f} + glm::normalize(phi_gradient.Sample(particle.position
  //      / delta_x).phi_gradient[particle.type]) * 0.5f, 1.0f
  //  };
  if (!InsideFreeSpace(particle.position)) {
    info.color = (info.color + glm::vec4{1.0f, 1.0f, 1.0f, 1.0f}) * 0.5f;
  }
  instance_infos[i] = info;
}
