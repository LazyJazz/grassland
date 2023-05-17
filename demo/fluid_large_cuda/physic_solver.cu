#include "curand_kernel.h"
#include "device_clock.cuh"
#include "fluid_large.cuh"
#include "linear_solvers.cuh"
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
  dev_instance_infos_.resize(physic_settings_.num_particle);
  InitParticleKernel<<<CALL_GRID(particles_.size())>>>(particles_.data().get(),
                                                       particles_.size());
  cell_range_ = SceneRange() / physic_settings_.delta_x;
  block_range_ = (cell_range_ + BLOCK_BIT_MASK_V3) / BLOCK_DIM_SIZE_V3;
  printf("Cell [%d %d %d]\n", cell_range_.x, cell_range_.y, cell_range_.z);
  printf("Block [%d %d %d] * (8 8 8)\n", block_range_.x, block_range_.y,
         block_range_.z);

  cell_index_lower_bound_.resize(Grid<int>::BufferSize(cell_range_) + 1);

  level_sets_ = Grid<LevelSet_t>(cell_range_ + 1);
  u_grid_ = Grid<MACGridContent>(cell_range_ + glm::ivec3{1, 0, 0});
  v_grid_ = Grid<MACGridContent>(cell_range_ + glm::ivec3{0, 1, 0});
  w_grid_ = Grid<MACGridContent>(cell_range_ + glm::ivec3{0, 0, 1});

  divergence_ = Grid<float>(cell_range_);
  pressure_ = Grid<float>(cell_range_);
  cell_coe_ = Grid<CellCoe>(cell_range_);
}

__global__ void InstanceInfoComposeKernel(const Particle *particles,
                                          InstanceInfo *instance_infos,
                                          int num_particle);

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

__global__ void ProcessMACGridKernel(Grid<MACGridContent>::DevRef grid,
                                     Grid<LevelSet_t>::DevRef level_set,
                                     glm::ivec3 t_axis,
                                     glm::ivec3 b_axis,
                                     float delta_x) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  auto grid_range = grid.Range();
  glm::ivec3 cell_index{id / (grid_range.y * grid_range.z),
                        (id / grid_range.z) % grid_range.y, id % grid_range.z};
  if (id < grid.Size()) {
    LevelSet_t set_value[2][2] = {
        {level_set(cell_index), level_set(cell_index + t_axis)},
        {level_set(cell_index + b_axis),
         level_set(cell_index + t_axis + b_axis)}};
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
      delta_phi = border_content.w[0] * (delta_t / (delta_x * rho[0])) +
                  border_content.w[1] * (delta_t / (delta_x * rho[1]));
      coe.local += delta_phi;
      if (!is_vacuum(neighbor_index)) {
        coe.x[0] -= delta_phi;
      }

      neighbor_index = cell_index + glm::ivec3{1, 0, 0};
      border_content = u_grid(cell_index + glm::ivec3{1, 0, 0});
      div -= border_content.w[0] * border_content.v[0] +
             border_content.w[1] * border_content.v[1];
      delta_phi = border_content.w[0] * (delta_t / (delta_x * rho[0])) +
                  border_content.w[1] * (delta_t / (delta_x * rho[1]));
      coe.local += delta_phi;
      if (!is_vacuum(neighbor_index)) {
        coe.x[1] -= delta_phi;
      }

      /* Y axis */
      neighbor_index = cell_index - glm::ivec3{0, 1, 0};
      border_content = v_grid(cell_index);
      div += border_content.w[0] * border_content.v[0] +
             border_content.w[1] * border_content.v[1];
      delta_phi = border_content.w[0] * (delta_t / (delta_x * rho[0])) +
                  border_content.w[1] * (delta_t / (delta_x * rho[1]));
      coe.local += delta_phi;
      if (!is_vacuum(neighbor_index)) {
        coe.y[0] -= delta_phi;
      }

      neighbor_index = cell_index + glm::ivec3{0, 1, 0};
      border_content = v_grid(cell_index + glm::ivec3{0, 1, 0});
      div -= border_content.w[0] * border_content.v[0] +
             border_content.w[1] * border_content.v[1];
      delta_phi = border_content.w[0] * (delta_t / (delta_x * rho[0])) +
                  border_content.w[1] * (delta_t / (delta_x * rho[1]));
      coe.local += delta_phi;
      if (!is_vacuum(neighbor_index)) {
        coe.y[1] -= delta_phi;
      }

      /* Z axis */
      neighbor_index = cell_index - glm::ivec3{0, 0, 1};
      border_content = w_grid(cell_index);
      div += border_content.w[0] * border_content.v[0] +
             border_content.w[1] * border_content.v[1];
      delta_phi = border_content.w[0] * (delta_t / (delta_x * rho[0])) +
                  border_content.w[1] * (delta_t / (delta_x * rho[1]));
      coe.local += delta_phi;
      if (!is_vacuum(neighbor_index)) {
        coe.z[0] -= delta_phi;
      }

      neighbor_index = cell_index + glm::ivec3{0, 0, 1};
      border_content = w_grid(cell_index + glm::ivec3{0, 0, 1});
      div -= border_content.w[0] * border_content.v[0] +
             border_content.w[1] * border_content.v[1];
      delta_phi = border_content.w[0] * (delta_t / (delta_x * rho[0])) +
                  border_content.w[1] * (delta_t / (delta_x * rho[1]));
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
    content.v[0] = content.v[0] * PIC_SCALE + ((pressure_low - pressure_high) *
                                               delta_t / (delta_x * RHO_AIR)) *
                                                  (1.0f - PIC_SCALE);
    content.v[1] = content.v[1] * PIC_SCALE + ((pressure_low - pressure_high) *
                                               delta_t / (delta_x * RHO_LIQ)) *
                                                  (1.0f - PIC_SCALE);
    grid(cell_index) = content;
  }
}

__device__ void GatherVelocity(thrust::pair<float, float> &v_wv,
                               const MACGridContent content,
                               float weight,
                               int type) {
  float wv = content.v[0] * content.w[0] + content.v[1] * content.w[1];
  if (content.w[0] + content.w[1] > 1e-6f) {
    wv /= content.w[0] + content.w[1];
  }
  v_wv.first += weight * content.w[type] * wv;
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

void PhysicSolver::UpdateStep() {
  DeviceClock dev_clock;

  level_sets_.Clear();
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
  dev_clock.Record("Trivial Grid2Particle");

  ConstructLevelSetKernel<<<CALL_GRID(level_sets_.Size())>>>(
      level_sets_, cell_index_lower_bound_.data().get(),
      particles_.data().get(), particles_.size(), physic_settings_.delta_x);
  dev_clock.Record("Construct Level Set");

  ProcessMACGridKernel<<<CALL_GRID(u_grid_.Size())>>>(
      u_grid_, level_sets_, glm::ivec3{0, 1, 0}, glm::ivec3{0, 0, 1},
      physic_settings_.delta_x);
  ProcessMACGridKernel<<<CALL_GRID(v_grid_.Size())>>>(
      v_grid_, level_sets_, glm::ivec3{0, 1, 0}, glm::ivec3{0, 0, 1},
      physic_settings_.delta_x);
  ProcessMACGridKernel<<<CALL_GRID(w_grid_.Size())>>>(
      w_grid_, level_sets_, glm::ivec3{0, 1, 0}, glm::ivec3{0, 0, 1},
      physic_settings_.delta_x);
  dev_clock.Record("Process MAC grid");

  PreparePoissonEquationKernel<<<CALL_GRID(divergence_.Size())>>>(
      divergence_, cell_coe_, cell_index_lower_bound_.data().get(), u_grid_,
      v_grid_, w_grid_, physic_settings_.delta_x, physic_settings_.delta_t);
  dev_clock.Record("Prepare Poisson Equation");

  ConjugateGradient(
      [this](const thrust::device_vector<float> &pressure,
             thrust::device_vector<float> &delta_gradient) {
        ApplyPressureKernel<<<CALL_GRID(pressure_.Size())>>>(
            cell_coe_.Vector().data().get(), pressure.data().get(),
            delta_gradient.data().get(), pressure_.Range());
      },
      divergence_.Vector(), pressure_.Vector());
  dev_clock.Record("Solve Poisson Equation");

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

  //        GridLinearHost<MACGridContent> u_grid_host(u_grid_);
  //      GridLinearHost<MACGridContent> v_grid_host(v_grid_);
  //      GridLinearHost<MACGridContent> w_grid_host(w_grid_);

  //   GridLinearHost<LevelSet_t> level_set_host(level_sets_);
  //      GridLinearHost<float> divergence_host(divergence_);
  //      GridLinearHost<CellCoe> cell_coe_host(cell_coe_);
  //        GridLinearHost<float> pressure_host(pressure_);
  //        auto &print_grid = v_grid_host;
  //        std::ofstream csv_file("grid.csv");
  //        for (int y = 0; y < print_grid.Range().y; y++) {
  //        for (int z = 0; z < print_grid.Range().z; z++) {
  //          auto content = print_grid(print_grid.Range().x / 2, y, z);
  //          csv_file << std::to_string(content.v[0]) << ",";
  //        }
  //        csv_file << std::endl;
  //        }
  //        csv_file.close();
  //        std::system("start grid.csv");
  //
  //        while (!GetAsyncKeyState(VK_ESCAPE))
  //          ;
  //        while (GetAsyncKeyState(VK_ESCAPE))
  //          ;
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

__global__ void InstanceInfoComposeKernel(const Particle *particles,
                                          InstanceInfo *instance_infos,
                                          int num_particle) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_particle)
    return;
  auto particle = particles[i];
  InstanceInfo info{};
  info.size = 3e-3f;
  info.offset = particle.position;
  info.color = particle.type ? glm::vec4{0.5f, 0.5f, 1.0f, 1.0f}
                             : glm::vec4{1.0f, 0.5f, 0.5f, 1.0f};
  instance_infos[i] = info;
}
