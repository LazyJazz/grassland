#pragma once
#include "curand_kernel.h"
#include "glm/glm.hpp"
#include "grid.cuh"
#include "linear_solvers.cuh"
#include "scene.cuh"
#include "thrust/device_vector.h"

#define RHO_AIR 0.01f
#define RHO_LIQ 1.0f
#define TYPE_AIR 0
#define TYPE_LIQ 1

#define PIC_SCALE 0.03f
#define SCALE 320
#define PARTICLE_SIZE (0.05f / float(SCALE))

#define FILE_SUFFIX "_320"

struct InstanceInfo;

struct Particle {
  glm::vec3 position;
  glm::vec3 velocity;
  int type;
};

struct PhysicSettings {
  int num_particle{SCALE * (2 * SCALE) * SCALE};
  float delta_x{1.0f / float(SCALE)};
  float delta_t{1e-3f};
  glm::vec3 gravity{0.0f, -9.8f, 0.0f};
};

struct MACGridContent {
  float v[2];
  float w[2];
  float ortho;
  __device__ __host__ MACGridContent operator+(const MACGridContent &b) const {
    return {v[0] + b.v[0], v[1] + b.v[1], w[0] + b.w[0], w[1] + b.w[1],
            ortho + b.ortho};
  }
  __device__ __host__ MACGridContent operator*(float s) const {
    return {v[0] * s, v[1] * s, w[0] * s, w[1] * s, ortho * s};
  }
};

struct LevelSet_t {
  float phi[2];
  __device__ __host__ LevelSet_t operator+(const LevelSet_t &b) const {
    return {phi[0] + b.phi[0], phi[1] + b.phi[1]};
  }
  __device__ __host__ LevelSet_t operator*(float s) const {
    return {phi[0] * s, phi[1] * s};
  }
};

struct LevelSetGradient_t {
  glm::vec3 phi_gradient[2];
  __device__ __host__ LevelSetGradient_t
  operator+(const LevelSetGradient_t &b) const {
    return {phi_gradient[0] + b.phi_gradient[0],
            phi_gradient[1] + b.phi_gradient[1]};
  }
  __device__ __host__ LevelSetGradient_t operator*(float s) const {
    return {phi_gradient[0] * s, phi_gradient[1] * s};
  }
};

class PhysicSolver {
 public:
  explicit PhysicSolver(const PhysicSettings &physic_settings);
  void GetInstanceInfoArray(InstanceInfo *instance_infos) const;
  void UpdateStep();
  void OutputXYZFile();

 private:
  thrust::device_vector<Particle> particles_;
  thrust::device_vector<int> cell_indices_;
  thrust::device_vector<int> cell_index_lower_bound_;
  thrust::device_vector<InstanceInfo> dev_instance_infos_;
  thrust::device_vector<curandState_t> dev_rand_states_;

  Grid<LevelSet_t> level_set_;
  Grid<LevelSetGradient_t> level_set_gradient_;
  Grid<MACGridContent> u_grid_;
  Grid<MACGridContent> v_grid_;
  Grid<MACGridContent> w_grid_;

  Grid<float> divergence_;
  Grid<float> pressure_;
  Grid<CellCoe> cell_coe_;

  PhysicSettings physic_settings_;
  glm::ivec3 cell_range_{};
  glm::ivec3 block_range_{};
};
