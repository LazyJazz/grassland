#pragma once
#include "thrust/device_vector.h"
#include "glm/glm.hpp"
#include "scene.cuh"
#include "grid.cuh"

struct InstanceInfo;

struct Particle {
  glm::vec3 position;
  glm::vec3 velocity;
  int type;
};

struct PhysicSettings {
  int num_particle{400*800*400};
  float delta_x{1.0f / 400.0f};
  float delta_t{1e-2f};
  glm::vec3 gravity{0.0f, -9.8f, 0.0f};
};

struct MACGridContent {
  float v[2];
  float w[2];
};

struct LevelSet_t {
  int type;
  float phi;
};

class PhysicSolver {
 public:
  explicit PhysicSolver(const PhysicSettings& physic_settings);
  void GetInstanceInfoArray(InstanceInfo* instance_infos) const;
  void UpdateStep();
  void OutputXYZFile();
 private:
  thrust::device_vector<Particle> particles_;
  thrust::device_vector<int> cell_indices_;
  thrust::device_vector<int> cell_index_lower_bound_;
  thrust::device_vector<InstanceInfo> dev_instance_infos_;

  Grid<LevelSet_t> level_sets_;
  Grid<MACGridContent> u_grid_;
  Grid<MACGridContent> v_grid_;
  Grid<MACGridContent> w_grid_;

  PhysicSettings physic_settings_;
  glm::ivec3 cell_range_{};
  glm::ivec3 block_range_{};
};

__device__ __host__ LevelSet_t LevelSetInterpolate(const LevelSet_t& ls_0, const LevelSet_t& ls_1, float alpha);
