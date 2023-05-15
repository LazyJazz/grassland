#pragma once
#include "thrust/device_vector.h"
#include "glm/glm.hpp"
#include "scene.cuh"

struct InstanceInfo;

struct Particle {
  glm::vec3 position;
  glm::vec3 velocity;
  int type;
};

struct PhysicSettings {
  int num_particle{400*800*400};
  float delta_x{0.5f};
  float delta_t{1e-2f};
  glm::vec3 gravity{0.0f, -9.8f, 0.0f};
};

class PhysicSolver {
 public:
  explicit PhysicSolver(const PhysicSettings& physic_settings);
  void GetInstanceInfoArray(InstanceInfo* instance_infos) const;
  void UpdateStep();
 private:
  thrust::device_vector<Particle> particles_;
  PhysicSettings physic_settings_;
  glm::ivec3 grid_range_{};
};
