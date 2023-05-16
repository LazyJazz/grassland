#pragma once
#include "cuda_runtime.h"
#include "glm/glm.hpp"


__device__ __host__ bool InsideFreeSpace(const glm::vec3& position);

__device__ __host__ glm::vec3 SceneRange();

__device__ __host__ int AssignType(const glm::vec3& position);

__device__ __host__ bool InSceneRange(const glm::vec3& position);
