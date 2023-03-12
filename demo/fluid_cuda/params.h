#pragma once

#define NUM_PARTICLE 65536
#define RENDER_SIZE 0.05f
#define DELTA_T 1e-2f;
#define GRAVITY       \
  glm::vec3 {         \
    0.0f, -9.8f, 0.0f \
  }
#define BOUNDARY_CENTER glm::vec3(10.0f, 10.0f, 10.0f)
#define BOUNDARY_RADIUS 8.0f
