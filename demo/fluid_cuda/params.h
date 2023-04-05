#pragma once

#define NUM_PARTICLE (65536 >> 3)
#define RENDER_SIZE 0.1f
#define GRAVITY       \
  glm::vec3 {         \
    0.0f, -9.8f, 0.0f \
  }

#define GRID_SIZE_X 20
#define GRID_SIZE_Y 40
#define GRID_SIZE_Z 20
#define DELTA_X 1.0f
#define SIZE_X (GRID_SIZE_X * DELTA_X)
#define SIZE_Y (GRID_SIZE_Y * DELTA_X)
#define SIZE_Z (GRID_SIZE_Z * DELTA_X)
#define TYPE_AIR 0
#define TYPE_LIQ 1
#define RHO_AIR 0.01f
#define RHO_LIQ 1.0f
#define GRID_POINT_ID(x, y, z) \
  ((x)*GRID_SIZE_Y * GRID_SIZE_Z + (y)*GRID_SIZE_Z + (z))
