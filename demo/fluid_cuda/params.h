#pragma once

#define NUM_PARTICLE (8192 << 4)
#define RENDER_SIZE 0.05f
#define GRAVITY       \
  glm::vec3 {         \
    0.0f, -9.8f, 0.0f \
  }

#define GRID_SIZE_X 40
#define GRID_SIZE_Y 80
#define GRID_SIZE_Z 40
#define DELTA_X 0.5f
#define SIZE_X (GRID_SIZE_X * DELTA_X)
#define SIZE_Y (GRID_SIZE_Y * DELTA_X)
#define SIZE_Z (GRID_SIZE_Z * DELTA_X)
#define TYPE_AIR 0
#define TYPE_LIQ 1
#define RHO_AIR 0.01f
#define RHO_LIQ 1.0f
#define GRID_POINT_ID(x, y, z) \
  ((x)*GRID_SIZE_Y * GRID_SIZE_Z + (y)*GRID_SIZE_Z + (z))
#define PIC_SCALE 0.03f
