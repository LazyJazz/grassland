#pragma once

#define EPS 1e-6f

#define BLOCK_SIZE 256
#define CALL_GRID(x) ((x + BLOCK_SIZE - 1) / BLOCK_SIZE),  BLOCK_SIZE
#define RANGE_INDEX_XYZ(X, Y, Z, range) ((X) * range.y * range.z + (Y) * range.z + (Z))
#define RANGE_INDEX(index3, range) (index3.x * range.y * range.z + index3.y * range.z + index3.z)

#define BLOCK_BIT_COUNT_X (3)
#define BLOCK_DIM_SIZE_X (1 << BLOCK_BIT_COUNT_X)
#define BLOCK_BIT_MASK_X (BLOCK_DIM_SIZE_X - 1)

#define BLOCK_BIT_COUNT_Y (3)
#define BLOCK_DIM_SIZE_Y (1 << BLOCK_BIT_COUNT_Y)
#define BLOCK_BIT_MASK_Y (BLOCK_DIM_SIZE_Y - 1)

#define BLOCK_BIT_COUNT_Z (3)
#define BLOCK_DIM_SIZE_Z (1 << BLOCK_BIT_COUNT_Z)
#define BLOCK_BIT_MASK_Z (BLOCK_DIM_SIZE_Z - 1)

#define BLOCK_BIT_MASK_V3 glm::ivec3{BLOCK_BIT_MASK_X, BLOCK_BIT_MASK_Y, BLOCK_BIT_MASK_Z}
#define BLOCK_DIM_SIZE_V3 glm::ivec3{BLOCK_DIM_SIZE_X, BLOCK_DIM_SIZE_Y, BLOCK_DIM_SIZE_Z}
#define BLOCK_BIT_COUNT_V3 glm::ivec3{BLOCK_BIT_COUNT_X, BLOCK_BIT_COUNT_Y, BLOCK_BIT_COUNT_Z}

#define VECTOR_ELEMENT_PRODUCT(v) (v.x * v.y * v.z)
