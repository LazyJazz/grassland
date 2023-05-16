#pragma once

#define BLOCK_SIZE 256
#define CALL_GRID(x) ((x + BLOCK_SIZE - 1) / BLOCK_SIZE),  BLOCK_SIZE
#define RANGE_INDEX_XYZ(X, Y, Z, range) ((X) * range.y * range.z + (Y) * range.z + (Z))
#define RANGE_INDEX(index3, range) (index3.x * range.y * range.z + index3.y * range.z + index3.z)
