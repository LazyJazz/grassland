#pragma once

#define BLOCK_SIZE 256
#define CALL_GRID(x) ((x + BLOCK_SIZE - 1) / BLOCK_SIZE),  BLOCK_SIZE
