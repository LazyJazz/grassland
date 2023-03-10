#pragma once
#include "cstdint"
#include "curand.h"
#include "curand_kernel.h"

void CurandFloat(float *data, uint64_t length);
