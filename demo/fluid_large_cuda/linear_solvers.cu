#include "glm/glm.hpp"
#include "linear_solvers.cuh"

__global__ void CalcResidualKernel(const CellCoe *cell_coe,
                                   const float *x,
                                   const float *b,
                                   float *r_2h,
                                   glm::ivec3 range,
                                   glm::ivec3 range_2h) {
  int id_2h = blockIdx.x * blockDim.x + threadIdx.x;
  glm::ivec3 cell_index_2h{id_2h / (range_2h.y * range_2h.z),
                           (id_2h / range_2h.z) % range_2h.y,
                           id_2h % range_2h.z};
  int num_cell_2h = range_2h.x * range_2h.y * range_2h.z;
  if (id_2h >= num_cell_2h)
    return;
  int num_cell = range.x * range.y * range.z;
  glm::ivec3 cell_index;
  float result = 0.0f;
  for (int dx = 0; dx < 2; dx++) {
    cell_index.x = cell_index_2h.x * 2 + dx;
    if (cell_index.x >= range.x)
      break;
    for (int dy = 0; dy < 2; dy++) {
      cell_index.y = cell_index_2h.y * 2 + dy;
      if (cell_index.y >= range.y)
        continue;
      for (int dz = 0; dz < 2; dz++) {
        cell_index.z = cell_index_2h.z * 2 + dz;
        if (cell_index.z >= range.z)
          continue;
        int id = RANGE_INDEX(cell_index, range);
        float ele = 0.0f;
        CellCoe coe = cell_coe[id];
        ele += coe.local * x[id];
        if (id > 1)
          ele += coe.z[0] * x[id - 1];
        if (id > range.z)
          ele += coe.y[0] * x[id - range.z];
        if (id > range.z * range.y)
          ele += coe.x[0] * x[id - range.z * range.y];
        if (id + 1 < num_cell)
          ele += coe.z[1] * x[id + 1];
        if (id + range.z < num_cell)
          ele += coe.y[1] * x[id + range.z];
        if (id + range.z * range.y < num_cell)
          ele += coe.x[1] * x[id + range.z * range.y];
        result += (b[id] - ele) * 0.125f;
      }
    }
  }
  r_2h[id_2h] = result;
}

__global__ void CalcResidualKernel(const CellCoe *cell_coe,
                                   const float *x,
                                   const float *b,
                                   float *r,
                                   glm::ivec3 range) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  glm::ivec3 cell_index_2h{id / (range.y * range.z), (id / range.z) % range.y,
                           id % range.z};
  int num_cell = range.x * range.y * range.z;
  if (id >= num_cell)
    return;
  float ele = 0.0f;
  CellCoe coe = cell_coe[id];
  ele += coe.local * x[id];
  if (id > 1)
    ele += coe.z[0] * x[id - 1];
  if (id > range.z)
    ele += coe.y[0] * x[id - range.z];
  if (id > range.z * range.y)
    ele += coe.x[0] * x[id - range.z * range.y];
  if (id + 1 < num_cell)
    ele += coe.z[1] * x[id + 1];
  if (id + range.z < num_cell)
    ele += coe.y[1] * x[id + range.z];
  if (id + range.z * range.y < num_cell)
    ele += coe.x[1] * x[id + range.z * range.y];
  r[id] = (b[id] - ele);
}

__global__ void MultiGridJacobiIterationKernel(const CellCoe *cell_coe,
                                               const float *divergence,
                                               const float *pressure,
                                               float *pressure_new,
                                               glm::ivec3 range) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_cell = range.x * range.y * range.z;
  if (id < num_cell) {
    float res = 0.0f;
    CellCoe coe = cell_coe[id];
    if (coe.local > 1e-6f) {
      if (id > 1)
        res += coe.z[0] * pressure[id - 1];
      if (id > range.z)
        res += coe.y[0] * pressure[id - range.z];
      if (id > range.z * range.y)
        res += coe.x[0] * pressure[id - range.z * range.y];
      if (id + 1 < num_cell)
        res += coe.z[1] * pressure[id + 1];
      if (id + range.z < num_cell)
        res += coe.y[1] * pressure[id + range.z];
      if (id + range.z * range.y < num_cell)
        res += coe.x[1] * pressure[id + range.z * range.y];
      res = (divergence[id] - res) / coe.local;
    }
    pressure_new[id] = res;
  }
}

__global__ void ConstructHalfKernel(Grid<CellCoe>::DevRef matrix,
                                    Grid<CellCoe>::DevRef matrix_2h) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= matrix_2h.Size())
    return;
  auto range_2h = matrix_2h.Range();
  glm::ivec3 cell_index_2h{id / (range_2h.y * range_2h.z),
                           (id / range_2h.z) % range_2h.y, id % range_2h.z};
  CellCoe cell_2h{};
  glm::ivec3 cell_index;
  auto range = matrix.Range();
  for (int dx = 0; dx < 2; dx++) {
    cell_index.x = cell_index_2h.x * 2 + dx;
    if (cell_index.x >= range.x)
      break;
    for (int dy = 0; dy < 2; dy++) {
      cell_index.y = cell_index_2h.y * 2 + dy;
      if (cell_index.y >= range.y)
        continue;
      for (int dz = 0; dz < 2; dz++) {
        cell_index.z = cell_index_2h.z * 2 + dz;
        if (cell_index.z >= range.z)
          continue;
        CellCoe cell = matrix(cell_index);
        cell_2h.local += cell.local * 0.125f;

        cell_2h.x[dx] += cell.x[dx] * 0.125f;
        cell_2h.local += cell.x[dx ^ 1] * 0.125f;

        cell_2h.y[dy] += cell.y[dy] * 0.125f;
        cell_2h.local += cell.y[dy ^ 1] * 0.125f;

        cell_2h.z[dz] += cell.z[dz] * 0.125f;
        cell_2h.local += cell.z[dz ^ 1] * 0.125f;
      }
    }
  }
  matrix_2h(cell_index_2h) = cell_2h;
}

__global__ void ApplyErrorKernel(Grid<float>::DevRef e, Grid<float>::DevRef x) {
  int id_2h = blockIdx.x * blockDim.x + threadIdx.x;
  auto range_2h = e.Range();
  glm::ivec3 cell_index_2h{id_2h / (range_2h.y * range_2h.z),
                           (id_2h / range_2h.z) % range_2h.y,
                           id_2h % range_2h.z};
  int num_cell_2h = range_2h.x * range_2h.y * range_2h.z;
  if (id_2h >= num_cell_2h)
    return;
  auto range = x.Range();
  int num_cell = range.x * range.y * range.z;
  glm::ivec3 cell_index;
  float e_value = e(cell_index_2h);
  for (int dx = 0; dx < 2; dx++) {
    cell_index.x = cell_index_2h.x * 2 + dx;
    if (cell_index.x >= range.x)
      break;
    for (int dy = 0; dy < 2; dy++) {
      cell_index.y = cell_index_2h.y * 2 + dy;
      if (cell_index.y >= range.y)
        continue;
      for (int dz = 0; dz < 2; dz++) {
        cell_index.z = cell_index_2h.z * 2 + dz;
        if (cell_index.z >= range.z)
          continue;
        x(cell_index) += e_value;
      }
    }
  }
}

struct CoePlus {
  __device__ CellCoe operator()(const CellCoe &a, const CellCoe &b) {
    return {a.local + b.local,
            {a.x[0] + b.x[0], a.x[1] + b.x[1]},
            {a.y[0] + b.y[0], a.y[1] + b.y[1]},
            {a.z[0] + b.z[0], a.z[1] + b.z[1]}};
  }
};

__global__ void ApplyMatrixKernel(const CellCoe *matrix,
                                  const float *x,
                                  float *b,
                                  glm::ivec3 range) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_cell = range.x * range.y * range.z;
  if (id < num_cell) {
    float div = 0.0f;
    CellCoe coe = matrix[id];
    div += coe.local * x[id];
    if (id > 1)
      div += coe.z[0] * x[id - 1];
    if (id > range.z)
      div += coe.y[0] * x[id - range.z];
    if (id > range.z * range.y)
      div += coe.x[0] * x[id - range.z * range.y];
    if (id + 1 < num_cell)
      div += coe.z[1] * x[id + 1];
    if (id + range.z < num_cell)
      div += coe.y[1] * x[id + range.z];
    if (id + range.z * range.y < num_cell)
      div += coe.x[1] * x[id + range.z * range.y];
    b[id] = div;
  }
}

void MultiGrid(Grid<CellCoe> &matrix, Grid<float> &b, Grid<float> &x) {
  auto matrix_range = matrix.Range();
#ifdef MULTIGRID_VERBOSE
  thrust::device_vector<float> r(x.Size());
  CalcResidualKernel<<<CALL_GRID(x.Size())>>>(
      matrix.Vector().data().get(), x.Vector().data().get(),
      b.Vector().data().get(), r.data().get(), matrix.Range());
  printf("Residual Before Iter: %f %f\n",
         thrust::transform_reduce(r.begin(), r.end(), SquareOp<float>(), 0.0f,
                                  thrust::plus<float>()),
         thrust::transform_reduce(x.Vector().begin(), x.Vector().end(),
                                  SquareOp<float>(), 0.0f,
                                  thrust::maximum<float>()));
#endif

  JacobiMethod(
      [&](const thrust::device_vector<float> &b,
          const thrust::device_vector<float> &x,
          thrust::device_vector<float> &x_new) {
        MultiGridJacobiIterationKernel<<<CALL_GRID(x.size())>>>(
            matrix.Vector().data().get(), b.data().get(), x.data().get(),
            x_new.data().get(), matrix_range);
      },
      b.Vector(), x.Vector(), 100);

  if (!(matrix_range.x <= 2 && matrix_range.y <= 2 && matrix_range.z <= 2)) {
    Grid<CellCoe> matrix_2h((matrix.Range() + 1) >> 1);
    Grid<float> r_2h((matrix.Range() + 1) >> 1);
    Grid<float> e_2h((matrix.Range() + 1) >> 1);
    ConstructHalfKernel<<<CALL_GRID(matrix_2h.Size())>>>(matrix, matrix_2h);
    CalcResidualKernel<<<CALL_GRID(r_2h.Size())>>>(
        matrix.Vector().data().get(), x.Vector().data().get(),
        b.Vector().data().get(), r_2h.Vector().data().get(), matrix.Range(),
        matrix_2h.Range());
#ifdef MULTIGRID_VERBOSE
    printf("Residual: %f\n",
           thrust::transform_reduce(r_2h.Vector().begin(), r_2h.Vector().end(),
                                    SquareOp<float>(), 0.0f,
                                    thrust::plus<float>()));
#endif

    MultiGrid(matrix_2h, r_2h, e_2h);
    ApplyErrorKernel<<<CALL_GRID(e_2h.Size())>>>(e_2h, x);
  }

#ifdef MULTIGRID_VERBOSE
  CalcResidualKernel<<<CALL_GRID(x.Size())>>>(
      matrix.Vector().data().get(), x.Vector().data().get(),
      b.Vector().data().get(), r.data().get(), matrix.Range());
  printf("MultiGrid Residual After Recurse: %f %f\n",
         thrust::transform_reduce(r.begin(), r.end(), SquareOp<float>(), 0.0f,
                                  thrust::plus<float>()),
         thrust::transform_reduce(x.Vector().begin(), x.Vector().end(),
                                  SquareOp<float>(), 0.0f,
                                  thrust::maximum<float>()));
#endif

  JacobiMethod(
      [&](const thrust::device_vector<float> &b,
          const thrust::device_vector<float> &x,
          thrust::device_vector<float> &x_new) {
        MultiGridJacobiIterationKernel<<<CALL_GRID(x.size())>>>(
            matrix.Vector().data().get(), b.data().get(), x.data().get(),
            x_new.data().get(), matrix_range);
      },
      b.Vector(), x.Vector(), 100);

#ifdef MULTIGRID_VERBOSE
  CalcResidualKernel<<<CALL_GRID(x.Size())>>>(
      matrix.Vector().data().get(), x.Vector().data().get(),
      b.Vector().data().get(), r.data().get(), matrix.Range());
  printf("MultiGrid Final Residual: %f %f\n",
         thrust::transform_reduce(r.begin(), r.end(), SquareOp<float>(), 0.0f,
                                  thrust::plus<float>()),
         thrust::transform_reduce(x.Vector().begin(), x.Vector().end(),
                                  SquareOp<float>(), 0.0f,
                                  thrust::maximum<float>()));
#endif
}
