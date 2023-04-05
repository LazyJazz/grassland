#include "curand.h"
#include "curand_kernel.h"
#include "fluid_app.cuh"
#include "glm/gtc/matrix_transform.hpp"
#include "grid_dev.cuh"
#include "params.h"
#include "thrust/device_vector.h"

#define BLOCK_SIZE 256
#define LAUNCH_SIZE(x) ((x + (BLOCK_SIZE - 1)) / BLOCK_SIZE), BLOCK_SIZE


__host__ __device__ bool InsideFreeVolume(glm::vec3 position);

FluidApp::FluidApp(const FluidAppSettings &settings) {
  settings_ = settings;
  grassland::vulkan::framework::CoreSettings core_settings{};
  core_settings.window_width = 2048;
  core_settings.window_height = 2048;
  core_settings.window_title = "Fluid Demo";
  core_ = std::make_unique<grassland::vulkan::framework::Core>(core_settings);
}

void FluidApp::Run() {
  OnInit();
  while (!glfwWindowShouldClose(core_->GetWindow())) {
    OnLoop();
    glfwPollEvents();
  }
  core_->GetDevice()->WaitIdle();
  OnClose();
}

void FluidApp::OnInit() {
  camera_object_buffer_ = std::make_unique<
      grassland::vulkan::framework::DynamicBuffer<CameraObject>>(core_.get(),
                                                                 1);

  frame_image_ = std::make_unique<grassland::vulkan::framework::TextureImage>(
      core_.get(), core_->GetFramebufferWidth(), core_->GetFramebufferHeight(),
      VK_FORMAT_B8G8R8A8_UNORM);
  render_info_buffer_ =
      std::make_unique<grassland::vulkan::framework::DynamicBuffer<RenderInfo>>(
          core_.get(), 1);
  core_->SetFrameSizeCallback([this](int width, int height) {
    frame_image_->Resize(width, height);
    if (render_node_) {
      render_node_->BuildRenderNode(width, height);
    }
  });
  render_node_ =
      std::make_unique<grassland::vulkan::framework::RenderNode>(core_.get());
  render_node_->AddColorAttachment(frame_image_.get());
  render_node_->AddDepthAttachment();
  render_node_->VertexInput(
      {VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT});
  render_node_->AddShader("render.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
  render_node_->AddShader("render.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddUniformBinding(camera_object_buffer_.get(),
                                  VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddBufferBinding(render_info_buffer_.get(),
                                 VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->BuildRenderNode();

  camera_ = Camera(glm::vec3{-20.0f, SIZE_Y * 0.5f, 30.0f},
                   glm::vec3{SIZE_X * 0.5f, SIZE_Y * 0.5f, SIZE_Z * 0.5f});
  RegisterSphere();
  RegisterCylinder();

  InitParticles();
  u_field_[0] = Grid<float>(GRID_SIZE_X + 1, GRID_SIZE_Y, GRID_SIZE_Z);
  v_field_[0] = Grid<float>(GRID_SIZE_X, GRID_SIZE_Y + 1, GRID_SIZE_Z);
  w_field_[0] = Grid<float>(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z + 1);
  u_weight_field_[0] = Grid<float>(GRID_SIZE_X + 1, GRID_SIZE_Y, GRID_SIZE_Z);
  v_weight_field_[0] = Grid<float>(GRID_SIZE_X, GRID_SIZE_Y + 1, GRID_SIZE_Z);
  w_weight_field_[0] = Grid<float>(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z + 1);
  u_field_[1] = Grid<float>(GRID_SIZE_X + 1, GRID_SIZE_Y, GRID_SIZE_Z);
  v_field_[1] = Grid<float>(GRID_SIZE_X, GRID_SIZE_Y + 1, GRID_SIZE_Z);
  w_field_[1] = Grid<float>(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z + 1);
  u_weight_field_[1] = Grid<float>(GRID_SIZE_X + 1, GRID_SIZE_Y, GRID_SIZE_Z);
  v_weight_field_[1] = Grid<float>(GRID_SIZE_X, GRID_SIZE_Y + 1, GRID_SIZE_Z);
  w_weight_field_[1] = Grid<float>(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z + 1);
  level_set_ = Grid<float>(GRID_SIZE_X + 1, GRID_SIZE_Y + 1, GRID_SIZE_Z + 1);
}

void FluidApp::OnLoop() {
  OnUpdate();
  OnRender();
}

void FluidApp::OnClose() {
}

void FluidApp::OnUpdate() {
  UpdatePhysicalSystem();
  DrawObjects();
  UpdateCamera();
  UpdateDynamicInfos();
}

void FluidApp::DrawObjects() {
  render_infos_.clear();
  render_objects_.clear();
  DrawLine({0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 0.03f,
           {1.0f, 0.0f, 0.0f, 1.0f});
  DrawLine({0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.03f,
           {0.0f, 1.0f, 0.0f, 1.0f});
  DrawLine({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.03f,
           {0.0f, 0.0f, 1.0f, 1.0f});
  DrawSphere({0.0f, 0.0f, 0.0f}, 0.1f, {1.0f, 1.0f, 1.0f, 0.0f});

  for (auto &particle : particles_) {
    if (InsideFreeVolume(particle.position)) {
      DrawSphere(particle.position, RENDER_SIZE,
                 (particle.type == TYPE_AIR) ? glm::vec4{1.0f, 0.5f, 0.5f, 1.0f}
                                             : glm::vec4{0.5f, 0.5f, 1.0f, 1.0f});
    }
  }
}

void FluidApp::UpdateDynamicInfos() {
  (*camera_object_buffer_)[0] =
      camera_.ComposeMatrix(glm::radians(60.0f),
                            (float)core_->GetFramebufferWidth() /
                                (float)core_->GetFramebufferHeight(),
                            0.1f, 100.0f);
  if (render_info_buffer_->Size() < render_infos_.size() ||
      render_infos_.size() < render_info_buffer_->Size() / 2) {
    render_info_buffer_->Resize(render_infos_.size());
    render_node_->UpdateDescriptorSetBinding(1);
  }
  for (int i = 0; i < render_infos_.size(); i++) {
    (*render_info_buffer_)[i] = render_infos_[i];
  }
}

void FluidApp::OnRender() {
  core_->BeginCommandRecord();
  frame_image_->ClearColor({0.0f, 0.0f, 0.0f, 1.0f});
  render_node_->GetDepthAttachmentImage()->ClearDepth({1.0f, 0});
  // render_node_->Draw(vertex_buffer_.get(), index_buffer_.get(),
  // index_buffer_->Size(), 0, 1);
  render_node_->BeginDraw();
  for (int i = 0; i < render_objects_.size(); i++) {
    int last_i = i;
    while (last_i + 1 < render_objects_.size() &&
           render_objects_[last_i + 1] == render_objects_[i]) {
      last_i++;
    }
    render_node_->DrawDirect(object_models_[render_objects_[i]].first.get(),
                             object_models_[render_objects_[i]].second.get(),
                             object_models_[render_objects_[i]].second->Size(),
                             i, last_i - i + 1);
    i = last_i;
  }
  render_node_->EndDraw();
  core_->Output(frame_image_.get());
  core_->EndCommandRecordAndSubmit();
}

void FluidApp::UpdateCamera() {
  glm::vec3 offset{};
  glm::vec3 rotation{};

  const float move_speed = 3.0f;
  static auto last_ts = std::chrono::steady_clock::now();
  auto duration = (std::chrono::steady_clock::now() - last_ts) /
                  std::chrono::milliseconds(1);
  auto sec = (float)duration * 0.001f;

  last_ts += duration * std::chrono::milliseconds(1);
  if (glfwGetKey(core_->GetWindow(), GLFW_KEY_W) == GLFW_PRESS) {
    offset.z -= move_speed * sec;
  }
  if (glfwGetKey(core_->GetWindow(), GLFW_KEY_S) == GLFW_PRESS) {
    offset.z += move_speed * sec;
  }
  if (glfwGetKey(core_->GetWindow(), GLFW_KEY_A) == GLFW_PRESS) {
    offset.x -= move_speed * sec;
  }
  if (glfwGetKey(core_->GetWindow(), GLFW_KEY_D) == GLFW_PRESS) {
    offset.x += move_speed * sec;
  }
  if (glfwGetKey(core_->GetWindow(), GLFW_KEY_SPACE) == GLFW_PRESS) {
    offset.y += move_speed * sec;
  }
  if (glfwGetKey(core_->GetWindow(), GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
    offset.y -= move_speed * sec;
  }

  double xpos, ypos;
  glfwGetCursorPos(core_->GetWindow(), &xpos, &ypos);
  auto cur_cursor = glm::vec2{xpos, ypos};
  static auto last_cursor = cur_cursor;
  auto cursor_diff = cur_cursor - last_cursor;
  last_cursor = cur_cursor;
  auto rotation_scale = 1.0f / core_->GetWindowWidth();
  if (glfwGetMouseButton(core_->GetWindow(), GLFW_MOUSE_BUTTON_LEFT) ==
      GLFW_PRESS) {
    rotation.x -= cursor_diff.y * rotation_scale;
    rotation.y -= cursor_diff.x * rotation_scale;
  }

  camera_.MoveLocal(offset, rotation);
}

int FluidApp::RegisterModel(const std::vector<Vertex> &vertices,
                            const std::vector<uint32_t> &indices) {
  auto res = object_models_.size();
  object_models_.emplace_back(
      std::make_unique<grassland::vulkan::framework::StaticBuffer<Vertex>>(
          core_.get(), vertices.size()),
      std::make_unique<grassland::vulkan::framework::StaticBuffer<uint32_t>>(
          core_.get(), indices.size()));
  object_models_[res].first->Upload(vertices.data());
  object_models_[res].second->Upload(indices.data());
  return res;
}

void FluidApp::RegisterSphere() {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  auto pi = glm::radians(180.0f);
  const auto precision = 20;
  const auto inv_precision = 1.0f / float(precision);
  std::vector<glm::vec2> circle;
  for (int i = 0; i <= precision * 2; i++) {
    float omega = inv_precision * float(i) * pi;
    circle.emplace_back(-std::sin(omega), -std::cos(omega));
  }
  for (int i = 0; i <= precision; i++) {
    float theta = inv_precision * float(i) * pi;
    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);
    int i_1 = i - 1;
    for (int j = 0; j <= 2 * precision; j++) {
      auto normal = glm::vec3{circle[j].x * sin_theta, cos_theta,
                              circle[j].y * sin_theta};
      vertices.push_back(Vertex{normal, normal});
      if (i) {
        int j1 = j + 1;
        if (j == 2 * precision) {
          j1 = 0;
        }
        indices.push_back(i * (2 * precision + 1) + j1);
        indices.push_back(i_1 * (2 * precision + 1) + j);
        indices.push_back(i * (2 * precision + 1) + j);
        indices.push_back(i * (2 * precision + 1) + j1);
        indices.push_back(i_1 * (2 * precision + 1) + j1);
        indices.push_back(i_1 * (2 * precision + 1) + j);
      }
    }
  }
  sphere_model_id_ = RegisterModel(vertices, indices);
}

void FluidApp::RegisterCylinder() {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  auto pi = glm::radians(180.0f);
  const auto precision = 60;
  const auto inv_precision = 1.0f / float(precision);
  for (int i = 0; i < precision; i++) {
    float theta = inv_precision * float(i) * pi * 2.0f;
    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);
    vertices.push_back(
        {{sin_theta, 1.0f, cos_theta}, {sin_theta, 0.0f, cos_theta}});
    vertices.push_back(
        {{sin_theta, -1.0f, cos_theta}, {sin_theta, 0.0f, cos_theta}});
    int j = (i + 1) % precision;
    indices.push_back(i * 2);
    indices.push_back(i * 2 + 1);
    indices.push_back(j * 2 + 1);
    indices.push_back(j * 2);
    indices.push_back(j * 2 + 1);
    indices.push_back(i * 2 + 0);
  }
  cylinder_model_id_ = RegisterModel(vertices, indices);
}

void FluidApp::DrawObject(int model_id, glm::mat4 model, glm::vec4 color) {
  render_objects_.push_back(model_id);
  render_infos_.push_back({model, color});
}

void FluidApp::DrawLine(const glm::vec3 &v0,
                        const glm::vec3 &v1,
                        float thickness,
                        const glm::vec4 &color) {
  auto y_axis = (v1 - v0) * 0.5f;
  if (glm::length(y_axis) < 1e-4f)
    return;
  auto offset = (v1 + v0) * 0.5f;
  auto x_axis = glm::cross(y_axis, glm::vec3{1.0f, 0.0f, 0.0f});
  if (glm::length(x_axis) < 1e-2f) {
    x_axis = glm::cross(y_axis, glm::vec3{0.0f, 0.0f, 1.0f});
  }
  x_axis = glm::normalize(x_axis);
  auto z_axis = glm::normalize(glm::cross(x_axis, y_axis));
  x_axis *= thickness;
  z_axis *= thickness;
  DrawObject(cylinder_model_id_,
             glm::mat4{glm::vec4{x_axis, 0.0f}, glm::vec4{y_axis, 0.0f},
                       glm::vec4{z_axis, 0.0f}, glm::vec4{offset, 1.0f}},
             color);
}

void FluidApp::DrawSphere(const glm::vec3 &origin,
                          float radius,
                          const glm::vec4 &color) {
  DrawObject(
      sphere_model_id_,
      glm::mat4{glm::vec4{radius, 0.0f, 0.0f, 0.0f},
                glm::vec4{0.0f, radius, 0.0f, 0.0f},
                glm::vec4{0.0f, 0.0f, radius, 0.0f}, glm::vec4{origin, 1.0f}},
      color);
}

__device__ glm::vec3 RandomOnSphere(curandState_t *state) {
  auto y = curand_uniform(state) * 2.0f - 1.0f;
  auto xz = sqrt(1 - y * y);
  auto theta = glm::pi<float>() * 2.0f * curand_uniform(state);
  return glm::vec3{sin(theta) * xz, y, cos(theta) * xz};
}

__device__ glm::vec3 RandomInSphere(curandState_t *state) {
  return RandomOnSphere(state) * pow(curand_uniform(state), 1.0f / 3.0f);
}

__device__ float KernelFunction(glm::vec3 v) {
  auto len = glm::length(v);
  if (len < 0.5f) {
    return 0.75f - len * len;
  } else if (len < 1.5f) {
    return 0.5f * (1.5f - len) * (1.5f - len);
  }
  return 0.0f;
}

__global__ void ApplyGravityKernel(Particle *particles,
                                   float delta_t,
                                   int num_particle) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= num_particle)
    return;
  particles[id].velocity += delta_t * GRAVITY;
}

__global__ void AdvectKernel(Particle *particles,
                             float delta_t,
                             int num_particle) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= num_particle)
    return;
  Particle particle = particles[id];
  particle.position += particle.velocity * delta_t;
  particles[id] = particle;
}

__global__ void Particle2GridKernel(Particle *particles,
                                    GridDev<float> field0,
                                    GridDev<float> field1,
                                    GridDev<float> weight_field0,
                                    GridDev<float> weight_field1,
                                    int num_particle,
                                    int dim) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= num_particle)
    return;
  GridDev<float> field[2] = {field0, field1};
  GridDev<float> weight_field[2] = {weight_field0, weight_field1};
  Particle particle = particles[id];
  glm::vec3 offset{0.5f, 0.5f, 0.5f};
  offset[dim] = 0.0f;
  glm::vec3 grid_pos = particle.position * (1.0f / DELTA_X) - offset;
  glm::ivec3 nearest_grid_pos = grid_pos + 0.5f;
  glm::ivec3 index = nearest_grid_pos;
  for (int dx = -1; dx <= 1; dx++) {
    index.x = nearest_grid_pos.x + dx;
    if (index.x < 0 || index.x >= field[0].size_x_)
      continue;
    for (int dy = -1; dy <= 1; dy++) {
      index.y = nearest_grid_pos.y + dy;
      if (index.y < 0 || index.y >= field[0].size_y_)
        continue;
      for (int dz = -1; dz <= 1; dz++) {
        index.z = nearest_grid_pos.z + dz;
        if (index.z < 0 || index.z >= field[0].size_z_)
          continue;
        float weight = KernelFunction(grid_pos - glm::vec3{index});
        atomicAdd(&field[particle.type](index),
                  weight * particle.velocity[dim]);
        atomicAdd(&weight_field[particle.type](index), weight);
      }
    }
  }
}

__global__ void NormalizeByWeightKernel(GridDev<float> field,
                                        GridDev<float> weight_field) {
  int size = field.size_x_ * field.size_y_ * field.size_z_;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size)
    return;
  float weight = weight_field.buffer_[id];
  if (weight > 1e-6f) {
    field.buffer_[id] /= weight;
  }
}

__global__ void BuildLevelSetKernel(GridDev<float> level_set,
                                    Particle *particle,
                                    int num_particle) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = id / (level_set.size_y_ * level_set.size_z_);
  int idy = (id / level_set.size_z_) % level_set.size_y_;
  int idz = id % level_set.size_z_;
  glm::vec3 grid_pos = glm::vec3{idx, idy, idz} * DELTA_X;
  if (id >= level_set.size_x_ * level_set.size_y_ * level_set.size_z_)
    return;
  float nearest_air = 1e30f;
  float nearest_liq = 1e30f;
  for (int i = 0; i < num_particle; i++) {
    glm::vec3 position = particle[i].position;
    int type = particle[i].type;
    if (type == TYPE_AIR) {
      nearest_air = min(nearest_air, glm::length(grid_pos - position));
    } else {
      nearest_liq = min(nearest_liq, glm::length(grid_pos - position));
    };
  }
  level_set.buffer_[id] = nearest_air - nearest_liq;
}

__host__ __device__ bool InsideFreeVolume(glm::vec3 position) {
  return (glm::length(position - glm::vec3{SIZE_X * 0.5f, SIZE_Y * 0.3f,
                                           SIZE_Z * 0.5f}) < SIZE_X * 0.4f) ||
         (glm::length(position - glm::vec3{SIZE_X * 0.5f, SIZE_Y * 0.7f,
                                           SIZE_Z * 0.5f}) < SIZE_X * 0.4f) ||
         ((position.y > SIZE_Y * 0.3f && position.y < SIZE_Y * 0.7f) &&
          glm::length(glm::vec2{position.x, position.z} -
                      glm::vec2{SIZE_X * 0.5f, SIZE_Z * 0.5f}) < SIZE_X * 0.2f);
}

__global__ void CalcBorderScaleKernel(GridDev<float> air_weight_field,
                                      GridDev<float> liq_weight_field,
                                      GridDev<float> level_set,
                                      int dim) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = id / (air_weight_field.size_y_ * air_weight_field.size_z_);
  int idy = (id / air_weight_field.size_z_) % air_weight_field.size_y_;
  int idz = id % air_weight_field.size_z_;
  if (id >= air_weight_field.size_x_ * air_weight_field.size_y_ *
                air_weight_field.size_z_)
    return;

  int precision = 8;
  float inv_precision = 1.0f / precision;
  float air_weight = 0.0f;
  float liq_weight = 0.0f;
  glm::ivec3 i_pos(idx, idy, idz);
  glm::ivec3 i_axis{0};
  glm::ivec3 j_axis{0};
  i_axis[(dim + 1) % 3] = 1;
  j_axis[(dim + 2) % 3] = 1;
  float v00 = level_set(i_pos);
  float v01 = level_set(i_pos + j_axis);
  float v10 = level_set(i_pos + i_axis);
  float v11 = level_set(i_pos + i_axis + j_axis);
  for (int i = 0; i < precision; i++) {
    float fi = (i + 0.5f) * inv_precision;
    float i_1 = 1.0f - fi;
    for (int j = 0; j < precision; j++) {
      float fj = (j + 0.5f) * inv_precision;
      float j_1 = 1.0f - fj;
      glm::vec3 sample_pos{idx, idy, idz};
      sample_pos += fi * glm::vec3{i_axis} + fj * glm::vec3{j_axis};
      sample_pos *= DELTA_X;
      if (InsideFreeVolume(sample_pos)) {
        float interpolated =
            (v00 * j_1 + v01 * fj) * i_1 + (v10 * j_1 + v11 * fj) * fi;
        if (interpolated < 0.0f) {
          air_weight += inv_precision * inv_precision;
        } else {
          liq_weight += inv_precision * inv_precision;
        }
      }
    }
  }
  air_weight_field.buffer_[id] = air_weight;
  liq_weight_field.buffer_[id] = liq_weight;
}

__global__ void PrepareReverseDivergence(float *divergence,
                                         GridDev<float> u_field0,
                                         GridDev<float> u_field1,
                                         GridDev<float> v_field0,
                                         GridDev<float> v_field1,
                                         GridDev<float> w_field0,
                                         GridDev<float> w_field1,
                                         GridDev<float> u_weight_field0,
                                         GridDev<float> u_weight_field1,
                                         GridDev<float> v_weight_field0,
                                         GridDev<float> v_weight_field1,
                                         GridDev<float> w_weight_field0,
                                         GridDev<float> w_weight_field1) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = id / (GRID_SIZE_Y * GRID_SIZE_Z);
  int idy = (id / GRID_SIZE_Z) % GRID_SIZE_Y;
  int idz = id % GRID_SIZE_Z;
  if (id >= GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z)
    return;
  float diver = 0.0f;
  diver += u_field0(idx, idy, idz) * u_weight_field0(idx, idy, idz) -
           u_field0(idx + 1, idy, idz) * u_weight_field0(idx + 1, idy, idz);
  diver += u_field1(idx, idy, idz) * u_weight_field1(idx, idy, idz) -
           u_field1(idx + 1, idy, idz) * u_weight_field1(idx + 1, idy, idz);
  diver += v_field0(idx, idy, idz) * v_weight_field0(idx, idy, idz) -
           v_field0(idx, idy + 1, idz) * v_weight_field0(idx, idy + 1, idz);
  diver += v_field1(idx, idy, idz) * v_weight_field1(idx, idy, idz) -
           v_field1(idx, idy + 1, idz) * v_weight_field1(idx, idy + 1, idz);
  diver += w_field0(idx, idy, idz) * w_weight_field0(idx, idy, idz) -
           w_field0(idx, idy, idz + 1) * w_weight_field0(idx, idy, idz + 1);
  diver += w_field1(idx, idy, idz) * w_weight_field1(idx, idy, idz) -
           w_field1(idx, idy, idz + 1) * w_weight_field1(idx, idy, idz + 1);
  divergence[id] = diver;
}

__global__ void CalcImpactToVelFieldKernel(GridDev<float> field,
                                           float *pressure,
                                           float rho,
                                           float delta_t,
                                           int dim) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = id / (field.size_y_ * field.size_z_);
  int idy = (id / field.size_z_) % field.size_y_;
  int idz = id % field.size_z_;
  if (id >= field.size_x_ * field.size_y_ * field.size_z_)
    return;
  glm::ivec3 index{idx, idy, idz};
  glm::ivec3 grid_size{GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z};
  if (index[dim] == 0 || index[dim] == grid_size[dim]) {
    return;
  }
  float delta_speed = -pressure[GRID_POINT_ID(index.x, index.y, index.z)];
  index[dim]--;
  delta_speed += pressure[GRID_POINT_ID(index.x, index.y, index.z)];
  delta_speed *= delta_t / (DELTA_X * rho);
  field(idx, idy, idz) = delta_speed;
}

__global__ void Grid2ParticleKernel(Particle *particles,
                                    GridDev<float> field0,
                                    GridDev<float> field1,
                                    GridDev<float> weight_field0,
                                    GridDev<float> weight_field1,
                                    int num_particle,
                                    int dim) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= num_particle)
    return;
  GridDev<float> field[2] = {field0, field1};
  GridDev<float> weight_field[2] = {weight_field0, weight_field1};
  Particle particle = particles[id];
  glm::vec3 offset{0.5f, 0.5f, 0.5f};
  offset[dim] = 0.0f;
  glm::vec3 grid_pos = particle.position * (1.0f / DELTA_X) - offset;
  glm::ivec3 nearest_grid_pos = grid_pos + 0.5f;
  glm::ivec3 index = nearest_grid_pos;
  float accum_weight = 0.0f;
  float accum_vel = 0.0f;
  for (int dx = -1; dx <= 1; dx++) {
    index.x = nearest_grid_pos.x + dx;
    if (index.x < 0 || index.x >= field[0].size_x_)
      continue;
    for (int dy = -1; dy <= 1; dy++) {
      index.y = nearest_grid_pos.y + dy;
      if (index.y < 0 || index.y >= field[0].size_y_)
        continue;
      for (int dz = -1; dz <= 1; dz++) {
        index.z = nearest_grid_pos.z + dz;
        if (index.z < 0 || index.z >= field[0].size_z_)
          continue;
        if (!weight_field[particle.type](index))
          continue;
        float weight = KernelFunction(grid_pos - glm::vec3{index});
        accum_weight += weight;
        accum_vel += weight * field[particle.type](index);
      }
    }
  }
  if (accum_weight > 1e-4f) {
    particle.velocity[dim] += accum_vel / accum_weight;
  }
  particles[id] = particle;
}
void FluidApp::UpdatePhysicalSystem() {
  thrust::device_vector<Particle> dev_particles = particles_;
  ApplyGravityKernel<<<LAUNCH_SIZE(particles_.size())>>>(
      dev_particles.data().get(), delta_t_, particles_.size());
  for (int i = 0; i < 2; i++) {
    u_field_[i].ClearData();
    v_field_[i].ClearData();
    w_field_[i].ClearData();
    u_weight_field_[i].ClearData();
    v_weight_field_[i].ClearData();
    w_weight_field_[i].ClearData();
  }
  Particle2GridKernel<<<LAUNCH_SIZE(particles_.size())>>>(
      dev_particles.data().get(), u_field_[0], u_field_[1], u_weight_field_[0],
      u_weight_field_[1], particles_.size(), 0);
  NormalizeByWeightKernel<<<LAUNCH_SIZE((GRID_SIZE_X + 1) * GRID_SIZE_Y *
                                        GRID_SIZE_Z)>>>(u_field_[0],
                                                        u_weight_field_[0]);
  NormalizeByWeightKernel<<<LAUNCH_SIZE((GRID_SIZE_X + 1) * GRID_SIZE_Y *
                                        GRID_SIZE_Z)>>>(u_field_[1],
                                                        u_weight_field_[1]);
  Particle2GridKernel<<<LAUNCH_SIZE(particles_.size())>>>(
      dev_particles.data().get(), v_field_[0], v_field_[1], v_weight_field_[0],
      v_weight_field_[1], particles_.size(), 1);
  NormalizeByWeightKernel<<<LAUNCH_SIZE(GRID_SIZE_X * (GRID_SIZE_Y + 1) *
                                        GRID_SIZE_Z)>>>(v_field_[0],
                                                        v_weight_field_[0]);
  NormalizeByWeightKernel<<<LAUNCH_SIZE(GRID_SIZE_X * (GRID_SIZE_Y + 1) *
                                        GRID_SIZE_Z)>>>(v_field_[1],
                                                        v_weight_field_[1]);
  Particle2GridKernel<<<LAUNCH_SIZE(particles_.size())>>>(
      dev_particles.data().get(), w_field_[0], w_field_[1], w_weight_field_[0],
      w_weight_field_[1], particles_.size(), 2);
  NormalizeByWeightKernel<<<LAUNCH_SIZE(GRID_SIZE_X * GRID_SIZE_Y *
                                        (GRID_SIZE_Z + 1))>>>(
      w_field_[0], w_weight_field_[0]);
  NormalizeByWeightKernel<<<LAUNCH_SIZE(GRID_SIZE_X * GRID_SIZE_Y *
                                        (GRID_SIZE_Z + 1))>>>(
      w_field_[1], w_weight_field_[1]);
  //  v_field_[0].PlotCSV(GRID_SIZE_X / 2);
  //  v_field_[1].PlotCSV(GRID_SIZE_X / 2);
  //  v_weight_field_[0].PlotCSV(GRID_SIZE_X / 2);
  //  v_weight_field_[1].PlotCSV(GRID_SIZE_X / 2);
  BuildLevelSetKernel<<<LAUNCH_SIZE(level_set_.Size())>>>(
      level_set_, dev_particles.data().get(), particles_.size());
  CalcBorderScaleKernel<<<LAUNCH_SIZE(u_weight_field_[0].Size())>>>(
      u_weight_field_[TYPE_AIR], u_weight_field_[TYPE_LIQ], level_set_, 0);
  CalcBorderScaleKernel<<<LAUNCH_SIZE(v_weight_field_[0].Size())>>>(
      v_weight_field_[TYPE_AIR], v_weight_field_[TYPE_LIQ], level_set_, 1);
  CalcBorderScaleKernel<<<LAUNCH_SIZE(w_weight_field_[0].Size())>>>(
      w_weight_field_[TYPE_AIR], w_weight_field_[TYPE_LIQ], level_set_, 2);
  //    u_weight_field_[0].PlotCSV(GRID_SIZE_X / 2);
  //    u_weight_field_[1].PlotCSV(GRID_SIZE_X / 2);
  //    v_weight_field_[0].PlotCSV(GRID_SIZE_X / 2);
  //    v_weight_field_[1].PlotCSV(GRID_SIZE_X / 2);
  //    w_weight_field_[0].PlotCSV(GRID_SIZE_X / 2);
  //    w_weight_field_[1].PlotCSV(GRID_SIZE_X / 2);
  PrepareReverseDivergence<<<LAUNCH_SIZE(divergence_.size())>>>(
      divergence_.data().get(), u_field_[0], u_field_[1], v_field_[0],
      v_field_[1], w_field_[0], w_field_[1], u_weight_field_[0],
      u_weight_field_[1], v_weight_field_[0], v_weight_field_[1],
      w_weight_field_[0], w_weight_field_[1]);

  // PlotMatrix();

  SolvePressure();
  //  CalcPressureImpactToDivergence(pressure_, buffer_);
  //  for (int i = 0; i < buffer_.size(); i++) {
  //    if (buffer_[i] || divergence_[i]) {
  //      std::cout << buffer_[i] << ' ' << divergence_[i] << '\n';
  //    }
  //  }

  //  std::ofstream file("grid.csv");
  //  int y = GRID_SIZE_Y / 2;
  //  file << std::fixed << std::setprecision(7);
  //  for (int x = 0; x < GRID_SIZE_X; x++) {
  //    for (int z = 0; z < GRID_SIZE_Z; z++) {
  //      file << pressure_[x * GRID_SIZE_Y * GRID_SIZE_Z + y * GRID_SIZE_Z +
  //                          z] << ",";
  //    }
  //    file << std::endl;
  //  }
  //  file.close();
  //  std::system("start grid.csv");
  //  std::system("pause");
  CalcImpactToVelFieldKernel<<<LAUNCH_SIZE(u_field_[0].Size())>>>(
      u_field_[0], pressure_.data().get(), RHO_AIR, delta_t_, 0);
  CalcImpactToVelFieldKernel<<<LAUNCH_SIZE(u_field_[1].Size())>>>(
      u_field_[1], pressure_.data().get(), RHO_LIQ, delta_t_, 0);
  CalcImpactToVelFieldKernel<<<LAUNCH_SIZE(v_field_[0].Size())>>>(
      v_field_[0], pressure_.data().get(), RHO_AIR, delta_t_, 1);
  CalcImpactToVelFieldKernel<<<LAUNCH_SIZE(v_field_[1].Size())>>>(
      v_field_[1], pressure_.data().get(), RHO_LIQ, delta_t_, 1);
  CalcImpactToVelFieldKernel<<<LAUNCH_SIZE(w_field_[0].Size())>>>(
      w_field_[0], pressure_.data().get(), RHO_AIR, delta_t_, 2);
  CalcImpactToVelFieldKernel<<<LAUNCH_SIZE(w_field_[1].Size())>>>(
      w_field_[1], pressure_.data().get(), RHO_LIQ, delta_t_, 2);

  Grid2ParticleKernel<<<LAUNCH_SIZE(particles_.size())>>>(
      dev_particles.data().get(), u_field_[0], u_field_[1], u_weight_field_[0],
      u_weight_field_[1], particles_.size(), 0);
  Grid2ParticleKernel<<<LAUNCH_SIZE(particles_.size())>>>(
      dev_particles.data().get(), v_field_[0], v_field_[1], v_weight_field_[0],
      v_weight_field_[1], particles_.size(), 1);
  Grid2ParticleKernel<<<LAUNCH_SIZE(particles_.size())>>>(
      dev_particles.data().get(), w_field_[0], w_field_[1], w_weight_field_[0],
      w_weight_field_[1], particles_.size(), 2);
  AdvectKernel<<<LAUNCH_SIZE(particles_.size())>>>(dev_particles.data().get(),
                                                   delta_t_, particles_.size());
  thrust::copy(dev_particles.begin(), dev_particles.end(), particles_.begin());
}

__global__ void InitParticleKernel(Particle *particles, int num_particle) {
  uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= num_particle) {
    return;
  }
  curandState_t state;
  curand_init(blockIdx.x, threadIdx.x, 0, &state);
  Particle particle{};
  do {
    particle.position = glm::vec3{curand_uniform(&state) * SIZE_X,
                                  curand_uniform(&state) * SIZE_Y,
                                  curand_uniform(&state) * SIZE_Z};
  } while (!InsideFreeVolume(particle.position));
  if (particle.position.y < SIZE_Y * 0.5f) {
    particle.type = TYPE_AIR;
  } else {
    particle.type = TYPE_LIQ;
  }
  particle.velocity = {};
  particles[id] = particle;
}

void FluidApp::InitParticles() {
  particles_.resize(settings_.num_particle);
  thrust::device_vector<Particle> dev_particles(settings_.num_particle);
  InitParticleKernel<<<LAUNCH_SIZE(settings_.num_particle)>>>(
      dev_particles.data().get(), settings_.num_particle);
  thrust::copy(dev_particles.begin(), dev_particles.end(), particles_.begin());
}

__global__ void CalcPressureImpactToDivergenceKernel(
    const float *pressure,
    float *delta_div,
    GridDev<float> u_weight_field0,
    GridDev<float> u_weight_field1,
    GridDev<float> v_weight_field0,
    GridDev<float> v_weight_field1,
    GridDev<float> w_weight_field0,
    GridDev<float> w_weight_field1,
    float delta_t) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = id / (GRID_SIZE_Y * GRID_SIZE_Z);
  int idy = (id / GRID_SIZE_Z) % GRID_SIZE_Y;
  int idz = id % GRID_SIZE_Z;
  if (id >= GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z)
    return;
  float p = pressure[id];
  float self = 0.0f;
  if (idx) {
    float value = delta_t * p *
                  (u_weight_field0(idx, idy, idz) / RHO_AIR +
                   u_weight_field1(idx, idy, idz) / RHO_LIQ) /
                  DELTA_X;
    self += value;
    atomicAdd(delta_div + (id - GRID_SIZE_Z * GRID_SIZE_Y), -value);
  }
  if (idx < GRID_SIZE_X - 1) {
    float value = delta_t * p *
                  (u_weight_field0(idx + 1, idy, idz) / RHO_AIR +
                   u_weight_field1(idx + 1, idy, idz) / RHO_LIQ) /
                  DELTA_X;
    self += value;
    atomicAdd(delta_div + (id + GRID_SIZE_Z * GRID_SIZE_Y), -value);
  }
  if (idy) {
    float value = delta_t * p *
                  (v_weight_field0(idx, idy, idz) / RHO_AIR +
                   v_weight_field1(idx, idy, idz) / RHO_LIQ) /
                  DELTA_X;
    self += value;
    atomicAdd(delta_div + (id - GRID_SIZE_Z), -value);
  }
  if (idy < GRID_SIZE_Y - 1) {
    float value = delta_t * p *
                  (v_weight_field0(idx, idy + 1, idz) / RHO_AIR +
                   v_weight_field1(idx, idy + 1, idz) / RHO_LIQ) /
                  DELTA_X;
    self += value;
    atomicAdd(delta_div + (id + GRID_SIZE_Z), -value);
  }
  if (idz) {
    float value = delta_t * p *
                  (w_weight_field0(idx, idy, idz) / RHO_AIR +
                   w_weight_field1(idx, idy, idz) / RHO_LIQ) /
                  DELTA_X;
    self += value;
    atomicAdd(delta_div + (id - 1), -value);
  }
  if (idz < GRID_SIZE_Z - 1) {
    float value = delta_t * p *
                  (w_weight_field0(idx, idy, idz + 1) / RHO_AIR +
                   w_weight_field1(idx, idy, idz + 1) / RHO_LIQ) /
                  DELTA_X;
    self += value;
    atomicAdd(delta_div + (id + 1), -value);
  }
  atomicAdd(delta_div + id, self);
}

void FluidApp::CalcPressureImpactToDivergence(
    const thrust::device_vector<float> &pressure,
    thrust::device_vector<float> &delta_divergence) {
  thrust::fill(delta_divergence.begin(), delta_divergence.end(), 0);
  CalcPressureImpactToDivergenceKernel<<<LAUNCH_SIZE(pressure.size())>>>(
      pressure.data().get(), delta_divergence.data().get(), u_weight_field_[0],
      u_weight_field_[1], v_weight_field_[0], v_weight_field_[1],
      w_weight_field_[0], w_weight_field_[1], delta_t_);
}

template <typename T>
struct SquareOp {
  __host__ __device__ T operator()(const T &x) const {
    return x * x;
  }
};

struct SaxpyFunctor {
  const float a;

  SaxpyFunctor(float _a) : a(_a) {
  }

  __host__ __device__ float operator()(const float &x, const float &y) const {
    return a * x + y;
  }
};

void FluidApp::SolvePressure() {
  CalcPressureImpactToDivergence(pressure_, Ap_vec_);
  SquareOp<float> square_op;
  thrust::plus<float> plus_op;
  thrust::minus<float> minus_op;
  thrust::multiplies<float> multi_op;
  auto dot = [this, multi_op, plus_op](const thrust::device_vector<float> &a,
                                       const thrust::device_vector<float> &b) {
    thrust::transform(a.begin(), a.end(), b.begin(), buffer_.begin(), multi_op);
    return thrust::reduce(buffer_.begin(), buffer_.end(), 0.0f, plus_op);
  };
  auto saxpy = [](float a, const thrust::device_vector<float> &x,
                  const thrust::device_vector<float> &y,
                  thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                      SaxpyFunctor(a));
  };
  auto add = [plus_op](const thrust::device_vector<float> &x,
                       const thrust::device_vector<float> &y,
                       thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), plus_op);
  };
  auto sub = [minus_op](const thrust::device_vector<float> &x,
                        const thrust::device_vector<float> &y,
                        thrust::device_vector<float> &z) {
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), minus_op);
  };
  auto assign = [](const thrust::device_vector<float> &x,
                   thrust::device_vector<float> &y) {
    thrust::copy(x.begin(), x.end(), y.begin());
  };
  CalcPressureImpactToDivergence(pressure_, buffer_);
  sub(divergence_, buffer_, r_vec_);
  assign(r_vec_, p_vec_);
  while (true) {
    float rk2 = dot(r_vec_, r_vec_);
    CalcPressureImpactToDivergence(p_vec_, Ap_vec_);
    float ak = rk2 / dot(p_vec_, Ap_vec_);
    saxpy(ak, p_vec_, pressure_, pressure_);
    saxpy(-ak, Ap_vec_, r_vec_, r_vec_);
    float new_rk2 = dot(r_vec_, r_vec_);
    //    printf("%f\n", new_rk2);
    if (new_rk2 < 1e-4f) {
      break;
    }
    float bk = new_rk2 / rk2;
    saxpy(bk, p_vec_, r_vec_, p_vec_);
  }
}
void FluidApp::PlotMatrix() {
  std::ofstream file("grid.csv");
  file << std::fixed << std::setprecision(7);
  for (int i = 0; i < pressure_.size(); i++) {
    std::cout << i << std::endl;
    if (i) {
      pressure_[i - 1] = 0.0f;
    }
    pressure_[i] = 1.0f;
    CalcPressureImpactToDivergence(pressure_, buffer_);
    for (int j = 0; j < buffer_.size(); j++) {
      file << buffer_[j] << ',';
    }
    file << std::endl;
  }
  file.close();
  std::system("start grid.csv");
}
