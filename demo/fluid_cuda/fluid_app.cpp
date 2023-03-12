#include "fluid_app.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

FluidApp::FluidApp(const FluidAppSettings &settings) {
  settings_ = settings;
  grassland::vulkan::framework::CoreSettings core_settings{};
  core_settings.window_width = 1024;
  core_settings.window_height = 1024;
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

  camera_ =
      Camera(glm::vec3{-10.0f, 20.0f, 20.0f}, glm::vec3{10.0f, 10.0f, 10.0f});
  RegisterSphere();
  RegisterCylinder();

  InitParticles();
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

  {
    const glm::vec3 origin = BOUNDARY_CENTER;
    const float sphere_radius = BOUNDARY_RADIUS;
    const float thickness = 0.01f;
    const glm::vec4 color = {0.5f, 0.6f, 0.8f, 1.0f};
    float pi = glm::pi<float>();
    int precision = 12;
    float inv_precision = 1.0f / precision;
    for (int i = 0; i < precision; i++) {
      int i1 = i + 1;
      float sin_i = std::sin(i * inv_precision * pi),
            cos_i = std::cos(i * inv_precision * pi);
      float sin_i1 = std::sin(i1 * inv_precision * pi),
            cos_i1 = std::cos(i1 * inv_precision * pi);
      for (int j = 0; j < precision * 2; j++) {
        float sin_j = std::sin(j * inv_precision * pi);
        float cos_j = std::cos(j * inv_precision * pi);
        int j1 = j + 1;
        float sin_j1 = std::sin(j1 * inv_precision * pi);
        float cos_j1 = std::cos(j1 * inv_precision * pi);
        DrawLine(
            glm::vec3{sin_i1 * sin_j, cos_i1, sin_i1 * cos_j} * sphere_radius +
                origin,
            glm::vec3{sin_i * sin_j, cos_i, sin_i * cos_j} * sphere_radius +
                origin,
            thickness, color);
        if (i) {
          DrawLine(
              glm::vec3{sin_i * sin_j1, cos_i, sin_i * cos_j1} * sphere_radius +
                  origin,
              glm::vec3{sin_i * sin_j, cos_i, sin_i * cos_j} * sphere_radius +
                  origin,
              thickness, color);
          DrawSphere(
              glm::vec3{sin_i * sin_j, cos_i, sin_i * cos_j} * sphere_radius +
                  origin,
              0.05f, glm::vec4{1.0f, 1.0f, 1.0f, 1.0f});
        }
      }
    }

    DrawSphere(glm::vec3{0.0f, -1.0f, 0.0f} * sphere_radius + origin, 0.05f,
               glm::vec4{1.0f, 1.0f, 1.0f, 1.0f});
    DrawSphere(glm::vec3{0.0f, 1.0f, 0.0f} * sphere_radius + origin, 0.05f,
               glm::vec4{1.0f, 1.0f, 1.0f, 1.0f});
  }

  for (auto &particle : particles_) {
    DrawSphere(particle.position, RENDER_SIZE,
               glm::vec4{0.2f, 0.2f, 0.8f, 1.0f});
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
