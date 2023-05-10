#include "nbody.h"

NBody::NBody(int n_particles) : n_particles_(n_particles) {
  vulkan_legacy::framework::CoreSettings core_settings;
  core_settings.window_width = 1920;
  core_settings.window_height = 1080;
  core_settings.window_title = "NBody";
  core_ = std::make_unique<vulkan_legacy::framework::Core>(core_settings);
}

void NBody::Run() {
  OnInit();
  while (!glfwWindowShouldClose(core_->GetWindow())) {
    OnUpdate();
    OnRender();
    glfwPollEvents();
  }
  OnClose();
  core_->GetDevice()->WaitIdle();
}

void NBody::OnUpdate() {
  UpdateParticles();
  UpdateImGui();
  auto world_to_cam =
      glm::lookAt(glm::vec3{rotation * glm::vec4{10.0f, 20.0f, 30.0f, 0.0f}},
                  glm::vec3{0.0f}, glm::vec3{0.0f, 1.0f, 0.0f});
  (*global_uniform_object_)[0] = GlobalUniformObject{
      glm::perspective(glm::radians(60.0f),
                       float(core_->GetFramebufferWidth()) /
                           float(core_->GetFramebufferHeight()),
                       0.1f, 100.0f) *
          world_to_cam,
      glm::inverse(world_to_cam), PARTICLE_SIZE};
  std::memcpy(&((*particle_positions_)[0]), positions_.data(),
              sizeof(glm::vec4) * n_particles_);
}

void NBody::OnRender() {
  core_->BeginCommandRecord();
  frame_buffer_->ClearColor({0.0f, 0.0f, 0.0f, 0.0f});
  render_node_->Draw(vertex_buffer_.get(), index_buffer_.get(),
                     index_buffer_->Size(), 0, n_particles_);
  core_->ImGuiRender();
  core_->Output(frame_buffer_.get());
  core_->EndCommandRecordAndSubmit();
}

void NBody::OnInit() {
  std::vector<glm::vec2> vertices = {
      {-1.0f, -1.0f}, {-1.0f, 1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}};
  std::vector<uint32_t> indices = {0, 1, 2, 1, 2, 3};
  vertex_buffer_ =
      std::make_unique<vulkan_legacy::framework::StaticBuffer<glm::vec2>>(
          core_.get(), vertices.size());
  vertex_buffer_->Upload(vertices.data());
  index_buffer_ =
      std::make_unique<vulkan_legacy::framework::StaticBuffer<uint32_t>>(
          core_.get(), indices.size());
  index_buffer_->Upload(indices.data());

  frame_buffer_ = std::make_unique<vulkan_legacy::framework::TextureImage>(
      core_.get(), core_->GetFramebufferWidth(), core_->GetFramebufferHeight(),
      VK_FORMAT_B8G8R8A8_UNORM);

  core_->SetFrameSizeCallback([this](int width, int height) {
    frame_buffer_->Resize(width, height);
    BuildRenderNode();
  });

  global_uniform_object_ = std::make_unique<
      vulkan_legacy::framework::DynamicBuffer<GlobalUniformObject>>(core_.get(),
                                                                    1);
  particle_positions_ =
      std::make_unique<vulkan_legacy::framework::DynamicBuffer<glm::vec4>>(
          core_.get(), n_particles_);

  positions_.resize(n_particles_);
  velocities_.resize(n_particles_);

  std::vector<glm::vec4> origins;
  std::vector<glm::vec4> initial_vels;
  for (int i = 0; i < 10; i++) {
    origins.emplace_back(RandomInSphere() * INITIAL_RADIUS * 2.0f, 0.0f);
    initial_vels.emplace_back(RandomInSphere() * INITIAL_RADIUS * 0.1f, 0.0f);
  }
  for (int i = 0; i < n_particles_; i++) {
    auto &pos = positions_[i];
    auto &vel = velocities_[i];
    int index = std::uniform_int_distribution<int>(
        0, origins.size() - 1)(random_device_);
    pos = glm::vec4{RandomInSphere() * INITIAL_RADIUS * 0.2f, 1.0f} +
          origins[index];
    vel =
        glm::vec4{RandomInSphere() * INITIAL_SPEED, 0.0f} + initial_vels[index];
  }
  core_->ImGuiInit(frame_buffer_.get());
  BuildRenderNode();
  core_->SetCursorPosCallback([this](double xpos, double ypos) {
    static auto last_xpos = xpos;
    static auto last_ypos = ypos;
    if (glfwGetMouseButton(core_->GetWindow(), GLFW_MOUSE_BUTTON_LEFT) ==
        GLFW_PRESS) {
      auto diffx = xpos - last_xpos;
      auto diffy = ypos - last_ypos;
      rotation *= glm::rotate(glm::mat4{1.0f}, glm::radians(float(diffx)),
                              glm::vec3{1.0f, 0.0f, 0.0f});
      rotation *= glm::rotate(glm::mat4{1.0f}, glm::radians(float(diffy)),
                              glm::vec3{0.0f, 1.0f, 0.0f});
    }
    last_xpos = xpos;
    last_ypos = ypos;
  });
}

void NBody::OnClose() {
}

void NBody::BuildRenderNode() {
  render_node_ =
      std::make_unique<vulkan_legacy::framework::RenderNode>(core_.get());
  render_node_->VertexInput({VK_FORMAT_R32G32_SFLOAT});
  render_node_->AddColorAttachment(
      frame_buffer_.get(),
      VkPipelineColorBlendAttachmentState{
          VK_TRUE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD,
          VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
          VK_BLEND_OP_ADD,
          VK_COLOR_COMPONENT_A_BIT | VK_COLOR_COMPONENT_R_BIT |
              VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT});
  render_node_->AddUniformBinding(global_uniform_object_.get(),
                                  VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddBufferBinding(particle_positions_.get(),
                                 VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddShader("color_shader.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddShader("color_shader.frag.spv",
                          VK_SHADER_STAGE_FRAGMENT_BIT);
  render_node_->BuildRenderNode();
}

float NBody::RandomFloat() {
  return std::uniform_real_distribution<float>()(random_device_);
}

glm::vec3 NBody::RandomOnSphere() {
  float z = RandomFloat() * 2.0f - 1.0f;
  float inv_z = std::sqrt(1.0f - z * z);
  float theta = RandomFloat() * glm::pi<float>() * 2.0f;
  float x = inv_z * std::sin(theta);
  float y = inv_z * std::cos(theta);
  return {x, y, z};
}

glm::vec3 NBody::RandomInSphere() {
  return RandomOnSphere() * std::pow(RandomFloat(), 0.333333333333333333f);
}

void NBody::UpdateParticles() {
#if !ENABLE_GPU
  for (int i = 0; i < n_particles_; i++) {
    auto &pos_i = positions_[i];
    for (int j = 0; j < n_particles_; j++) {
      auto &pos_j = positions_[j];
      auto diff = pos_i - pos_j;
      auto l = glm::length(diff);
      if (l < DELTA_T) {
        continue;
      }
      diff /= l * l * l;
      velocities_[i] += -diff * DELTA_T * GRAVITY_COE;
    }
  }

  for (int i = 0; i < n_particles_; i++) {
    positions_[i] += velocities_[i] * DELTA_T;
  }
#else
  UpdateStep(positions_.data(), velocities_.data(), n_particles_);
#endif
}

void NBody::UpdateImGui() {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2{0.0f, 0.0f}, ImGuiCond_Once);
  ImGui::SetNextWindowBgAlpha(0.3f);
  if (ImGui::Begin("Statistics"), nullptr, ImGuiWindowFlags_NoMove) {
    auto current_tp = std::chrono::steady_clock::now();
    static auto last_frame_tp = current_tp;
    auto duration = current_tp - last_frame_tp;
    auto duration_ms = float(duration / std::chrono::microseconds(1)) * 1e-3f;
    ImGui::Text("Frame Duration: %.3f ms", duration_ms);
    ImGui::Text("FPS: %.3f", 1e3f / duration_ms);
    float ops =
        float(n_particles_) * float(n_particles_) / (duration_ms * 1e-3f);
    if (ops < 8e2f) {
      ImGui::Text("%.2f op/s", ops);
    } else if (ops < 8e5f) {
      ImGui::Text("%.2f Kop/s", ops * 1e-3f);
    } else if (ops < 8e8f) {
      ImGui::Text("%.2f Mop/s", ops * 1e-6f);
    } else {
      ImGui::Text("%.2f Gop/s", ops * 1e-9f);
    }
    ImGui::End();
    last_frame_tp = current_tp;
  }
  ImGui::Render();
}
