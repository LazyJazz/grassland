#include "rtapp.h"

RayTracingApp::RayTracingApp(uint32_t width, uint32_t height) {
  grassland::vulkan::framework::CoreSettings core_settings;
  core_settings.window_width = width;
  core_settings.window_height = height;
  core_settings.raytracing_pipeline_required = true;
  core_settings.validation_layer = true;
  core_ = std::make_unique<grassland::vulkan::framework::Core>(core_settings);
}

void RayTracingApp::Run() {
  OnInit();
  while (!glfwWindowShouldClose(core_->GetWindow())) {
    OnLoop();
    glfwPollEvents();
  }
  OnClose();
}

void RayTracingApp::OnInit() {
  frame_texture_ = std::make_unique<grassland::vulkan::framework::TextureImage>(
      core_.get(), core_->GetFramebufferWidth(), core_->GetFramebufferHeight(),
      VK_FORMAT_B8G8R8A8_UNORM);
  core_->SetFrameSizeCallback(
      [this](int width, int height) { frame_texture_->Resize(width, height); });

  std::vector<glm::vec3> vertices = {
      {1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}};
  std::vector<uint32_t> indices = {0, 1, 2};
  bottom_level_acceleration_structure_ = std::make_unique<
      grassland::vulkan::raytracing::BottomLevelAccelerationStructure>(
      core_->GetDevice(), core_->GetCommandPool(), vertices.data(),
      sizeof(glm::vec3) * vertices.size(), indices.data(), indices.size(),
      sizeof(glm::vec3));
  top_level_acceleration_structure_ = std::make_unique<
      grassland::vulkan::raytracing::TopLevelAccelerationStructure>(
      core_->GetDevice(), core_->GetCommandPool(),
      std::vector<std::pair<
          grassland::vulkan::raytracing::BottomLevelAccelerationStructure *,
          glm::mat4>>{
          {bottom_level_acceleration_structure_.get(), glm::mat4{1.0f}}});
}

void RayTracingApp::OnLoop() {
  OnUpdate();
  OnRender();
}

void RayTracingApp::OnClose() {
  core_->GetDevice()->WaitIdle();
  // frame_texture_.reset();
}

void RayTracingApp::OnUpdate() {
}

void RayTracingApp::OnRender() {
  core_->BeginCommandRecord();
  frame_texture_->ClearColor({0.6f, 0.7f, 0.8f, 1.0f});
  core_->Output(frame_texture_.get());
  core_->EndCommandRecordAndSubmit();
}
