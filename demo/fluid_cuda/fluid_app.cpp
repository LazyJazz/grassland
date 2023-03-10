#include "fluid_app.h"

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
  frame_image_ = std::make_unique<grassland::vulkan::framework::TextureImage>(
      core_.get(), core_->GetFramebufferWidth(), core_->GetFramebufferHeight(),
      VK_FORMAT_B8G8R8A8_UNORM);
  core_->SetFrameSizeCallback(
      [this](int width, int height) { frame_image_->Resize(width, height); });
}

void FluidApp::OnLoop() {
  OnUpdate();
  OnRender();
}

void FluidApp::OnClose() {
}

void FluidApp::OnUpdate() {
}

void FluidApp::OnRender() {
  core_->BeginCommandRecord();
  frame_image_->ClearColor({0.6f, 0.7f, 0.8f, 1.0f});
  core_->Output(frame_image_.get());
  core_->EndCommandRecordAndSubmit();
}
