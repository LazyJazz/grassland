#include "gui_example.h"

GuiExample::GuiExample() {
}

void GuiExample::Run() {
  OnInit();
  while (!glfwWindowShouldClose(core_->GetWindow())) {
    OnLoop();
    glfwPollEvents();
  }
  OnClose();
}

void GuiExample::OnInit() {
  grassland::vulkan::framework::CoreSettings core_settings;
  core_ = std::make_unique<grassland::vulkan::framework::Core>(core_settings);
  paint_buffer_ = std::make_unique<grassland::vulkan::framework::TextureImage>(
      core_.get(), core_->GetFramebufferWidth(), core_->GetFramebufferHeight(),
      VK_FORMAT_B8G8R8A8_UNORM);
  core_->SetFrameSizeCallback(
      [this](int width, int height) { paint_buffer_->Resize(width, height); });
}

void GuiExample::OnLoop() {
  core_->BeginCommandRecord();
  paint_buffer_->ClearColor({0.6f, 0.7f, 0.8f, 1.0f});
  core_->Output(paint_buffer_.get());
  core_->EndCommandRecordAndSubmit();
}

void GuiExample::OnClose() {
  core_->GetDevice()->WaitIdle();
  paint_buffer_.reset();
  core_.reset();
}
