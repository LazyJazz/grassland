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

  manager_ = std::make_unique<grassland::vulkan::gui::Manager>(core_.get());
  manager_->BindFrameTexture(paint_buffer_.get());
  window_ = std::make_unique<grassland::vulkan::gui::Window>(
      manager_.get(), grassland::vulkan::gui::Layout{10, 10, 320, 320},
      "Test Window", grassland::vulkan::gui::WindowFlag::WINDOW_FLAG_BAR_BIT);
}

void GuiExample::OnLoop() {
  core_->BeginCommandRecord();
  paint_buffer_->ClearColor({0.6f, 0.7f, 0.8f, 1.0f});
  manager_->Draw();
  core_->Output(paint_buffer_.get());
  core_->EndCommandRecordAndSubmit();
}

void GuiExample::OnClose() {
  core_->GetDevice()->WaitIdle();
  window_.reset();
  manager_.reset();
  paint_buffer_.reset();
  core_.reset();
}
