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
  test_model_ = std::make_unique<grassland::vulkan::gui::Model>(manager_.get());
  test_model_->UploadMesh(
      {{{10.0f, 10.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 0.5f}},
       {{10.0f, 110.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 0.5f}},
       {{110.0f, 10.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 0.5f}},
       {{110.0f, 110.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 0.5f}}},
      {0, 1, 2, 1, 2, 3});
}

void GuiExample::OnLoop() {
  core_->BeginCommandRecord();
  paint_buffer_->ClearColor({0.6f, 0.7f, 0.8f, 1.0f});
  manager_->BeginDraw();
  test_model_->Draw();
  manager_->EndDraw();
  core_->Output(paint_buffer_.get());
  core_->EndCommandRecordAndSubmit();
}

void GuiExample::OnClose() {
  core_->GetDevice()->WaitIdle();
  test_model_.reset();
  manager_.reset();
  paint_buffer_.reset();
  core_.reset();
}
