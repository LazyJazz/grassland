#include "fluid_large.cuh"

FluidLarge::FluidLarge(const char *title, int width, int height) {
  vulkan::framework::CoreSettings core_settings;
  core_settings.window_width = width;
  core_settings.window_height = height;
  core_settings.window_title = title;
  core_ = std::make_unique<vulkan::framework::Core>(core_settings);
}

void FluidLarge::Run() {
  OnInit();
  while (!glfwWindowShouldClose(core_->GetWindow())) {
    OnLoop();
    glfwPollEvents();
  }
  core_->GetDevice()->WaitIdle();
  OnClose();
}

void FluidLarge::OnInit() {

}

void FluidLarge::OnLoop() {

}

void FluidLarge::OnClose() {

}
