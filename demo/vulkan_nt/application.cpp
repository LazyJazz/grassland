#include "application.h"

Application::Application(const char *title, int width, int height) {
  if (!glfwInit()) {
    LAND_ERROR("GLFW init failed.")
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if (!window_) {
    LAND_ERROR("GLFW create window failed.");
  }
}

Application::~Application() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void Application::Run() {
  OnInit();
  while (!glfwWindowShouldClose(window_)) {
    OnLoop();
    glfwPollEvents();
  }
  OnClose();
}

void Application::OnInit() {
  instance_ =
      std::make_unique<vulkan::Instance>(vulkan::InstanceSettings{true, true});
  surface_ = std::make_unique<vulkan::Surface>(*instance_, window_);
  device_ =
      std::make_unique<vulkan::Device>(instance_->PickDevice(), surface_.get());
  swapchain_ = std::make_unique<vulkan::Swapchain>(*device_);
}

void Application::OnLoop() {
}

void Application::OnClose() {
}
