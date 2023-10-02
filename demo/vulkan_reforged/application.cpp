#include "application.h"
Application::Application(const std::string &name,
                         int width,
                         int height,
                         bool headless) {
  if (!headless) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window_ = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);
  }
  vulkan::CoreSettings core_settings;
  core_settings.window = window_;
  core_ = std::make_unique<vulkan::Core>(core_settings);
}

void Application::Run() {
  OnInit();
  if (window_) {
    while (!glfwWindowShouldClose(window_)) {
      glfwPollEvents();
      OnUpdate();
      OnRender();
    }
  } else {
    while (!application_should_close_) {
      OnUpdate();
      OnRender();
    }
  }
  OnClose();
}

void Application::OnUpdate() {
}

void Application::OnRender() {
  core_->BeginFrame();
  core_->EndFrame();
}

void Application::OnInit() {
}

void Application::OnClose() {
}
