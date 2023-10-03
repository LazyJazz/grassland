#include "application.h"

#include <iostream>
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
  // List all the builtin shaders, show the filename and get the compiled spv
  // code.
  std::vector<std::string> shader_names =
      vulkan::built_in_shaders::ListAllBuiltInShaders();
  for (const auto &shader_name : shader_names) {
    spdlog::info(
        "Shader name: {} Spv size: {}", shader_name,
        vulkan::built_in_shaders::GetShaderCompiledSpv(shader_name).size() *
            sizeof(uint32_t));
  }
}

void Application::OnClose() {
}
