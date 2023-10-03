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
  // Define a hello world triangle
  struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
  };

  std::vector<Vertex> vertices = {
      {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
      {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
      {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
  };

  std::vector<uint16_t> indices = {0, 1, 2};

  vertex_buffer_ = std::make_unique<vulkan::Buffer>(
      core_.get(), vertices.size() * sizeof(Vertex),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);
  vulkan::UploadBuffer(vertex_buffer_.get(), vertices.data(),
                       vertices.size() * sizeof(Vertex));
  index_buffer_ = std::make_unique<vulkan::Buffer>(
      core_.get(), indices.size() * sizeof(uint16_t),
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);
  vulkan::UploadBuffer(index_buffer_.get(), indices.data(),
                       indices.size() * sizeof(uint16_t));

  // Output notice: What shader is creating

  spdlog::info("Compiling vertex shader: {}", "hello_world.vert");

  vertex_shader_ = std::make_unique<vulkan::ShaderModule>(
      core_.get(),
      vulkan::built_in_shaders::GetShaderCompiledSpv("hello_world.vert"));

  spdlog::info("Compiling fragment shader: {}", "hello_world.frag");
  fragment_shader_ = std::make_unique<vulkan::ShaderModule>(
      core_.get(),
      vulkan::built_in_shaders::GetShaderCompiledSpv("hello_world.frag"));
}

void Application::OnClose() {
  // Release resources in reverse order of creation
  vertex_shader_.reset();
  fragment_shader_.reset();
  index_buffer_.reset();
  vertex_buffer_.reset();
  core_.reset();
  if (window_) {
    glfwDestroyWindow(window_);
    glfwTerminate();
  }
}
