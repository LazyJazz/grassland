#include "application.h"
// Enable GLFW WIN32 natives
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

Application::Application(int width, int height, const char *title) {
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);

  d3d12::CoreSettings settings;
  settings.hwnd = glfwGetWin32Window(window_);
  core_ = std::make_unique<d3d12::Core>(settings);
}

Application::~Application() = default;

void Application::Run() {
  OnInit();

  while (!glfwWindowShouldClose(window_)) {
    OnUpdate();
    OnRender();
    glfwPollEvents();
  }

  OnClose();
}

void Application::OnInit() {
}

void Application::OnUpdate() {
}

void Application::OnRender() {
}

void Application::OnClose() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}
