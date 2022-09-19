#include "app.h"

#include <grassland/logging/logging.h>

App::App(int width, int height, const char *title) {
  if (!glfwInit()) {
    LAND_ERROR("GLFW Init failed!");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);

  if (!window_) {
    LAND_ERROR("glfwCreateWindow failed!");
  }

  OnCreate();
}

App::~App() {
  if (window_) {
    glfwDestroyWindow(window_);
  }
  glfwTerminate();

  OnDestroy();
}

void App::Run() {
  OnInit();
  while (!glfwWindowShouldClose(window_)) {
    OnLoop();
    glfwPollEvents();
  }
  OnClose();
}

void App::OnCreate() {
}

void App::OnInit() {
  instance_ = std::make_unique<vulkan::Instance>();
  surface_ = std::make_unique<vulkan::Surface>(instance_.get(), window_);

  auto physical_devices = vulkan::GetPhysicalDevices(instance_.get());
  spdlog::info("Vulkan physical devices:");
  for (auto &physical_device : physical_devices) {
    spdlog::info("  {}", physical_device.DeviceName());
  }
}

void App::OnLoop() {
}

void App::OnClose() {
  surface_.reset();
  instance_.reset();
}

void App::OnDestroy() {
}
