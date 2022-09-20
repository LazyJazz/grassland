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
  physical_device_ = std::make_unique<vulkan::PhysicalDevice>(
      vulkan::PickPhysicalDevice(instance_.get(), [](vulkan::PhysicalDevice
                                                         physical_device) {
        int score = 0;
        if (physical_device.IsDiscreteGPU())
          score += 1000;
        score +=
            int(physical_device.GetProperties().limits.maxImageDimension2D);
        return score;
      }).GetHandle());
  spdlog::info("Picked device:");
  physical_device_->PrintDeviceProperties();
  device_ =
      std::make_unique<vulkan::Device>(physical_device_.get(), surface_.get());
  graphics_queue_ = std::make_unique<vulkan::Queue>(
      device_.get(), physical_device_->GraphicsFamilyIndex());
  present_queue_ = std::make_unique<vulkan::Queue>(
      device_.get(), physical_device_->PresentFamilyIndex(surface_.get()));
  swap_chain_ = std::make_unique<vulkan::SwapChain>(window_, device_.get());
  render_pass_ = std::make_unique<vulkan::RenderPass>(device_.get(),
                                                      swap_chain_->GetFormat());
}

void App::OnLoop() {
}

void App::OnClose() {
  render_pass_.reset();
  swap_chain_.reset();
  device_.reset();
  physical_device_.reset();
  surface_.reset();
  instance_.reset();
}

void App::OnDestroy() {
}
