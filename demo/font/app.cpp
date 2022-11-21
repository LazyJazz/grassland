#include "app.h"

#include <utility>

namespace {
struct UniformBufferObject {};
}  // namespace

FontViewer::FontViewer(grassland::font::Mesh font_mesh)
    : font_mesh_(std::move(font_mesh)) {
}

void FontViewer::Run() {
  OnInit();
  while (!glfwWindowShouldClose(core_->GetWindow())) {
    OnLoop();
    glfwPollEvents();
  }
  OnClose();
}

void FontViewer::OnInit() {
  core_ = std::make_unique<grassland::vulkan::framework::Core>(core_settings_);
}

void FontViewer::OnLoop() {
}

void FontViewer::OnClose() {
  core_->GetDevice()->WaitIdle();
  core_.reset();
}
