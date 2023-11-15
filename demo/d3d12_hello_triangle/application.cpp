#include "application.h"
// Enable GLFW WIN32 natives
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

Application::Application(int width, int height, const char *title)
    : title_(title) {
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);

  d3d12::CoreSettings settings;
  settings.hwnd = glfwGetWin32Window(window_);
  core_ = std::make_unique<d3d12::Core>(settings);

  // Get Swap Chain and output width and height
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(
      window_, [](GLFWwindow *window, int width, int height) {
        auto app =
            reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
        app->core_->WaitForGPU();
        app->core_->RebuildSwapChain(width, height);
      });
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
  // Output the frame rate
  //  static auto last_time = std::chrono::high_resolution_clock::now();
  //  static int frame_count = 0;
  //  auto now = std::chrono::high_resolution_clock::now();
  //  auto delta =
  //      std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time)
  //          .count();
  //  if (delta > 1000) {
  //    auto fps = frame_count * 1000.0f / delta;
  //    std::string title = title_ + " - " + std::to_string(fps) + " fps";
  //    glfwSetWindowTitle(window_, title.c_str());
  //    frame_count = 0;
  //    last_time = now;
  //  }
}

void Application::OnRender() {
  core_->BeginFrame();
  auto command_list = core_->CommandList()->Ptr();
  // Transit resource state from present to render target
  auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      core_->SwapChain()->BackBuffer(core_->ImageIndex()),
      D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
  command_list->ResourceBarrier(1, &barrier);

  // Transit resource state from render target to present
  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      core_->SwapChain()->BackBuffer(core_->ImageIndex()),
      D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
  command_list->ResourceBarrier(1, &barrier);
  core_->EndFrame();
}

void Application::OnClose() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}
