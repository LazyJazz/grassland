#include "grassland/d3d12/core/core.h"

namespace grassland::d3d12 {

Core::Core(const CoreSettings &settings) : settings_(settings) {
  // Create the factory
  DXGIFactorySettings factory_settings;
  factory_settings.enable_debug_layer = settings_.enable_debug_layer;
  factory_ = std::make_unique<DXGIFactory>(factory_settings);

  // Enumerate the adapters
  auto adapters = factory_->EnumerateAdapters();
  if (adapters.empty()) {
    throw std::runtime_error("No adapters found");
  }

  DeviceSettings device_settings;
  if (settings_.device_index >= 0) {
    if (settings_.device_index >= adapters.size()) {
      throw std::runtime_error("Invalid device index");
    }
    device_settings.adapter = adapters[settings_.device_index];
  } else {
    // Pick the first adapter
    device_settings.adapter = adapters[0];
    for (const auto &adapter : adapters) {
      if (adapter.Evaluate() > device_settings.adapter.Evaluate()) {
        device_settings.adapter = adapter;
        break;
      }
    }
  }

  // Output all the adapters and properties
  spdlog::info("--- {} ---", device_settings.adapter.Description());
  spdlog::info(
      "Device vendor: {}",
      util::PCIVendorIDToName(device_settings.adapter.GetDesc().VendorId));

  // Create the device
  device_ = std::make_unique<class Device>(device_settings);

  // Create the command allocator
  command_allocator_ = std::make_unique<class CommandAllocator>(*device_);

  // Create the command queue
  command_queue_ = std::make_unique<class CommandQueue>(*device_);

  // Create the command lists
  command_list_.resize(settings_.max_frames_in_flight);
  for (auto &command_list : command_list_) {
    command_list = std::make_unique<class GraphicsCommandList>(
        *device_, *command_allocator_);
  }

  // Create the swap chain
  if (settings_.hwnd) {
    SwapChainSettings swap_chain_settings(settings_.hwnd);
    swap_chain_settings.frame_count = settings_.max_frames_in_flight;
    swap_chain_ = std::make_unique<class SwapChain>(*factory_, *command_queue_,
                                                    swap_chain_settings);
    image_index_ = swap_chain_->Ptr()->GetCurrentBackBufferIndex();
  }

  // Create the fence
  fence_ = std::make_unique<class Fence>(*device_);
  fence_values_.resize(settings_.max_frames_in_flight);
  fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (fence_event_ == nullptr) {
    ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()),
                  "Failed to create fence event");
  }
}

void Core::RebuildSwapChain(int width, int height, DXGI_FORMAT format) {
  if (swap_chain_) {
    swap_chain_->ResizeBuffer(width, height, format);
    image_index_ = swap_chain_->Ptr()->GetCurrentBackBufferIndex();
  }
}

void Core::BeginFrame() {
  // Reset the command allocator
  ThrowIfFailed(command_allocator_->Ptr()->Reset(),
                "Failed to reset allocator");

  // Reset the command list
  ThrowIfFailed(command_list_[current_frame_]->Ptr()->Reset(
                    command_allocator_->Ptr().Get(), nullptr),
                "Failed to reset command list");

  // Reset the swap chain
  if (swap_chain_) {
    image_index_ = swap_chain_->Ptr()->GetCurrentBackBufferIndex();
  }
}

void Core::EndFrame() {
  // Close the command list
  command_list_[current_frame_]->Ptr()->Close();

  // Execute the command list
  ID3D12CommandList *command_lists[] = {
      command_list_[current_frame_]->Ptr().Get()};
  command_queue_->Ptr()->ExecuteCommandLists(1, command_lists);

  // Present
  if (swap_chain_) {
    swap_chain_->Ptr()->Present(1, 0);
  }

  MoveToNextFrame();

  current_frame_ = (current_frame_ + 1) % settings_.max_frames_in_flight;
}

void Core::MoveToNextFrame() {
  // Schedule a Signal command in the queue.
  const uint64_t current_fence_value = fence_values_[image_index_];
  ThrowIfFailed(
      command_queue_->Ptr()->Signal(fence_->Ptr().Get(), current_fence_value),
      "Failed to signal fence");

  // Update the frame index.
  image_index_ = swap_chain_->Ptr()->GetCurrentBackBufferIndex();

  // If the next frame is not ready to be rendered yet, wait until it is ready.
  if (fence_->Ptr()->GetCompletedValue() < fence_values_[image_index_]) {
    ThrowIfFailed(fence_->Ptr()->SetEventOnCompletion(
                      fence_values_[image_index_], fence_event_),
                  "Failed to set event on completion");
    WaitForSingleObjectEx(fence_event_, INFINITE, FALSE);
  }

  // Set the fence value for the next frame.
  fence_values_[image_index_] = current_fence_value + 1;
}

void Core::WaitForGPU() {
  // Schedule a Signal command in the queue.
  ThrowIfFailed(command_queue_->Ptr()->Signal(fence_->Ptr().Get(),
                                              fence_values_[image_index_]),
                "Failed to signal fence");

  // Wait until the fence has been processed.
  ThrowIfFailed(fence_->Ptr()->SetEventOnCompletion(fence_values_[image_index_],
                                                    fence_event_),
                "Failed to set event on completion");
  WaitForSingleObjectEx(fence_event_, INFINITE, FALSE);

  // Increment the fence value for the current frame.
  fence_values_[image_index_]++;
}

Core::~Core() {
  // Wait for the GPU to be done with all resources.
  WaitForGPU();

  CloseHandle(fence_event_);
}

}  // namespace grassland::d3d12
