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
    swap_chain_ = std::make_unique<class SwapChain>(*factory_, *command_queue_,
                                                    settings_.hwnd);
  }
}

void Core::RebuildSwapChain(int width, int height, DXGI_FORMAT format) {
  if (swap_chain_) {
    swap_chain_->ResizeBuffer(width, height, format);
  }
}

Core::~Core() = default;

}  // namespace grassland::d3d12
