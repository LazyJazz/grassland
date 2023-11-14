#include "grassland/d3d12/core/core.h"

namespace grassland::d3d12 {

Core::Core(const CoreSettings &settings) {
  // Create the factory
  DXGIFactorySettings factory_settings;
  factory_settings.enable_debug_layer = settings.enable_debug_layer;
  factory_ = std::make_unique<DXGIFactory>(factory_settings);

  // Enumerate the adapters
  auto adapters = factory_->EnumerateAdapters();
  if (adapters.empty()) {
    throw std::runtime_error("No adapters found");
  }

  DeviceSettings device_settings;
  if (settings.device_index >= 0) {
    if (settings.device_index >= adapters.size()) {
      throw std::runtime_error("Invalid device index");
    }
    device_settings.adapter = adapters[settings.device_index];
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
}

Core::~Core() {
  // Release all resources in reverse order of creation
  factory_.reset();
}

}  // namespace grassland::d3d12
