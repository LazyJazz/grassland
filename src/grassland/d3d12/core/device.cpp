#include "grassland/d3d12/core/device.h"

namespace grassland::d3d12 {

Device::Device(const DeviceSettings &settings) : adapter_(settings.adapter) {
  // Create the device
  if (settings.adapter.Ptr()) {
    ThrowIfFailed(
        D3D12CreateDevice(settings.adapter.Ptr().Get(), D3D_FEATURE_LEVEL_11_0,
                          IID_PPV_ARGS(&device_)),
        "Failed to create device");
  }
}

Device::~Device() = default;

ComPtr<ID3D12Device> Device::Ptr() const {
  return device_;
}

class Adapter Device::Adapter() const {
  return adapter_;
}

}  // namespace grassland::d3d12
