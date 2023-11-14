#pragma once
#include "grassland/d3d12/core/adapter.h"
#include "grassland/d3d12/core/dxgi_factory.h"

namespace grassland::d3d12 {

struct DeviceSettings {
  Adapter adapter{nullptr};
};

class Device {
 public:
  explicit Device(const DeviceSettings &settings = {});
  ~Device();

  [[nodiscard]] class Adapter Adapter() const;
  [[nodiscard]] ComPtr<ID3D12Device> Ptr() const;

 private:
  class Adapter adapter_;
  ComPtr<ID3D12Device> device_;
};
}  // namespace grassland::d3d12
