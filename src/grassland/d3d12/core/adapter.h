#pragma once
#include "grassland/d3d12/core/dxgi_factory.h"

namespace grassland::d3d12 {
class Adapter {
 public:
  explicit Adapter(ComPtr<IDXGIAdapter1> adaptor = nullptr);
  ~Adapter();

  [[nodiscard]] DXGI_ADAPTER_DESC1 GetDesc() const;
  [[nodiscard]] std::string Description() const;

  [[nodiscard]] bool SupportRayTracing() const;
  [[nodiscard]] bool SupportMeshShader() const;

  [[nodiscard]] uint64_t Evaluate() const;

  [[nodiscard]] ComPtr<IDXGIAdapter1> Ptr() const;

 private:
  ComPtr<IDXGIAdapter1> adaptor_;
};
}  // namespace grassland::d3d12
