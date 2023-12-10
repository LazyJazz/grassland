#include "grassland/d3d12/core/adapter.h"

namespace grassland::d3d12 {
Adapter::Adapter(ComPtr<IDXGIAdapter1> adaptor) : adaptor_(std::move(adaptor)) {
}

Adapter::~Adapter() = default;

DXGI_ADAPTER_DESC1 Adapter::GetDesc() const {
  DXGI_ADAPTER_DESC1 desc;
  adaptor_->GetDesc1(&desc);
  return desc;
}

bool Adapter::SupportRayTracing() const {
  // Check Whether the adapter supports ray tracing.
  ComPtr<ID3D12Device> device;
  HRESULT hr = D3D12CreateDevice(adaptor_.Get(), D3D_FEATURE_LEVEL_11_0,
                                 _uuidof(ID3D12Device), &device);
  if (FAILED(hr)) {
    return false;
  }
  ComPtr<ID3D12Device5> device5;
  hr = device.As(&device5);
  if (FAILED(hr)) {
    return false;
  }
  D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5;
  hr = device5->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5,
                                    sizeof(options5));
  if (FAILED(hr)) {
    return false;
  }
  return options5.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
}

bool Adapter::SupportMeshShader() const {
  // Check Whether the adapter supports mesh shader.
  ComPtr<ID3D12Device> device;
  HRESULT hr = D3D12CreateDevice(adaptor_.Get(), D3D_FEATURE_LEVEL_11_0,
                                 _uuidof(ID3D12Device), &device);
  if (FAILED(hr)) {
    return false;
  }

  D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7;
  hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options7,
                                   sizeof(options7));
  if (FAILED(hr)) {
    return false;
  }
  return options7.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED;
}

uint64_t Adapter::Evaluate() const {
  DXGI_ADAPTER_DESC1 desc = GetDesc();
  uint64_t score = 1000;
  if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
    return 0;
  }
  score += desc.DedicatedVideoMemory / 1024 / 1024;
  return score;
}

ComPtr<IDXGIAdapter1> Adapter::Ptr() const {
  return adaptor_;
}

std::string Adapter::Description() const {
  return util::WideStringToU8String(GetDesc().Description);
}
}  // namespace grassland::d3d12
