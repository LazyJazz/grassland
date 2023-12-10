#include "grassland/d3d12/core/dxgi_factory.h"

#include "grassland/d3d12/core/adapter.h"

namespace grassland::d3d12 {

DXGIFactory::DXGIFactory(const DXGIFactorySettings &settings) {
  UINT flags = 0;
  if (settings.enable_debug_layer) {
    ComPtr<ID3D12Debug> debug_controller;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller)))) {
      debug_controller->EnableDebugLayer();
      flags |= DXGI_CREATE_FACTORY_DEBUG;
      ComPtr<ID3D12Debug1> debug_controller1;
      if (SUCCEEDED(debug_controller->QueryInterface(
              IID_PPV_ARGS(&debug_controller1)))) {
        debug_controller1->SetEnableGPUBasedValidation(true);
      }
    }
  }

  ThrowIfFailed(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory_)),
                "Failed to create DXGI factory");
}

ComPtr<IDXGIFactory4> DXGIFactory::Ptr() const {
  return factory_;
}

std::vector<class Adapter> DXGIFactory::EnumerateAdapters() const {
  std::vector<class Adapter> adapters;
  ComPtr<IDXGIAdapter1> adaptor;
  for (UINT i = 0; factory_->EnumAdapters1(i, &adaptor) != DXGI_ERROR_NOT_FOUND;
       ++i) {
    adapters.emplace_back(adaptor);
  }
  return adapters;
}

DXGIFactory::~DXGIFactory() = default;

}  // namespace grassland::d3d12
