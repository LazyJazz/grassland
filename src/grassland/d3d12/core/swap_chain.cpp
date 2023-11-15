#include "grassland/d3d12/core/swap_chain.h"

namespace grassland::d3d12 {

SwapChain::SwapChain(const DXGIFactory &factory,
                     const CommandQueue &command_queue,
                     const SwapChainSettings &settings)
    : settings_(settings) {
  DXGI_SWAP_CHAIN_DESC1 swap_chain_desc{};
  swap_chain_desc.Width = settings_.width;
  swap_chain_desc.Height = settings_.height;
  swap_chain_desc.Format = settings_.format;

  swap_chain_desc.BufferCount = settings_.frame_count;
  swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  swap_chain_desc.SampleDesc.Count = 1;

  ComPtr<IDXGISwapChain1> swap_chain1;
  ThrowIfFailed(factory.Ptr()->CreateSwapChainForHwnd(
                    command_queue.Ptr().Get(), settings_.hwnd, &swap_chain_desc,
                    nullptr, nullptr, &swap_chain1),
                "Failed to create swap chain");

  ThrowIfFailed(factory.Ptr()->MakeWindowAssociation(settings_.hwnd,
                                                     DXGI_MWA_NO_ALT_ENTER),
                "Failed to make window association");

  ThrowIfFailed(swap_chain1.As(&swap_chain_), "Failed to cast swap chain");

  // Create Render Targets
  back_buffers_.resize(settings_.frame_count);
  for (int i = 0; i < settings_.frame_count; ++i) {
    ThrowIfFailed(swap_chain_->GetBuffer(i, IID_PPV_ARGS(&back_buffers_[i])),
                  "Failed to get swap chain buffer");
  }

  // Get width and height if zero
  if (settings_.width == 0 || settings_.height == 0) {
    DXGI_SWAP_CHAIN_DESC desc;
    ThrowIfFailed(swap_chain_->GetDesc(&desc), "Failed to get swap chain desc");
    settings_.width = desc.BufferDesc.Width;
    settings_.height = desc.BufferDesc.Height;
  }
}

SwapChain::~SwapChain() = default;

ComPtr<IDXGISwapChain3> SwapChain::Ptr() const {
  return swap_chain_;
}

void SwapChain::ResizeBuffer(int width, int height, DXGI_FORMAT format) {
  settings_.width = width;
  settings_.height = height;
  if (format != DXGI_FORMAT_UNKNOWN) {
    settings_.format = format;
  }

  back_buffers_.clear();

  ThrowIfFailed(
      swap_chain_->ResizeBuffers(settings_.frame_count, settings_.width,
                                 settings_.height, settings_.format, 0),
      "Failed to resize buffers");
  back_buffers_.resize(settings_.frame_count);
  for (int i = 0; i < settings_.frame_count; ++i) {
    ThrowIfFailed(swap_chain_->GetBuffer(i, IID_PPV_ARGS(&back_buffers_[i])),
                  "Failed to get swap chain buffer");
  }

  // Get width and height if zero
  if (settings_.width == 0 || settings_.height == 0) {
    DXGI_SWAP_CHAIN_DESC desc;
    ThrowIfFailed(swap_chain_->GetDesc(&desc), "Failed to get swap chain desc");
    settings_.width = desc.BufferDesc.Width;
    settings_.height = desc.BufferDesc.Height;
  }
}

}  // namespace grassland::d3d12
