#pragma once
#include "grassland/d3d12/core/command_queue.h"
#include "grassland/d3d12/core/device.h"

namespace grassland::d3d12 {

struct SwapChainSettings {
  HWND hwnd{nullptr};
  int width{0};
  int height{0};
  int frame_count{2};
  DXGI_FORMAT format{DXGI_FORMAT_R8G8B8A8_UNORM};

  SwapChainSettings(HWND hwnd = nullptr) : hwnd(hwnd) {
  }
};

class SwapChain {
 public:
  SwapChain(const DXGIFactory &factory,
            const CommandQueue &command_queue,
            const SwapChainSettings &settings);
  ~SwapChain();

  DXGI_FORMAT Format() const {
    return settings_.format;
  }
  int Width() const {
    return settings_.width;
  }
  int Height() const {
    return settings_.height;
  }
  int FrameCount() const {
    return settings_.frame_count;
  }

  std::vector<ID3D12Resource *> BackBuffers() const {
    std::vector<ID3D12Resource *> back_buffers;
    back_buffers.reserve(back_buffers_.size());
    for (auto &back_buffer : back_buffers_) {
      back_buffers.push_back(back_buffer.Get());
    }
    return back_buffers;
  }

  ID3D12Resource *BackBuffer(int index) const {
    return back_buffers_[index].Get();
  }

  [[nodiscard]] ComPtr<IDXGISwapChain3> Ptr() const;

  void ResizeBuffer(int width, int height, DXGI_FORMAT format);

 private:
  ComPtr<IDXGISwapChain3> swap_chain_;
  std::vector<ComPtr<ID3D12Resource>> back_buffers_;
  SwapChainSettings settings_;
};
}  // namespace grassland::d3d12
