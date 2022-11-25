#pragma once
#include "grassland/vulkan/gui/util.h"

namespace grassland::vulkan::gui {
class Manager {
 public:
  Manager(framework::Core *core);
  void BindFrameTexture(framework::TextureImage *frame_texture);

  void Refresh();

 private:
  framework::Core *core_{nullptr};
  std::unique_ptr<framework::RenderNode> render_node_;
  std::unique_ptr<framework::TextureImage> frame_;
};
}  // namespace grassland::vulkan::gui
