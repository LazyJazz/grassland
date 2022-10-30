#include <grassland/vulkan/framework/render_node.h>

namespace grassland::vulkan::framework {
RenderNode::RenderNode(Core *core) {
  core_ = core;
}
TextureImage *RenderNode::GetColorImage(int color_image_index) const {
  return color_attachment_textures_[color_image_index].get();
}
TextureImage *RenderNode::GetDepthImage() const {
  return depth_buffer_texture_.get();
}
}  // namespace grassland::vulkan::framework
