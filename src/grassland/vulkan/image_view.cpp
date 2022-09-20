#include <grassland/vulkan/image_view.h>

namespace grassland::vulkan {

ImageView::ImageView(const VkImageView &image_view) {
  handle_ = image_view;
}

ImageView::ImageView() : handle_{} {
}

ImageView::~ImageView() = default;

}  // namespace grassland::vulkan
