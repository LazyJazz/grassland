#include <grassland/vulkan/image.h>

namespace grassland::vulkan {

Image::Image() : handle_{} {
}

Image::Image(const VkImage &image) {
  handle_ = image;
}

Image::~Image() = default;

}  // namespace grassland::vulkan
