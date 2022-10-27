#pragma once
#include "grassland/vulkan/framework/core.h"

namespace grassland::vulkan::framework {
class TextureImage {
 public:
  TextureImage(Core *core,
               uint32_t width,
               uint32_t height,
               VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT);
  ~TextureImage();

  void LoadImage(const char *path);
  void LoadHDRImage(const char *path);
  void StoreImage(const char *path);
  void StoreHDRImage(const char *path);

 private:
  Core *core_;
  std::vector<std::unique_ptr<Image>> images_;
  std::vector<std::unique_ptr<ImageView>> image_views_;
};
}  // namespace grassland::vulkan::framework
