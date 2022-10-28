#include <grassland/util/util.h>
#include <grassland/vulkan/buffer.h>
#include <grassland/vulkan/framework/texture_image.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace grassland::vulkan::framework {
TextureImage::TextureImage(Core *core,
                           uint32_t width,
                           uint32_t height,
                           VkFormat format,
                           VkImageUsageFlags usage) {
  core_ = core;
  image_ =
      std::make_unique<Image>(core_->GetDevice(), width, height, format, usage);
}

TextureImage::~TextureImage() = default;

Core *TextureImage::GetCore() const {
  return core_;
}

void TextureImage::ReadImage(const char *path) {
  int x, y, c;
  auto buffer = stbi_load(path, &x, &y, &c, 4);
  if (!buffer) {
    LAND_ERROR("[Vulkan] failed to load image \"{}\".", path);
  }
  std::unique_ptr<Buffer> image_buffer = std::make_unique<Buffer>(
      core_->GetDevice(), sizeof(uint8_t) * x * y * 4,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  std::memcpy(image_buffer->Map(), buffer, sizeof(uint8_t) * x * y * 4);
  image_buffer->Unmap();
  stbi_image_free(buffer);

  std::unique_ptr<Image> image = std::make_unique<Image>(
      core_->GetDevice(), x, y, VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  UploadImage(core_->GetCommandPool(), image.get(), image_buffer.get());

  CopyImage(core_->GetCommandPool(), image.get(), image_.get());
}

void TextureImage::WriteImage(const char *path) {
  std::vector<uint8_t> buffer(image_->GetWidth() * image_->GetHeight() * 4);
  std::unique_ptr<Image> image = std::make_unique<Image>(
      core_->GetDevice(), image_->GetWidth(), image_->GetHeight(),
      VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  CopyImage(core_->GetCommandPool(), image_.get(), image.get());
  std::unique_ptr<Buffer> image_buffer = std::make_unique<Buffer>(
      core_->GetDevice(),
      sizeof(uint8_t) * image_->GetWidth() * image_->GetHeight() * 4,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  DownloadImage(core_->GetCommandPool(), image.get(), image_buffer.get());
  std::memcpy(buffer.data(), image_buffer->Map(),
              sizeof(uint8_t) * image_->GetWidth() * image_->GetHeight() * 4);
  image_buffer->Unmap();
  auto tail = path + std::strlen(path);
  while (*tail != '.' && tail > path) {
    tail--;
  }
  std::string suffix{tail};
  if (suffix == ".bmp") {
    stbi_write_bmp(path, int(image_->GetWidth()), int(image_->GetHeight()), 4,
                   buffer.data());
  } else if (suffix == ".jpg" || suffix == ".jpeg") {
    stbi_write_jpg(path, int(image_->GetWidth()), int(image_->GetHeight()), 4,
                   buffer.data(), 100);
  } else {
    stbi_write_png(path, int(image_->GetWidth()), int(image_->GetHeight()), 4,
                   buffer.data(), int(image_->GetWidth()) * 4);
  }
}
}  // namespace grassland::vulkan::framework
