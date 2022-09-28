#include <grassland/logging/logging.h>
#include <grassland/vulkan/image.h>

namespace grassland::vulkan {

namespace {
uint32_t FindMemoryType(PhysicalDevice *physical_device,
                        uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physical_device->GetHandle(),
                                      &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  LAND_ERROR("[Vulkan] failed to find suitable memory type!");
}
}  // namespace

Image::Image(Device *device,
             uint32_t width,
             uint32_t height,
             VkFormat format,
             VkImageUsageFlags usage,
             VkImageTiling tiling)
    : handle_{} {
  device_ = device;
  width_ = width;
  height_ = height;
  format_ = format;

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateImage(device_->GetHandle(), &imageInfo, nullptr, &handle_) !=
      VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create image!");
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device_->GetHandle(), handle_, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = FindMemoryType(
      device_->GetPhysicalDevice(), memRequirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (vkAllocateMemory(device_->GetHandle(), &allocInfo, nullptr,
                       &device_memory_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to allocate image memory!");
  }

  vkBindImageMemory(device_->GetHandle(), handle_, device_memory_, 0);
}

Image::~Image() {
  vkDestroyImage(device_->GetHandle(), handle_, nullptr);
}

VkFormat Image::GetFormat() const {
  return format_;
}
uint32_t Image::GetWidth() const {
  return width_;
}
uint32_t Image::GetHeight() const {
  return height_;
}

}  // namespace grassland::vulkan
