#include <grassland/util/logging.h>
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
  vkFreeMemory(device_->GetHandle(), device_memory_, nullptr);
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

void Image::TransitImageLayout(CommandBuffer *command_buffer,
                               VkImageLayout new_layout,
                               VkPipelineStageFlags new_stage_flags,
                               VkAccessFlags new_access_flags) {
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = handle_;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  barrier.srcAccessMask = access_flags_;
  barrier.dstAccessMask = new_access_flags;

  vkCmdPipelineBarrier(command_buffer->GetHandle(), pipeline_stage_flags_,
                       new_stage_flags, 0, 0, nullptr, 0, nullptr, 1, &barrier);

  image_layout_ = new_layout;
  pipeline_stage_flags_ = new_stage_flags;
  access_flags_ = new_access_flags;
}

void Image::Update(CommandBuffer *command_buffer, Buffer *buffer) {
  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {width_, height_, 1};

  vkCmdCopyBufferToImage(command_buffer->GetHandle(), buffer->GetHandle(),
                         GetHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                         &region);
}

void Image::Retrieve(CommandBuffer *command_buffer, Buffer *buffer) {
  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {width_, height_, 1};

  vkCmdCopyImageToBuffer(command_buffer->GetHandle(), GetHandle(),
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         buffer->GetHandle(), 1, &region);
}
VkImageLayout Image::GetImageLayout() {
  return image_layout_;
}

void UploadImage(Queue *graphics_queue,
                 CommandPool *command_pool,
                 Image *image,
                 Buffer *buffer) {
  auto command_buffer = std::make_unique<CommandBuffer>(command_pool);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buffer->GetHandle(), &beginInfo);

  image->TransitImageLayout(
      command_buffer.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

  image->Update(command_buffer.get(), buffer);

  image->TransitImageLayout(command_buffer.get(), VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_ACCESS_SHADER_READ_BIT);

  vkEndCommandBuffer(command_buffer->GetHandle());

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffer->GetHandle();

  vkQueueSubmit(graphics_queue->GetHandle(), 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue->GetHandle());
}

void DownloadImage(Queue *graphics_queue,
                   CommandPool *command_pool,
                   Image *image,
                   Buffer *buffer) {
  auto command_buffer = std::make_unique<CommandBuffer>(command_pool);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buffer->GetHandle(), &beginInfo);

  image->TransitImageLayout(
      command_buffer.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);

  image->Retrieve(command_buffer.get(), buffer);

  image->TransitImageLayout(command_buffer.get(), VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_ACCESS_SHADER_READ_BIT);

  vkEndCommandBuffer(command_buffer->GetHandle());

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffer->GetHandle();

  vkQueueSubmit(graphics_queue->GetHandle(), 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue->GetHandle());
}

void vulkan::UploadImage(CommandPool *command_pool,
                         Image *image,
                         Buffer *buffer) {
  UploadImage(command_pool->GetDevice()->GetGraphicsQueue(), command_pool,
              image, buffer);
}

void vulkan::DownloadImage(CommandPool *command_pool,
                           Image *image,
                           Buffer *buffer) {
  DownloadImage(command_pool->GetDevice()->GetGraphicsQueue(), command_pool,
                image, buffer);
}
}  // namespace grassland::vulkan
