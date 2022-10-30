#include <grassland/util/util.h>
#include <grassland/vulkan/framework/core.h>
#include <grassland/vulkan/framework/texture_image.h>

namespace grassland::vulkan::framework {

Core::Core(const CoreSettings &core_settings) {
  core_settings_ = core_settings;
  if (core_settings_.has_window) {
    if (!glfwInit()) {
      LAND_ERROR("[Vulkan] GLFW initialization failed.");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window_ = glfwCreateWindow(int(core_settings_.window_width),
                               int(core_settings_.window_height),
                               core_settings_.window_title, nullptr, nullptr);
    if (!window_) {
      LAND_ERROR("[Vulkan] GLFW create window failed.");
    }
  }

  instance_ = std::make_unique<Instance>(core_settings_.has_window,
                                         core_settings_.validation_layer);

  if (core_settings_.has_window) {
    surface_ = std::make_unique<Surface>(instance_.get(), window_);
  }

  physical_device_ = std::make_unique<PhysicalDevice>(PickPhysicalDevice(
      instance_.get(), [&core_settings](PhysicalDevice physical_device) {
        if (core_settings.raytracing_pipeline_required &&
            !physical_device.HasRayTracingPipeline()) {
          return -1;
        }
        int score = 0;
        if (core_settings.has_window &&
            !physical_device.HasPresentationSupport()) {
          return -1;
        }
        score += int(physical_device.DeviceMemorySize() >> 20);
        if (physical_device.IsDiscreteGPU()) {
          score *= 2;
        }
        return score;
      }));

  physical_device_->PrintDeviceProperties();

  if (!physical_device_->GetHandle()) {
    LAND_ERROR("[Vulkan] failed to find available device.");
  }

  device_ = std::make_unique<Device>(
      physical_device_.get(),
      core_settings_.has_window ? surface_.get() : nullptr,
      core_settings_.raytracing_pipeline_required
          ? std::vector<
                const char *>{VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                              VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                              VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME}
          : std::vector<const char *>{},
      core_settings_.validation_layer);

  if (core_settings_.has_window) {
    swapchain_ = std::make_unique<Swapchain>(device_.get(), window_);
  }

  command_pool_ = std::make_unique<CommandPool>(device_.get());
  command_buffers_.clear();
  for (int i = 0; i < core_settings_.frames_in_flight; i++) {
    command_buffers_.push_back(
        std::make_unique<CommandBuffer>(command_pool_.get()));
  }

  for (int i = 0; i < core_settings_.frames_in_flight; i++) {
    in_flight_fences_.push_back(std::make_unique<Fence>(device_.get()));
    image_available_semaphores_.push_back(
        std::make_unique<Semaphore>(device_.get()));
    render_finish_semaphores_.push_back(
        std::make_unique<Semaphore>(device_.get()));
  }
}

Core::~Core() {
  device_->WaitIdle();
  render_finish_semaphores_.clear();
  image_available_semaphores_.clear();
  in_flight_fences_.clear();
  command_buffers_.clear();
  command_pool_.reset();
  swapchain_.reset();
  device_.reset();
  physical_device_.reset();
  if (core_settings_.has_window) {
    surface_.reset();
    glfwDestroyWindow(window_);
    glfwTerminate();
  }
  instance_.reset();
}

const CoreSettings &Core::GetCoreSettings() const {
  return core_settings_;
}

Instance *Core::GetInstance() const {
  return instance_.get();
}

PhysicalDevice *Core::GetPhysicalDevice() const {
  return physical_device_.get();
}

Device *Core::GetDevice() const {
  return device_.get();
}

Surface *Core::GetSurface() const {
  return surface_.get();
}

Swapchain *Core::GetSwapchain() const {
  return swapchain_.get();
}

CommandPool *Core::GetCommandPool() const {
  return command_pool_.get();
}

GLFWwindow *Core::GetWindow() const {
  return window_;
}

CommandBuffer *Core::GetCommandBuffer(int frame_index) const {
  return command_buffers_[frame_index].get();
}

CommandBuffer *Core::GetCommandBuffer() const {
  return GetCommandBuffer(frame_index_);
}

int Core::GetCurrentFrameIndex() const {
  return frame_index_;
}

void Core::BeginCommandRecord() {
  frame_index_ = (frame_index_ + 1) % core_settings_.frames_in_flight;
  vkWaitForFences(device_->GetHandle(), 1,
                  &in_flight_fences_[frame_index_]->GetHandle(), VK_TRUE,
                  UINT64_MAX);

  if (core_settings_.has_window) {
    VkResult result = vkAcquireNextImageKHR(
        device_->GetHandle(), swapchain_->GetHandle(), UINT64_MAX,
        image_available_semaphores_[frame_index_]->GetHandle(), VK_NULL_HANDLE,
        &current_image_index);
  }

  vkResetFences(device_->GetHandle(), 1,
                &in_flight_fences_[frame_index_]->GetHandle());

  vkResetCommandBuffer(command_buffers_[frame_index_]->GetHandle(),
                       /*VkCommandBufferResetFlagBits*/ 0);

  vulkan::helper::CommandBegin(command_buffers_[frame_index_]->GetHandle());
}

void Core::EndCommandRecordAndSubmit() {
  vulkan::helper::CommandEnd(command_buffers_[frame_index_]->GetHandle());

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore waitSemaphores[] = {
      image_available_semaphores_[frame_index_]->GetHandle()};
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  VkSemaphore signalSemaphores[] = {
      render_finish_semaphores_[frame_index_]->GetHandle()};

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffers_[frame_index_]->GetHandle();

  if (core_settings_.has_window) {
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
  } else {
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.pWaitDstStageMask = nullptr;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;
  }

  if (vkQueueSubmit(device_->GetGraphicsQueue()->GetHandle(), 1, &submitInfo,
                    in_flight_fences_[frame_index_]->GetHandle()) !=
      VK_SUCCESS) {
    LAND_ERROR("failed to submit draw command buffer!");
  }

  if (core_settings_.has_window) {
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {swapchain_->GetHandle()};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &current_image_index;

    auto result = vkQueuePresentKHR(device_->GetPresentQueue()->GetHandle(),
                                    &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      LAND_ERROR("[Vulkan] present failed.");
    }
  }
}

void Core::Output(TextureImage *texture_image) {
  TransitImageLayout(GetCommandBuffer()->GetHandle(),
                     swapchain_->GetImage(current_image_index),
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
  TransitImageLayout(
      GetCommandBuffer()->GetHandle(), texture_image->GetImage()->GetHandle(),
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageCopy imageCopy{};
  imageCopy.srcOffset = VkOffset3D{0, 0, 0};
  imageCopy.dstOffset = VkOffset3D{0, 0, 0};

  imageCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageCopy.dstSubresource.mipLevel = 0;
  imageCopy.dstSubresource.baseArrayLayer = 0;
  imageCopy.dstSubresource.layerCount = 1;
  imageCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageCopy.srcSubresource.mipLevel = 0;
  imageCopy.srcSubresource.baseArrayLayer = 0;
  imageCopy.srcSubresource.layerCount = 1;
  imageCopy.extent = {swapchain_->GetExtent().width,
                      swapchain_->GetExtent().height, 1};
  vkCmdCopyImage(GetCommandBuffer()->GetHandle(),
                 texture_image->GetImage()->GetHandle(),
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 swapchain_->GetImage(current_image_index),
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);
  TransitImageLayout(GetCommandBuffer()->GetHandle(),
                     swapchain_->GetImage(current_image_index),
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_NONE,
                     VK_IMAGE_ASPECT_COLOR_BIT);
  TransitImageLayout(
      GetCommandBuffer()->GetHandle(), texture_image->GetImage()->GetHandle(),
      VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      VK_ACCESS_NONE, VK_IMAGE_ASPECT_COLOR_BIT);
}

}  // namespace grassland::vulkan::framework
