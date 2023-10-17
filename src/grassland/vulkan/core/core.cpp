#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {
Core::Core(const CoreSettings &settings) : settings_(settings) {
  InstanceSettings instance_settings;
  if (settings.window) {
    instance_settings.EnableSurfaceSupport();
  }
  if (settings.enable_validation_layers) {
    instance_settings.EnableValidationLayers();
  }
  instance_ = std::make_unique<class Instance>(instance_settings);

  // Find the optimal physical device
  std::vector<PhysicalDevice> physical_devices;
  instance_->EnumeratePhysicalDevices(physical_devices);
  PhysicalDevice *physical_device = nullptr;
  uint64_t device_score = 0;
  for (auto &device : physical_devices) {
    if (settings.enable_ray_tracing && !device.SupportRayTracing()) {
      continue;
    }
    uint64_t score = device.Evaluate();
    if (score > device_score) {
      physical_device = &device;
      device_score = score;
    }
  }

  if (!physical_device) {
    LAND_ERROR("No suitable physical device found");
  }

  if (settings.window) {
    // Create the surface
    surface_ =
        std::make_unique<class Surface>(instance_.get(), settings.window);
  }

  // Create the logical device
  device_ = std::make_unique<class Device>(instance_.get(), *physical_device,
                                           surface_.get(),
                                           settings.enable_ray_tracing);

  // Create the command pool
  command_pool_ = std::make_unique<class CommandPool>(device_.get());

  // Create command buffers
  command_buffers_.resize(settings.max_frames_in_flight);
  for (auto &command_buffer : command_buffers_) {
    command_buffer = std::make_unique<class CommandBuffer>(command_pool_.get());
  }

  // Create the swap chain
  if (settings.window) {
    swap_chain_ =
        std::make_unique<class SwapChain>(device_.get(), surface_.get());
  }

  // Create fences and semaphores, semaphores only if not headless
  in_flight_fences_.resize(settings.max_frames_in_flight);
  if (settings.window) {
    image_available_semaphores_.resize(settings.max_frames_in_flight);
    render_finish_semaphores_.resize(settings.max_frames_in_flight);
  }
  for (int i = 0; i < settings.max_frames_in_flight; i++) {
    in_flight_fences_[i] = std::make_unique<class Fence>(device_.get());
    if (settings.window) {
      image_available_semaphores_[i] =
          std::make_unique<class Semaphore>(device_.get());
      render_finish_semaphores_[i] =
          std::make_unique<class Semaphore>(device_.get());
    }
  }
}

Core::~Core() {
  // Wait for device finish
  vkDeviceWaitIdle(device_->Handle());
  // Release all the resources in reverse order of creation
  for (int i = 0; i < settings_.max_frames_in_flight; i++) {
    in_flight_fences_[i].reset();
    if (settings_.window) {
      image_available_semaphores_[i].reset();
      render_finish_semaphores_[i].reset();
    }
  }
  swap_chain_.reset();
  command_buffers_.clear();
  command_pool_.reset();
  device_.reset();
  surface_.reset();
  instance_.reset();
}

void Core::SingleTimeCommands(
    const std::function<void(VkCommandBuffer)> &function) {
  grassland::vulkan::SingleTimeCommands(command_pool_.get(), function);
}

void Core::BeginFrame() {
  // Wait for current frame fence
  VkFence fence = in_flight_fences_[current_frame_]->Handle();
  vkWaitForFences(device_->Handle(), 1, &fence, VK_TRUE, UINT64_MAX);
  vkResetFences(device_->Handle(), 1, &fence);

  if (settings_.window) {
    // Acquire next image
    VkSemaphore image_available_semaphore =
        image_available_semaphores_[current_frame_]->Handle();
    VkResult result = swap_chain_->AcquireNextImage(
        &image_index_, image_available_semaphore, VK_NULL_HANDLE);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      // Recreate swap chain
      RebuildSwapChain();
      LAND_INFO("Swap chain out of date");
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      LAND_ERROR("Failed to acquire swap chain image");
    }
  }

  // Reset command buffer
  VkCommandBuffer command_buffer = command_buffers_[current_frame_]->Handle();

  vkResetCommandBuffer(command_buffer, 0);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
    LAND_ERROR("Failed to begin recording command buffer");
  }
}

void Core::EndFrame() {
  VkCommandBuffer command_buffer = command_buffers_[current_frame_]->Handle();

  if (settings_.window) {
    // Transit current image layout to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, use
    // barrier
    VkImage image = swap_chain_->Images()[image_index_];
    VkImageView image_view = swap_chain_->ImageViews()[image_index_];

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcQueueFamilyIndex = device_->GraphicsQueueFamilyIndex();
    barrier.dstQueueFamilyIndex = device_->PresentQueueFamilyIndex();
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    // add barrier to command buffer
    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);
  }

  if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
    LAND_ERROR("Failed to record command buffer");
  }

  // Submit command buffer
  VkSemaphore image_available_semaphore;
  VkSemaphore render_finish_semaphore;

  if (settings_.window) {
    image_available_semaphore =
        image_available_semaphores_[current_frame_]->Handle();
    render_finish_semaphore =
        render_finish_semaphores_[current_frame_]->Handle();
  }

  VkFence fence = in_flight_fences_[current_frame_]->Handle();
  VkPipelineStageFlags wait_stages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pWaitDstStageMask = wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  if (settings_.window) {
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &image_available_semaphore;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &render_finish_semaphore;
  }

  if (vkQueueSubmit(device_->GraphicsQueue().Handle(), 1, &submit_info,
                    fence) != VK_SUCCESS) {
    LAND_ERROR("Failed to submit draw command buffer");
  }
  // Present image
  if (settings_.window) {
    VkSwapchainKHR swap_chain = swap_chain_->Handle();
    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &render_finish_semaphore;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swap_chain;
    present_info.pImageIndices = &image_index_;
    present_info.pResults = nullptr;
    VkResult result =
        vkQueuePresentKHR(device_->PresentQueue().Handle(), &present_info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      // Recreate swap chain
      RebuildSwapChain();
      LAND_INFO("Swap chain out of date");
    } else if (result != VK_SUCCESS) {
      LAND_ERROR("Failed to present swap chain image");
    }
  }

  current_frame_ = (current_frame_ + 1) % settings_.max_frames_in_flight;
}

void Core::RebuildSwapChain() {
  swap_chain_.reset();
  swap_chain_ =
      std::make_unique<class SwapChain>(device_.get(), surface_.get());
}
}  // namespace grassland::vulkan
