#include <grassland/vulkan/framework/application.h>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "grassland/util/util.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace grassland::vulkan::framework {

namespace {

struct Vertex {
  glm::vec3 pos;
  glm::vec2 tex_coord;
};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
    {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f}}, {{-1.0f, -1.0f, 1.0f}, {0.0f, 0.0f}},
    {{-1.0f, 1.0f, -1.0f}, {0.0f, 1.0f}},  {{-1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
    {{1.0f, -1.0f, -1.0f}, {1.0f, 0.0f}},  {{1.0f, -1.0f, 1.0f}, {1.0f, 0.0f}},
    {{1.0f, 1.0f, -1.0f}, {1.0f, 1.0f}},   {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}};

const std::vector<uint16_t> indices = {
    0b000, 0b010, 0b001, 0b011, 0b001, 0b010, 0b100, 0b101, 0b110,
    0b111, 0b110, 0b101, 0b000, 0b100, 0b010, 0b110, 0b010, 0b100,
    0b001, 0b011, 0b101, 0b111, 0b101, 0b011, 0b000, 0b001, 0b100,
    0b001, 0b101, 0b100, 0b010, 0b110, 0b011, 0b111, 0b011, 0b110};

}  // namespace

Application::Application(int width, int height, const char *title) {
  if (!glfwInit()) {
    LAND_ERROR("GLFW Init failed!");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);

  if (!window_) {
    LAND_ERROR("glfwCreateWindow failed!");
  }

  OnCreate();
}

Application::~Application() {
  if (window_) {
    glfwDestroyWindow(window_);
  }
  glfwTerminate();

  OnDestroy();
}

void Application::Run() {
  OnInit();
  while (!glfwWindowShouldClose(window_)) {
    OnLoop();
    glfwPollEvents();
  }
  OnClose();
}

void Application::OnCreate() {
}

void Application::OnInit() {
  instance_ = std::make_unique<vulkan::Instance>();
  surface_ = std::make_unique<vulkan::Surface>(instance_.get(), window_);
  physical_device_ = std::make_unique<vulkan::PhysicalDevice>(
      vulkan::PickPhysicalDevice(instance_.get(), [](vulkan::PhysicalDevice
                                                         physical_device) {
        int score = 0;
        if (physical_device.IsDiscreteGPU())
          score += 1000;
        score +=
            int(physical_device.GetProperties().limits.maxImageDimension2D);
        return score;
      }).GetHandle());
  spdlog::info("Picked device:");
  physical_device_->PrintDeviceProperties();
  device_ =
      std::make_unique<vulkan::Device>(physical_device_.get(), surface_.get());
  present_queue_ = std::make_unique<vulkan::Queue>(
      device_.get(), physical_device_->PresentFamilyIndex(surface_.get()));
  swapchain_ = std::make_unique<vulkan::Swapchain>(window_, device_.get());

  depth_buffer_image_ = std::make_unique<vulkan::Image>(
      device_.get(), swapchain_->GetExtent().width,
      swapchain_->GetExtent().height,
      vulkan::helper::FindDepthFormat(physical_device_.get()),
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  depth_buffer_image_view_ =
      std::make_unique<vulkan::ImageView>(depth_buffer_image_.get());
  color_attachment_image_ = std::make_unique<vulkan::Image>(
      device_.get(), swapchain_->GetExtent().width,
      swapchain_->GetExtent().height, VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  color_attachment_image_view_ =
      std::make_unique<vulkan::ImageView>(color_attachment_image_.get());

  render_pass_ = std::make_unique<vulkan::RenderPass>(
      device_.get(), color_attachment_image_->GetFormat(),
      depth_buffer_image_->GetFormat());

  vulkan::helper::DescriptorSetLayoutBindings descriptorSetLayoutBindings;
  descriptorSetLayoutBindings.AddBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                         1, VK_SHADER_STAGE_VERTEX_BIT);
  descriptorSetLayoutBindings.AddBinding(
      1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
      VK_SHADER_STAGE_FRAGMENT_BIT);
  descriptor_set_layout_ = std::make_unique<vulkan::DescriptorSetLayout>(
      device_.get(), descriptorSetLayoutBindings);

  pipeline_layout_ = std::make_unique<vulkan::PipelineLayout>(
      device_.get(), descriptor_set_layout_.get());
  vulkan::ShaderModule vert_shader(device_.get(),
                                   "../shaders/shader_base.vert.spv");
  vulkan::ShaderModule frag_shader(device_.get(),
                                   "../shaders/shader_base.frag.spv");
  vulkan::helper::ShaderStages shader_stages;
  shader_stages.AddShaderModule(&vert_shader, VK_SHADER_STAGE_VERTEX_BIT);
  shader_stages.AddShaderModule(&frag_shader, VK_SHADER_STAGE_FRAGMENT_BIT);
  vulkan::helper::VertexInputDescriptions vertex_input_descriptions;
  vertex_input_descriptions.AddBinding(0, sizeof(Vertex));
  vertex_input_descriptions.AddAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                                         offsetof(Vertex, pos));
  vertex_input_descriptions.AddAttribute(0, 1, VK_FORMAT_R32G32_SFLOAT,
                                         offsetof(Vertex, tex_coord));
  graphics_pipeline_ = std::make_unique<vulkan::Pipeline>(
      device_.get(), render_pass_.get(), pipeline_layout_.get(), shader_stages,
      vertex_input_descriptions, true);
  framebuffers_.resize(swapchain_->GetImageCount());
  for (int i = 0; i < swapchain_->GetImageCount(); i++) {
    framebuffers_[i] = std::make_unique<vulkan::Framebuffer>(
        device_.get(), swapchain_->GetExtent().width,
        swapchain_->GetExtent().height, render_pass_.get(),
        std::vector<vulkan::ImageView *>{color_attachment_image_view_.get(),
                                         depth_buffer_image_view_.get()});
  }
  descriptor_pool_ = std::make_unique<vulkan::DescriptorPool>(
      device_.get(), descriptorSetLayoutBindings, kMaxFramesInFlight);
  descriptor_sets_ = std::make_unique<vulkan::DescriptorSets>(
      device_.get(), descriptor_set_layout_.get(), descriptor_pool_.get(),
      kMaxFramesInFlight);

  command_pool_ = std::make_unique<vulkan::CommandPool>(device_.get());
  for (int i = 0; i < kMaxFramesInFlight; i++) {
    command_buffers_.push_back(
        std::make_unique<CommandBuffer>(command_pool_.get()));
  }

  in_flight_fence_.resize(kMaxFramesInFlight);
  image_available_semaphores_.resize(kMaxFramesInFlight);
  render_finished_semaphores_.resize(kMaxFramesInFlight);
  for (int i = 0; i < kMaxFramesInFlight; i++) {
    in_flight_fence_[i] = std::make_unique<vulkan::Fence>(device_.get());
    image_available_semaphores_[i] =
        std::make_unique<vulkan::Semaphore>(device_.get());
    render_finished_semaphores_[i] =
        std::make_unique<vulkan::Semaphore>(device_.get());
  }

  vertex_buffer_ = std::make_unique<vulkan::Buffer>(
      device_.get(), vertices.size() * sizeof(Vertex),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  index_buffer_ = std::make_unique<vulkan::Buffer>(
      device_.get(), indices.size() * sizeof(uint16_t),
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  vertex_buffer_->UploadData(command_pool_.get(),
                             reinterpret_cast<const void *>(vertices.data()));
  index_buffer_->UploadData(command_pool_.get(),
                            reinterpret_cast<const void *>(indices.data()));

  for (size_t i = 0; i < kMaxFramesInFlight; i++) {
    uniform_buffers_.push_back(std::make_unique<vulkan::Buffer>(
        device_.get(), sizeof(UniformBufferObject),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
  }

  int x, y, c;
  auto image_data = stbi_load("../textures/xor_grid.png", &x, &y, &c, 4);
  auto image_buffer = std::make_unique<vulkan::Buffer>(
      device_.get(), x * y * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  std::memcpy(image_buffer->Map(), image_data, x * y * 4);
  image_buffer->Unmap();
  stbi_image_free(image_data);
  image_ = std::make_unique<vulkan::Image>(device_.get(), x, y,
                                           VK_FORMAT_R8G8B8A8_SRGB);
  image_view_ = std::make_unique<vulkan::ImageView>(image_.get());
  sampler_ = std::make_unique<vulkan::Sampler>(device_.get());
  vulkan::UploadImage(command_pool_.get(), image_.get(), image_buffer.get());

  for (size_t i = 0; i < kMaxFramesInFlight; i++) {
    vulkan::helper::UpdateDescriptorWrite(device_->GetHandle(),
                                          descriptor_sets_->GetHandle(i), 0,
                                          uniform_buffers_[i].get());
    vulkan::helper::UpdateDescriptorWrite(device_->GetHandle(),
                                          descriptor_sets_->GetHandle(i), 1,
                                          image_view_.get(), sampler_.get());
  }
}

void Application::OnLoop() {
  OnUpdate();
  OnRender();
}

void Application::OnClose() {
  device_->WaitIdle();

  sampler_.reset();
  image_view_.reset();
  image_.reset();
  vertex_buffer_.reset();
  index_buffer_.reset();

  uniform_buffers_.clear();

  for (int i = 0; i < kMaxFramesInFlight; i++) {
    in_flight_fence_[i].reset();
    image_available_semaphores_[i].reset();
    render_finished_semaphores_[i].reset();
  }
  command_buffers_.clear();
  command_pool_.reset();
  descriptor_sets_.reset();
  descriptor_pool_.reset();
  framebuffers_.clear();
  graphics_pipeline_.reset();
  pipeline_layout_.reset();
  descriptor_set_layout_.reset();
  render_pass_.reset();
  swapchain_.reset();
  depth_buffer_image_view_.reset();
  depth_buffer_image_.reset();
  color_attachment_image_view_.reset();
  color_attachment_image_.reset();
  device_.reset();
  physical_device_.reset();
  surface_.reset();
  instance_.reset();
}

void Application::OnDestroy() {
}

void Application::OnUpdate() {
}

void Application::OnRender() {
  vkWaitForFences(device_->GetHandle(), 1,
                  &in_flight_fence_[currentFrame]->GetHandle(), VK_TRUE,
                  UINT64_MAX);

  VkResult result = vkAcquireNextImageKHR(
      device_->GetHandle(), swapchain_->GetHandle(), UINT64_MAX,
      image_available_semaphores_[currentFrame]->GetHandle(), VK_NULL_HANDLE,
      &imageIndex);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    recreateSwapChain();
    return;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    LAND_ERROR("failed to acquire swap chain image!");
  }

  [&]() {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     currentTime - startTime)
                     .count();

    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view =
        glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f),
                                (float)swapchain_->GetExtent().width /
                                    (float)swapchain_->GetExtent().height,
                                0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    memcpy(uniform_buffers_[currentFrame]->Map(), &ubo, sizeof(ubo));
    uniform_buffers_[currentFrame]->Unmap();
  }();

  vkResetFences(device_->GetHandle(), 1,
                &in_flight_fence_[currentFrame]->GetHandle());

  vkResetCommandBuffer(command_buffers_[currentFrame]->GetHandle(),
                       /*VkCommandBufferResetFlagBits*/ 0);
  recordCommandBuffer(command_buffers_[currentFrame].get());

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore waitSemaphores[] = {
      image_available_semaphores_[currentFrame]->GetHandle()};
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffers_[currentFrame]->GetHandle();

  VkSemaphore signalSemaphores[] = {
      render_finished_semaphores_[currentFrame]->GetHandle()};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  if (vkQueueSubmit(device_->GetGraphicsQueue()->GetHandle(), 1, &submitInfo,
                    in_flight_fence_[currentFrame]->GetHandle()) !=
      VK_SUCCESS) {
    LAND_ERROR("failed to submit draw command buffer!");
  }

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = signalSemaphores;

  VkSwapchainKHR swapChains[] = {swapchain_->GetHandle()};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;

  presentInfo.pImageIndices = &imageIndex;

  result = vkQueuePresentKHR(present_queue_->GetHandle(), &presentInfo);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  currentFrame = (currentFrame + 1) % kMaxFramesInFlight;
}

void Application::recreateSwapChain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window_, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window_, &width, &height);
    glfwWaitEvents();
  }

  device_->WaitIdle();

  framebuffers_.clear();
  swapchain_.reset();
  depth_buffer_image_view_.reset();
  depth_buffer_image_.reset();

  swapchain_ = std::make_unique<vulkan::Swapchain>(window_, device_.get());
  depth_buffer_image_ = std::make_unique<vulkan::Image>(
      device_.get(), swapchain_->GetExtent().width,
      swapchain_->GetExtent().height,
      vulkan::helper::FindDepthFormat(physical_device_.get()),
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  depth_buffer_image_view_ =
      std::make_unique<vulkan::ImageView>(depth_buffer_image_.get());
  framebuffers_.resize(swapchain_->GetImageCount());
  for (int i = 0; i < swapchain_->GetImageCount(); i++) {
    framebuffers_[i] = std::make_unique<vulkan::Framebuffer>(
        device_.get(), swapchain_->GetExtent().width,
        swapchain_->GetExtent().height, render_pass_.get(),
        std::vector<vulkan::ImageView *>{swapchain_->GetImageView(i),
                                         depth_buffer_image_view_.get()});
  }
}
void Application::recordCommandBuffer(CommandBuffer *command_buffer) {
  vulkan::helper::CommandBegin(command_buffer->GetHandle());

  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = render_pass_->GetHandle();
  renderPassInfo.framebuffer = framebuffers_[imageIndex]->GetHandle();
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = swapchain_->GetExtent();

  VkClearValue clearColor[2] = {{{{0.0f, 0.0f, 0.0f, 1.0f}}}};
  clearColor[1].depthStencil.depth = 1.0f;
  renderPassInfo.clearValueCount = 2;
  renderPassInfo.pClearValues = clearColor;

  vkCmdBeginRenderPass(command_buffer->GetHandle(), &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(command_buffer->GetHandle(),
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    graphics_pipeline_->GetHandle());

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)swapchain_->GetExtent().width;
  viewport.height = (float)swapchain_->GetExtent().height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(command_buffer->GetHandle(), 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = swapchain_->GetExtent();
  vkCmdSetScissor(command_buffer->GetHandle(), 0, 1, &scissor);

  VkDeviceSize offsets = 0;
  vkCmdBindVertexBuffers(command_buffer->GetHandle(), 0, 1,
                         &vertex_buffer_->GetHandle(), &offsets);
  vkCmdBindIndexBuffer(command_buffer->GetHandle(), index_buffer_->GetHandle(),
                       0, VK_INDEX_TYPE_UINT16);
  vkCmdBindDescriptorSets(
      command_buffer->GetHandle(), VK_PIPELINE_BIND_POINT_GRAPHICS,
      pipeline_layout_->GetHandle(), 0, 1,
      &descriptor_sets_->GetHandle(currentFrame), 0, nullptr);

  vkCmdDrawIndexed(command_buffer->GetHandle(), indices.size(), 1, 0, 0, 0);

  vkCmdEndRenderPass(command_buffer->GetHandle());

  {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapchain_->GetImage(imageIndex);
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(
        command_buffer->GetHandle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  }

  OutputImage(command_buffer, color_attachment_image_.get());

  {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapchain_->GetImage(imageIndex);
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    barrier.dstAccessMask = 0;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(command_buffer->GetHandle(),
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
  }

  vulkan::helper::CommandEnd(command_buffer->GetHandle());
}

void Application::OutputImage(CommandBuffer *command_buffer, Image *image) {
  image->TransitImageLayout(command_buffer,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_NONE);
  VkImageCopy imageCopy{};
  imageCopy.srcOffset = VkOffset3D{};
  imageCopy.dstOffset = VkOffset3D{};

  imageCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageCopy.dstSubresource.mipLevel = 0;
  imageCopy.dstSubresource.baseArrayLayer = 0;
  imageCopy.dstSubresource.layerCount = 1;
  imageCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageCopy.srcSubresource.mipLevel = 0;
  imageCopy.srcSubresource.baseArrayLayer = 0;
  imageCopy.srcSubresource.layerCount = 1;
  imageCopy.extent = {
      std::min(image->GetWidth(), swapchain_->GetExtent().width),
      std::min(image->GetHeight(), swapchain_->GetExtent().height), 1};
  vkCmdCopyImage(command_buffer->GetHandle(), image->GetHandle(),
                 image->GetImageLayout(), swapchain_->GetImage(imageIndex),
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);
  image->TransitImageLayout(command_buffer,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
}

}  // namespace grassland::vulkan::framework