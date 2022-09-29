#include "app.h"

#include <grassland/logging/logging.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace {

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
    {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, 0.0f}},
    {{-1.0f, -1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {{-1.0f, 1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
    {{-1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}},
    {{1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},
    {{1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 1.0f}},
    {{1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, 0.0f}},
    {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}};

const std::vector<uint16_t> indices = {
    0b000, 0b010, 0b001, 0b011, 0b001, 0b010, 0b100, 0b101, 0b110,
    0b111, 0b110, 0b101, 0b000, 0b100, 0b010, 0b110, 0b010, 0b100,
    0b001, 0b011, 0b101, 0b111, 0b101, 0b011, 0b000, 0b001, 0b100,
    0b001, 0b101, 0b100, 0b010, 0b110, 0b011, 0b111, 0b011, 0b110};

}  // namespace

App::App(int width, int height, const char *title) {
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

App::~App() {
  if (window_) {
    glfwDestroyWindow(window_);
  }
  glfwTerminate();

  OnDestroy();
}

void App::Run() {
  OnInit();
  while (!glfwWindowShouldClose(window_)) {
    OnLoop();
    glfwPollEvents();
  }
  OnClose();
}

void App::OnCreate() {
}

void App::OnInit() {
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
  graphics_queue_ = std::make_unique<vulkan::Queue>(
      device_.get(), physical_device_->GraphicsFamilyIndex());
  present_queue_ = std::make_unique<vulkan::Queue>(
      device_.get(), physical_device_->PresentFamilyIndex(surface_.get()));
  swap_chain_ = std::make_unique<vulkan::SwapChain>(window_, device_.get());
  render_pass_ = std::make_unique<vulkan::RenderPass>(device_.get(),
                                                      swap_chain_->GetFormat());

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
  vertex_input_descriptions.AddAttribute(0, 1, VK_FORMAT_R32G32B32_SFLOAT,
                                         offsetof(Vertex, color));
  pipeline_graphics_ = std::make_unique<vulkan::Pipeline>(
      device_.get(), render_pass_.get(), pipeline_layout_.get(), shader_stages,
      vertex_input_descriptions);
  frame_buffers_.resize(swap_chain_->GetImageCount());
  for (int i = 0; i < swap_chain_->GetImageCount(); i++) {
    frame_buffers_[i] = std::make_unique<vulkan::FrameBuffer>(
        device_.get(), swap_chain_->GetExtent().width,
        swap_chain_->GetExtent().height, render_pass_.get(),
        std::vector<vulkan::ImageView *>{swap_chain_->GetImageView(i)});
  }
  descriptor_pool_ = std::make_unique<vulkan::DescriptorPool>(
      device_.get(), descriptorSetLayoutBindings, kMaxFramesInFlight);
  descriptor_sets_ = std::make_unique<vulkan::DescriptorSets>(
      device_.get(), descriptor_set_layout_.get(), descriptor_pool_.get(),
      kMaxFramesInFlight);

  command_pool_ = std::make_unique<vulkan::CommandPool>(device_.get());
  command_buffers_ = std::make_unique<vulkan::CommandBuffers>(
      command_pool_.get(), kMaxFramesInFlight);

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
  vertex_buffer_->UploadData(graphics_queue_.get(), command_pool_.get(),
                             reinterpret_cast<const void *>(vertices.data()));
  index_buffer_->UploadData(graphics_queue_.get(), command_pool_.get(),
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
  vulkan::UploadImage(graphics_queue_.get(), command_pool_.get(), image_.get(),
                      image_buffer.get());

  for (size_t i = 0; i < kMaxFramesInFlight; i++) {
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = uniform_buffers_[i]->GetHandle();
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfo.imageView = image_view_->GetHandle();
    imageInfo.sampler = sampler_->GetHandle();

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptor_sets_->GetHandle(i);
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;
    std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptor_sets_->GetHandle(i);
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfo;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptor_sets_->GetHandle(i);
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device_->GetHandle(),
                           static_cast<uint32_t>(descriptorWrites.size()),
                           descriptorWrites.data(), 0, nullptr);
  }
}

void App::OnLoop() {
  OnUpdate();
  OnRender();
}

void App::OnClose() {
  vkDeviceWaitIdle(device_->GetHandle());

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
  command_buffers_.reset();
  command_pool_.reset();
  descriptor_sets_.reset();
  descriptor_pool_.reset();
  frame_buffers_.clear();
  pipeline_graphics_.reset();
  pipeline_layout_.reset();
  descriptor_set_layout_.reset();
  render_pass_.reset();
  swap_chain_.reset();
  device_.reset();
  physical_device_.reset();
  surface_.reset();
  instance_.reset();
}

void App::OnDestroy() {
}

void App::OnUpdate() {
}

void App::OnRender() {
  vkWaitForFences(device_->GetHandle(), 1,
                  &in_flight_fence_[currentFrame]->GetHandle(), VK_TRUE,
                  UINT64_MAX);

  uint32_t imageIndex;
  VkResult result = vkAcquireNextImageKHR(
      device_->GetHandle(), swap_chain_->GetHandle(), UINT64_MAX,
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
                                (float)swap_chain_->GetExtent().width /
                                    (float)swap_chain_->GetExtent().height,
                                0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    memcpy(uniform_buffers_[currentFrame]->Map(), &ubo, sizeof(ubo));
    uniform_buffers_[currentFrame]->Unmap();
  }();

  vkResetFences(device_->GetHandle(), 1,
                &in_flight_fence_[currentFrame]->GetHandle());

  vkResetCommandBuffer((*command_buffers_)[currentFrame],
                       /*VkCommandBufferResetFlagBits*/ 0);
  recordCommandBuffer((*command_buffers_)[currentFrame], imageIndex);

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
  submitInfo.pCommandBuffers = &(*command_buffers_)[currentFrame];

  VkSemaphore signalSemaphores[] = {
      render_finished_semaphores_[currentFrame]->GetHandle()};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  if (vkQueueSubmit(graphics_queue_->GetHandle(), 1, &submitInfo,
                    in_flight_fence_[currentFrame]->GetHandle()) !=
      VK_SUCCESS) {
    LAND_ERROR("failed to submit draw command buffer!");
  }

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = signalSemaphores;

  VkSwapchainKHR swapChains[] = {swap_chain_->GetHandle()};
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

void App::recreateSwapChain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window_, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window_, &width, &height);
    glfwWaitEvents();
  }

  vkDeviceWaitIdle(device_->GetHandle());

  frame_buffers_.clear();
  swap_chain_.reset();

  swap_chain_ = std::make_unique<vulkan::SwapChain>(window_, device_.get());
  frame_buffers_.resize(swap_chain_->GetImageCount());
  for (int i = 0; i < swap_chain_->GetImageCount(); i++) {
    frame_buffers_[i] = std::make_unique<vulkan::FrameBuffer>(
        device_.get(), swap_chain_->GetExtent().width,
        swap_chain_->GetExtent().height, render_pass_.get(),
        std::vector<vulkan::ImageView *>{swap_chain_->GetImageView(i)});
  }
}
void App::recordCommandBuffer(VkCommandBuffer commandBuffer,
                              uint32_t imageIndex) {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    throw std::runtime_error("failed to begin recording command buffer!");
  }

  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = render_pass_->GetHandle();
  renderPassInfo.framebuffer = frame_buffers_[imageIndex]->GetHandle();
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = swap_chain_->GetExtent();

  VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearColor;

  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline_graphics_->GetHandle());

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)swap_chain_->GetExtent().width;
  viewport.height = (float)swap_chain_->GetExtent().height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = swap_chain_->GetExtent();
  vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

  VkDeviceSize offsets = 0;
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertex_buffer_->GetHandle(),
                         &offsets);
  vkCmdBindIndexBuffer(commandBuffer, index_buffer_->GetHandle(), 0,
                       VK_INDEX_TYPE_UINT16);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipeline_layout_->GetHandle(), 0, 1,
                          &descriptor_sets_->GetHandle(currentFrame), 0,
                          nullptr);

  vkCmdDrawIndexed(commandBuffer, indices.size(), 1, 0, 0, 0);

  vkCmdEndRenderPass(commandBuffer);

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    LAND_ERROR("failed to record command buffer!");
  }
}
