#include "app.h"

#include <grassland/logging/logging.h>

#include <glm/glm.hpp>

namespace {

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;

  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    return attributeDescriptions;
  }
};

const std::vector<Vertex> vertices = {{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                      {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
                                      {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
                                      {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

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
  pipeline_layout_ = std::make_unique<vulkan::PipelineLayout>(device_.get());
  vulkan::ShaderModule vert_shader(device_.get(),
                                   "../shaders/shader_base.vert.spv");
  vulkan::ShaderModule frag_shader(device_.get(),
                                   "../shaders/shader_base.frag.spv");
  vulkan::helper::ShaderStages shader_stages;
  shader_stages.AddShaderModule(&vert_shader, VK_SHADER_STAGE_VERTEX_BIT);
  shader_stages.AddShaderModule(&frag_shader, VK_SHADER_STAGE_FRAGMENT_BIT);
  vulkan::helper::VertexInputDescriptions vertex_input_descriptions;
  vertex_input_descriptions.AddBinding(0, sizeof(Vertex));
  vertex_input_descriptions.AddAttribute(0, 0, VK_FORMAT_R32G32_SFLOAT,
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

  vertex_buffer = std::make_unique<vulkan::Buffer>(
      device_.get(), vertices.size() * sizeof(Vertex),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  index_buffer = std::make_unique<vulkan::Buffer>(
      device_.get(), indices.size() * sizeof(uint16_t),
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  vertex_buffer->UploadData(graphics_queue_.get(), command_pool_.get(),
                            reinterpret_cast<const void *>(vertices.data()));
  index_buffer->UploadData(graphics_queue_.get(), command_pool_.get(),
                           reinterpret_cast<const void *>(indices.data()));
}

void App::OnLoop() {
  OnUpdate();
  OnRender();
}

void App::OnClose() {
  vkDeviceWaitIdle(device_->GetHandle());

  vertex_buffer.reset();
  index_buffer.reset();

  for (int i = 0; i < kMaxFramesInFlight; i++) {
    in_flight_fence_[i].reset();
    image_available_semaphores_[i].reset();
    render_finished_semaphores_[i].reset();
  }
  command_buffers_.reset();
  command_pool_.reset();
  frame_buffers_.clear();
  pipeline_graphics_.reset();
  pipeline_layout_.reset();
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
  static int currentFrame = 0;
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
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertex_buffer->GetHandle(),
                         &offsets);
  vkCmdBindIndexBuffer(commandBuffer, index_buffer->GetHandle(), 0,
                       VK_INDEX_TYPE_UINT16);
  // vkCmdDraw(commandBuffer, 3, 1, 0, 0);
  vkCmdDrawIndexed(commandBuffer, 6, 1, 0, 0, 0);

  vkCmdEndRenderPass(commandBuffer);

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    LAND_ERROR("failed to record command buffer!");
  }
}
