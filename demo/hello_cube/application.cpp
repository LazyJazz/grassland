#include "application.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
Application::Application(const std::string &name,
                         int width,
                         int height,
                         bool headless)
    : name_(name) {
  if (!headless) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window_ = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);
  }
  vulkan::CoreSettings core_settings;
  core_settings.window = window_;
  core_ = std::make_unique<vulkan::Core>(core_settings);
}

void Application::Run() {
  OnInit();
  if (window_) {
    while (!glfwWindowShouldClose(window_)) {
      glfwPollEvents();
      OnUpdate();
      OnRender();
    }
  } else {
    while (!application_should_close_) {
      OnUpdate();
      OnRender();
    }
  }
  OnClose();
}

void Application::OnUpdate() {
  static glm::mat4 model = glm::mat4(1.0f);

  // Update uniform buffer
  UniformBufferObject ubo{};
  ubo.model = model;
  ubo.view =
      glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                  glm::vec3(0.0f, 0.0f, 1.0f));
  ubo.proj = glm::perspective(
      glm::radians(45.0f),
      static_cast<float>(core_->SwapChain()->Extent().width) /
          static_cast<float>(core_->SwapChain()->Extent().height),
      0.1f, 10.0f);
  ubo.proj[1][1] *= -1;

  // Map to host buffer first, then copy to device buffer
  void *data = host_uniform_buffers_->Map();
  memcpy(data, &ubo, sizeof(ubo));
  host_uniform_buffers_->Unmap();
  core_->SingleTimeCommands([&](VkCommandBuffer cmd_buffer) {
    vulkan::CopyBuffer(cmd_buffer, host_uniform_buffers_.get(),
                       uniform_buffers_[core_->CurrentFrame()].get(),
                       sizeof(ubo));
  });

  // Calculate time duration of last frame using static timestamp
  static auto last_frame_time = std::chrono::high_resolution_clock::now();
  auto current_frame_time = std::chrono::high_resolution_clock::now();
  float delta_time = std::chrono::duration<float, std::chrono::seconds::period>(
                         current_frame_time - last_frame_time)
                         .count();
  last_frame_time = current_frame_time;

  // Update model matrix by rotate a little bit according to duration of last
  // frame
  model = glm::rotate(model, delta_time * glm::radians(90.0f),
                      glm::vec3(0.0f, 0.0f, 1.0f));

  // Set FPS on window title
  static int frame_count = 0;
  static float time_count = 0.0f;
  frame_count++;
  time_count += delta_time;
  if (time_count >= 1.0f) {
    glfwSetWindowTitle(window_,
                       fmt::format("{} FPS: {}", name_, frame_count).c_str());
    frame_count = 0;
    time_count = 0.0f;
  }
}

void Application::OnRender() {
  core_->BeginFrame();
  auto command_buffer = core_->CommandBuffer();

  // Begin render pass
  VkRenderPassBeginInfo render_pass_begin_info{};
  render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_pass_begin_info.renderPass = render_pass_->Handle();
  render_pass_begin_info.framebuffer = framebuffer_->Handle();
  render_pass_begin_info.renderArea.offset = {0, 0};
  render_pass_begin_info.renderArea.extent = core_->SwapChain()->Extent();
  std::array<VkClearValue, 1> clear_values{};
  clear_values[0].color = {0.6f, 0.7f, 0.8f, 1.0f};
  render_pass_begin_info.clearValueCount =
      static_cast<uint32_t>(clear_values.size());
  render_pass_begin_info.pClearValues = clear_values.data();
  vkCmdBeginRenderPass(command_buffer->Handle(), &render_pass_begin_info,
                       VK_SUBPASS_CONTENTS_INLINE);

  // Bind pipeline
  vkCmdBindPipeline(command_buffer->Handle(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline_->Handle());

  // Bind vertex buffer
  VkBuffer vertex_buffers[] = {vertex_buffer_->Handle()};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(command_buffer->Handle(), 0, 1, vertex_buffers,
                         offsets);

  // Bind index buffer
  vkCmdBindIndexBuffer(command_buffer->Handle(), index_buffer_->Handle(), 0,
                       VK_INDEX_TYPE_UINT16);

  // Bind descriptor sets
  auto descriptor_set = descriptor_sets_[core_->CurrentFrame()]->Handle();
  vkCmdBindDescriptorSets(
      command_buffer->Handle(), VK_PIPELINE_BIND_POINT_GRAPHICS,
      pipeline_layout_->Handle(), 0, 1, &descriptor_set, 0, nullptr);

  // Set viewport and scissor
  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = core_->SwapChain()->Extent().width;
  viewport.height = core_->SwapChain()->Extent().height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(command_buffer->Handle(), 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = core_->SwapChain()->Extent();
  vkCmdSetScissor(command_buffer->Handle(), 0, 1, &scissor);

  // Draw
  vkCmdDrawIndexed(command_buffer->Handle(), 36, 1, 0, 0, 0);

  // End render pass
  vkCmdEndRenderPass(command_buffer->Handle());

  // Transit framebuffer image layout
  vulkan::TransitImageLayout(
      command_buffer->Handle(),
      core_->SwapChain()->Images()[core_->ImageIndex()],
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT);

  // Copy framebuffer image to swap chain image
  VkImageCopy image_copy{};
  image_copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  image_copy.srcSubresource.layerCount = 1;
  image_copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  image_copy.dstSubresource.layerCount = 1;
  image_copy.extent.width = core_->SwapChain()->Extent().width;
  image_copy.extent.height = core_->SwapChain()->Extent().height;
  image_copy.extent.depth = 1;
  vkCmdCopyImage(command_buffer->Handle(), framebuffer_image_->Handle(),
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 core_->SwapChain()->Images()[core_->ImageIndex()],
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_copy);

  // Transit framebuffer image layout
  vulkan::TransitImageLayout(
      command_buffer->Handle(),
      core_->SwapChain()->Images()[core_->ImageIndex()],
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_NONE);

  core_->EndFrame();
}

void Application::OnInit() {
  // Define a hello world triangle
  struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
  };

  std::vector<Vertex> vertices = {
      {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}},
      {{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
      {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}},
      {{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
      {{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
      {{0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},
      {{0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
      {{-0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}},
  };

  std::vector<uint16_t> indices = {
      0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7,
      4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3,
  };

  vertex_buffer_ = std::make_unique<vulkan::Buffer>(
      core_.get(), vertices.size() * sizeof(Vertex),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);
  vulkan::UploadBuffer(vertex_buffer_.get(), vertices.data(),
                       vertices.size() * sizeof(Vertex));
  index_buffer_ = std::make_unique<vulkan::Buffer>(
      core_.get(), indices.size() * sizeof(uint16_t),
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);
  vulkan::UploadBuffer(index_buffer_.get(), indices.data(),
                       indices.size() * sizeof(uint16_t));

  // Create uniform buffers
  uniform_buffers_.resize(core_->MaxFramesInFlight());
  for (size_t i = 0; i < core_->MaxFramesInFlight(); i++) {
    uniform_buffers_[i] = std::make_unique<vulkan::Buffer>(
        core_.get(), sizeof(UniformBufferObject),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
  }

  // Create host uniform buffers
  host_uniform_buffers_ = std::make_unique<vulkan::Buffer>(
      core_.get(), sizeof(UniformBufferObject),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VMA_MEMORY_USAGE_CPU_ONLY);

  spdlog::info("Compiling vertex shader: {}", "hello_cube.vert");
  vertex_shader_ = std::make_unique<vulkan::ShaderModule>(
      core_.get(),
      vulkan::built_in_shaders::GetShaderCompiledSpv("hello_cube.vert"));

  spdlog::info("Compiling fragment shader: {}", "hello_cube.frag");
  fragment_shader_ = std::make_unique<vulkan::ShaderModule>(
      core_.get(),
      vulkan::built_in_shaders::GetShaderCompiledSpv("hello_cube.frag"));

  // Create descriptor pool and sets
  descriptor_pool_ = std::make_unique<vulkan::DescriptorPool>(core_.get());
  descriptor_set_layout_ = std::make_unique<vulkan::DescriptorSetLayout>(
      core_.get(),
      std::vector<VkDescriptorSetLayoutBinding>{VkDescriptorSetLayoutBinding{
          0,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          1,
          VK_SHADER_STAGE_VERTEX_BIT,
          nullptr,
      }});
  descriptor_sets_.resize(core_->MaxFramesInFlight());
  for (size_t i = 0; i < core_->MaxFramesInFlight(); i++) {
    descriptor_sets_[i] = std::make_unique<vulkan::DescriptorSet>(
        core_.get(), descriptor_pool_.get(), descriptor_set_layout_.get());
    // Update descriptor set
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = uniform_buffers_[i]->Handle();
    buffer_info.offset = 0;
    buffer_info.range = sizeof(UniformBufferObject);
    VkWriteDescriptorSet descriptor_write{};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet = descriptor_sets_[i]->Handle();
    descriptor_write.dstBinding = 0;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_write.descriptorCount = 1;
    descriptor_write.pBufferInfo = &buffer_info;
    vkUpdateDescriptorSets(core_->Device()->Handle(), 1, &descriptor_write, 0,
                           nullptr);
  }

  // Create render pass, in C++17 we can use std::vector::data() directly
  std::vector<VkAttachmentDescription> attachment_descriptions = {{
      0,
      core_->SwapChain()->Format(),
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_CLEAR,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
  }};

  std::vector<VkAttachmentReference> color_attachment_references = {
      VkAttachmentReference{
          0,
          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      },
  };

  // Create render pass
  render_pass_ = std::make_unique<vulkan::RenderPass>(
      core_.get(), attachment_descriptions, color_attachment_references);

  // Create pipeline layout
  pipeline_layout_ = std::make_unique<vulkan::PipelineLayout>(
      core_.get(),
      std::vector<VkDescriptorSetLayout>{descriptor_set_layout_->Handle()});

  // Create pipeline
  vulkan::PipelineSettings pipeline_settings(render_pass_.get(),
                                             pipeline_layout_.get());
  pipeline_settings.AddShaderStage(vertex_shader_.get(),
                                   VK_SHADER_STAGE_VERTEX_BIT);
  pipeline_settings.AddShaderStage(fragment_shader_.get(),
                                   VK_SHADER_STAGE_FRAGMENT_BIT);
  pipeline_settings.AddInputBinding(0, sizeof(Vertex));
  pipeline_settings.AddInputAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                                      offsetof(Vertex, pos));
  pipeline_settings.AddInputAttribute(0, 1, VK_FORMAT_R32G32B32_SFLOAT,
                                      offsetof(Vertex, color));
  pipeline_settings.SetPrimitiveTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  pipeline_settings.SetCullMode(VK_CULL_MODE_BACK_BIT);

  pipeline_ =
      std::make_unique<vulkan::Pipeline>(core_.get(), pipeline_settings);

  framebuffer_image_ = std::make_unique<vulkan::Image>(
      core_.get(), core_->SwapChain()->Format(), core_->SwapChain()->Extent(),
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      VK_IMAGE_ASPECT_COLOR_BIT, VK_SAMPLE_COUNT_1_BIT);

  framebuffer_ = std::make_unique<vulkan::Framebuffer>(
      core_.get(), core_->SwapChain()->Extent(), render_pass_->Handle(),
      std::vector<VkImageView>{framebuffer_image_->ImageView()});
}

void Application::OnClose() {
  // Release resources in reverse order of creation
  core_->Device()->WaitIdle();
  framebuffer_image_.reset();
  framebuffer_.reset();
  pipeline_.reset();
  pipeline_layout_.reset();
  render_pass_.reset();
  descriptor_sets_.clear();
  descriptor_set_layout_.reset();
  descriptor_pool_.reset();
  vertex_shader_.reset();
  fragment_shader_.reset();
  host_uniform_buffers_.reset();
  uniform_buffers_.clear();
  index_buffer_.reset();
  vertex_buffer_.reset();
  core_.reset();
  if (window_) {
    glfwDestroyWindow(window_);
    glfwTerminate();
  }
}
