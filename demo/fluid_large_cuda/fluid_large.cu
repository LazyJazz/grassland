#include "fluid_large.cuh"
#include "glm/gtc/matrix_transform.hpp"
#include "random"

FluidLarge::FluidLarge(const char *title,
                       int width,
                       int height,
                       const PhysicSettings &physic_settings) {
  vulkan_legacy::framework::CoreSettings core_settings;
  core_settings.window_width = width;
  core_settings.window_height = height;
  core_settings.window_title = title;
  core_ = std::make_unique<vulkan_legacy::framework::Core>(core_settings);
  color_frame_ = std::make_unique<vulkan_legacy::framework::TextureImage>(
      core_.get(), width, height, VK_FORMAT_B8G8R8A8_UNORM);
  depth_frame_ = std::make_unique<vulkan_legacy::framework::TextureImage>(
      core_.get(), width, height, VK_FORMAT_D32_SFLOAT,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  stencil_frame_ = std::make_unique<vulkan_legacy::framework::TextureImage>(
      core_.get(), width, height, VK_FORMAT_R32_UINT);

  global_uniform_buffer_ = std::make_unique<
      vulkan_legacy::framework::StaticBuffer<GlobalUniformObject>>(core_.get(),
                                                                   1);
  instance_info_buffer_ =
      std::make_unique<vulkan_legacy::framework::StaticBuffer<InstanceInfo>>(
          core_.get(), physic_settings.num_particle);
  instance_infos_.resize(physic_settings.num_particle);

  render_node_ =
      std::make_unique<vulkan_legacy::framework::RenderNode>(core_.get());
  render_node_->AddShader("particle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
  render_node_->AddShader("particle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->VertexInput({VK_FORMAT_R32G32_SFLOAT});
  render_node_->AddColorAttachment(color_frame_.get());
  render_node_->AddDepthAttachment(depth_frame_.get());
  render_node_->AddUniformBinding(global_uniform_buffer_.get(),
                                  VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddBufferBinding(instance_info_buffer_.get(),
                                 VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->BuildRenderNode();

  core_->SetFrameSizeCallback([this](int width, int height) {
    color_frame_->Resize(width, height);
    depth_frame_->Resize(width, height);
    stencil_frame_->Resize(width, height);
    render_node_->BuildRenderNode();

    GlobalUniformObject ubo{};
    ubo.world = glm::lookAt(SceneRange() * 0.5f - glm::vec3{0.0f, 0.0f, 2.0f},
                            SceneRange() * 0.5f, glm::vec3{0.0f, 1.0f, 0.0f});
    ubo.camera = glm::perspectiveZO(glm::radians(60.0f),
                                    (float)width / (float)height, 0.1f, 10.0f);
    global_uniform_buffer_->Upload(&ubo, sizeof(ubo));
  });

  std::vector<Vertex> vertices = {
      {{-1.0f, -1.0f}}, {{-1.0f, 1.0f}}, {{1.0f, -1.0f}}, {{1.0f, 1.0f}}};
  std::vector<uint32_t> indices = {0, 1, 2, 1, 2, 3};

  vertex_buffer_ =
      std::make_unique<vulkan_legacy::framework::StaticBuffer<Vertex>>(
          core_.get(), vertices.size());
  index_buffer_ =
      std::make_unique<vulkan_legacy::framework::StaticBuffer<uint32_t>>(
          core_.get(), indices.size());
  vertex_buffer_->Upload(vertices.data(), vertices.size() * sizeof(Vertex));
  index_buffer_->Upload(indices.data(), indices.size() * sizeof(uint32_t));

  GlobalUniformObject ubo{};
  ubo.world = glm::lookAt(SceneRange() * 0.5f - glm::vec3{0.0f, 0.0f, 2.0f},
                          SceneRange() * 0.5f, glm::vec3{0.0f, 1.0f, 0.0f});
  ubo.camera = glm::perspectiveZO(glm::radians(60.0f),
                                  (float)width / (float)height, 0.1f, 10.0f);
  global_uniform_buffer_->Upload(&ubo, sizeof(ubo));

  physic_solver_ = std::make_unique<PhysicSolver>(physic_settings);
}

void FluidLarge::Run() {
  OnInit();
  while (!glfwWindowShouldClose(core_->GetWindow())) {
    OnLoop();
    glfwPollEvents();
  }
  core_->GetDevice()->WaitIdle();
  OnClose();
}

void FluidLarge::OnInit() {
}

void FluidLarge::OnLoop() {
  OnUpdate();
  OnRender();
}

void FluidLarge::OnClose() {
}

void FluidLarge::OnRender() {
  core_->BeginCommandRecord();
  color_frame_->ClearColor({0.6f, 0.7f, 0.8f, 1.0f});
  depth_frame_->ClearDepth({1.0f, 0});
  render_node_->Draw(vertex_buffer_.get(), index_buffer_.get(),
                     index_buffer_->Size(), 0, instance_infos_.size());
  core_->Output(color_frame_.get());
  core_->EndCommandRecordAndSubmit();
  LAND_INFO("Loop...");
}

void FluidLarge::OnUpdate() {
  for (int i = 0; i < 1; i++) {
    physic_solver_->UpdateStep();
  }
  physic_solver_->GetInstanceInfoArray(instance_infos_.data());
  instance_info_buffer_->Upload(instance_infos_.data(),
                                sizeof(InstanceInfo) * instance_infos_.size());
}
