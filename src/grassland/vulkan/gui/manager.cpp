#include "grassland/vulkan/gui/manager.h"

#include "glm/gtc/matrix_transform.hpp"
#include "grassland/vulkan/gui/model.h"
#include "grassland/vulkan/gui/window.h"

namespace grassland::vulkan::gui {

Manager::Manager(framework::Core *core) {
  core_ = core;

  global_uniform_buffer_ =
      std::make_unique<framework::StaticBuffer<GlobalUniformObject>>(core_, 1);
  model_uniform_buffer_ =
      std::make_unique<framework::DynamicBuffer<ModelUniformObject>>(core_,
                                                                     16384);

  core_->SetFrameSizeCallback([this](int width, int height) {
    UpdateGlobalObject(width, height);
    frame_->Resize(width * super_sample_scale_, height * super_sample_scale_);
    render_node_->BuildRenderNode(width * super_sample_scale_,
                                  height * super_sample_scale_);
    if (output_render_node_) {
      output_render_node_->BuildRenderNode(width, height);
    }
  });

  font_factory_ =
      std::make_unique<grassland::font::Factory>("../fonts/consolab.ttf");
  frame_ = std::make_unique<framework::TextureImage>(
      core_, core_->GetFramebufferWidth() * super_sample_scale_,
      core_->GetFramebufferHeight() * super_sample_scale_);
  sampler_ = std::make_unique<Sampler>(core_->GetDevice());

  render_node_ = std::make_unique<framework::RenderNode>(core_);
  render_node_->AddUniformBinding(global_uniform_buffer_.get(),
                                  VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddBufferBinding(model_uniform_buffer_.get(),
                                 VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddShader("../shaders/gui_shader.vert.spv",
                          VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddShader("../shaders/gui_shader.frag.spv",
                          VK_SHADER_STAGE_FRAGMENT_BIT);
  render_node_->AddColorAttachment(frame_.get(), true);
  render_node_->VertexInput(
      {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT});
  render_node_->BuildRenderNode(
      core_->GetFramebufferWidth() * super_sample_scale_,
      core_->GetFramebufferHeight() * super_sample_scale_);
  texture_vertex_buffer_ =
      std::make_unique<framework::StaticBuffer<glm::vec2>>(core_, 4);
  texture_index_buffer_ =
      std::make_unique<framework::StaticBuffer<uint32_t>>(core_, 6);
  glm::vec2 vertices[] = {
      {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
  uint32_t indices[] = {0, 1, 2, 1, 2, 3};
  texture_vertex_buffer_->Upload(vertices);
  texture_index_buffer_->Upload(indices);

  UpdateGlobalObject(core_->GetFramebufferWidth(),
                     core_->GetFramebufferHeight());
}

void Manager::RegisterWindow(Window *window) {
  windows_.push_back(window);
}

void Manager::BindFrameTexture(framework::TextureImage *frame_texture) {
  output_render_node_ = std::make_unique<framework::RenderNode>(core_);
  output_render_node_->AddUniformBinding(frame_.get(), sampler_.get(),
                                         VK_SHADER_STAGE_FRAGMENT_BIT);
  output_render_node_->AddShader("../shaders/texture_output.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
  output_render_node_->AddShader("../shaders/texture_output.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
  output_render_node_->VertexInput({VK_FORMAT_R32G32_SFLOAT});
  output_render_node_->AddColorAttachment(
      frame_texture,
      VkPipelineColorBlendAttachmentState{
          VK_TRUE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
          VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE,
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD,
          VK_COLOR_COMPONENT_A_BIT | VK_COLOR_COMPONENT_R_BIT |
              VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT});
  output_render_node_->BuildRenderNode(core_->GetFramebufferWidth(),
                                       core_->GetFramebufferHeight());
}

framework::Core *Manager::GetCore() const {
  return core_;
}

font::Factory *Manager::GetFontFactory() const {
  return font_factory_.get();
}

int Manager::GetUnitLength() const {
  return unit_length_;
}

int Manager::RegisterModel(Model *model) {
  models_.push_back(model);
  return int(models_.size()) - 1;
}

void Manager::UpdateGlobalObject(int width, int height) {
  GlobalUniformObject global_object{};
  global_object.screen_to_frame =
      glm::translate(glm::mat4{1.0f}, glm::vec3{-1.0f, -1.0f, 0.0f}) *
      glm::scale(glm::mat4{1.0f},
                 glm::vec3{2.0f / float(width), 2.0f / float(height), 1.0f});
  global_object.super_sample_scale = super_sample_scale_;
  global_uniform_buffer_->Upload(&global_object);
}

void Manager::UpdateModelObjects() {
  for (int i = 0; i < models_.size(); i++) {
    model_uniform_buffer_->operator[](i) = models_[i]->GetModelObject();
  }
}

void Manager::BeginDraw() {
  UpdateModelObjects();
  render_node_->BeginDraw();
}

void Manager::EndDraw() {
  render_node_->EndDraw();
  output_render_node_->BeginDraw();
  output_render_node_->DrawDirect(texture_vertex_buffer_.get(),
                                  texture_index_buffer_.get(),
                                  texture_index_buffer_->Size(), 0);
  output_render_node_->EndDraw();
}

void Manager::Draw() {
  BeginDraw();
  for (auto it = windows_.rbegin(); it != windows_.rend(); it++) {
    (*it)->Draw();
  }
  EndDraw();
}

void Manager::SetScissorRect(const VkRect2D &scissor) {
  render_node_->SetScissorRect({{scissor.offset.x * super_sample_scale_,
                                 scissor.offset.y * super_sample_scale_},
                                {scissor.extent.width * super_sample_scale_,
                                 scissor.extent.height * super_sample_scale_}});
}

void Manager::SetScissorRect(int x, int y, int width, int height) {
  render_node_->SetScissorRect(x * super_sample_scale_, y * super_sample_scale_,
                               width * super_sample_scale_,
                               height * super_sample_scale_);
}

}  // namespace grassland::vulkan::gui
