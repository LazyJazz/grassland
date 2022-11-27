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
    if (render_node_) {
      render_node_->BuildRenderNode(width, height);
    }
  });

  font_factory_ = std::make_unique<grassland::font::Factory>(
      "../fonts/NotoSansSC-Regular.otf");
}

void Manager::RegisterWindow(Window *window) {
  windows_.push_back(window);
}

void Manager::BindFrameTexture(framework::TextureImage *frame_texture) {
  render_node_ = std::make_unique<framework::RenderNode>(core_);
  render_node_->AddUniformBinding(global_uniform_buffer_.get(),
                                  VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddBufferBinding(model_uniform_buffer_.get(),
                                 VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddShader("../shaders/gui_shader.vert.spv",
                          VK_SHADER_STAGE_VERTEX_BIT);
  render_node_->AddShader("../shaders/gui_shader.frag.spv",
                          VK_SHADER_STAGE_FRAGMENT_BIT);
  render_node_->AddColorAttachment(frame_texture, true);
  render_node_->VertexInput(
      {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT});
  render_node_->BuildRenderNode(core_->GetFramebufferWidth(),
                                core_->GetFramebufferHeight());
  UpdateGlobalObject(core_->GetFramebufferWidth(),
                     core_->GetFramebufferHeight());
}

framework::Core *Manager::GetCore() const {
  return core_;
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
}

void Manager::Draw() {
  BeginDraw();
  for (auto it = windows_.rbegin(); it != windows_.rend(); it++) {
    (*it)->Draw();
  }
  EndDraw();
}

}  // namespace grassland::vulkan::gui
