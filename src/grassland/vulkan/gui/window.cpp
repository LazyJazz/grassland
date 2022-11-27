#include "grassland/vulkan/gui/window.h"

#include "glm/gtc/matrix_transform.hpp"

namespace grassland::vulkan::gui {

Window::Window(Manager *manager,
               const Layout &layout,
               const std::string &title,
               WindowFlag flag) {
  manager_ = manager;
  title_ = title;
  flag_ = flag;
  manager_->RegisterWindow(this);
  model_bar_ = std::make_unique<Model>(manager_);
  model_title_ = std::make_unique<Model>(manager_);
  model_frame_ = std::make_unique<Model>(manager_);
  Resize(layout);
}

void Window::Draw() {
  manager_->render_node_->SetScissorRect(layout_.x, layout_.y, layout_.width,
                                         layout_.height);
  if (flag_ & WINDOW_FLAG_BAR_BIT) {
    model_bar_->Draw();
  }
  model_frame_->Draw();
}

const Layout &Window::GetLayout() const {
  return layout_;
}

Manager *Window::GetManager() const {
  return manager_;
}

void Window::Focus() {
  manager_->focus_window_ = this;
  int window_id = -1;
  for (int i = 0; i < manager_->windows_.size(); i++) {
    if (manager_->windows_[i] == this) {
      window_id = i;
      break;
    }
  }

  for (int i = window_id; i > 0; i++) {
    manager_->windows_[i] = manager_->windows_[i - 1];
  }

  manager_->windows_[0] = this;
}

void Window::Resize(const Layout &new_layout) {
  layout_ = new_layout;
  auto unit_length = manager_->GetUnitLength();
  if (flag_ & WINDOW_FLAG_BAR_BIT) {
    layout_.height = std::max(layout_.height, manager_->GetUnitLength());
  }

  ModelUniformObject model_object{};

  model_object.width = float(layout_.width);
  model_object.height = float(layout_.height);
  model_object.x = float(layout_.x);
  model_object.y = float(layout_.y);
  model_object.render_flag = MODEL_RENDER_FLAG_ROUNDED_RECT_BIT;
  model_object.round_radius = float(unit_length) * 0.5f;
  model_object.local_to_screen = glm::translate(
      glm::mat4{1.0f}, glm::vec3{float(layout_.x), float(layout_.y), 0.0f});
  model_bar_->GetModelObject() = model_object;
  if (flag_ & WINDOW_FLAG_BAR_BIT) {
    model_object.local_to_screen = glm::translate(
        glm::mat4{1.0f},
        glm::vec3{float(layout_.x), float(layout_.y + unit_length), 0.0f});
  }

  model_frame_->GetModelObject() = model_object;

  if (flag_ & WINDOW_FLAG_BAR_BIT) {
    model_bar_->UploadMesh({{0.0f, 0.0f},
                            {0.0f, float(unit_length)},
                            {float(layout_.width), 0.0f},
                            {float(layout_.width), float(unit_length)}},
                           {0, 1, 2, 1, 2, 3}, {glm::vec3{0.9f}, 1.0f});
    model_frame_->UploadMesh(
        {{0.0f, 0.0f},
         {0.0f, float(layout_.height - unit_length)},
         {float(layout_.width), 0.0f},
         {float(layout_.width), float(layout_.height - unit_length)}},
        {0, 1, 2, 1, 2, 3}, {glm::vec3{0.8f}, 0.8f});
  } else {
    model_frame_->UploadMesh({{0.0f, 0.0f},
                              {0.0f, float(layout_.height)},
                              {float(layout_.width), 0.0f},
                              {float(layout_.width), float(layout_.height)}},
                             {0, 1, 2, 1, 2, 3}, {glm::vec3{0.8f}, 0.8f});
  }
}
}  // namespace grassland::vulkan::gui
