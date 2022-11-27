#pragma once
#include "grassland/vulkan/gui/manager.h"
#include "grassland/vulkan/gui/model.h"

namespace grassland::vulkan::gui {
class Window {
 public:
  Window(Manager *manager,
         const Layout &layout,
         const std::string &title,
         WindowFlag flag);
  void Resize(const Layout &new_layout);
  void Draw();
  [[nodiscard]] const Layout &GetLayout() const;
  [[nodiscard]] Manager *GetManager() const;
  void Focus();

 private:
  Manager *manager_{};
  std::string title_{};
  Layout layout_{};
  WindowFlag flag_{};
  std::unique_ptr<Model> model_bar_;
  std::unique_ptr<Model> model_title_;
  std::unique_ptr<Model> model_frame_;
};
}  // namespace grassland::vulkan::gui
