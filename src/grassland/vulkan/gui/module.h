#pragma once
#include "grassland/vulkan/gui/defs.h"
#include "grassland/vulkan/gui/manager.h"
#include "grassland/vulkan/gui/util.h"

namespace grassland::vulkan::gui {
// class Module {
//  public:
//   explicit Module(const ModuleSettings& module_settings);
//   virtual ~Module() = 0;
//   virtual void UpdateExtent(const Extent& budget) = 0;
//   virtual void Draw() = 0;
//   virtual void PostDraw();
//   [[nodiscard]] const Extent& GetExtent() const;
//   [[nodiscard]] const ModuleSettings& GetModuleSettings() const;
//   void UpdateLayout(const Layout& layout);
//   protected:
//    virtual void Update() = 0;
//    Offset offset_{};
//    Extent extent_{};
//    Layout layout_{};
//    ModuleSettings module_settings_{};
//   std::vector<Module *> module_list_;
//   std::unique_ptr<Model> slide_bar_;
// };
}  // namespace grassland::vulkan::gui
