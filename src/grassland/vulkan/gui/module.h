#pragma once
#include "grassland/vulkan/gui/defs.h"
#include "grassland/vulkan/gui/manager.h"
#include "grassland/vulkan/gui/util.h"

namespace grassland::vulkan::gui {
class Module {
 public:
  Module(Manager *manager,
         const Layout &ref_layout,
         const ModuleSettings &module_settings);
  virtual ~Module() = 0;
  void AddSubmodule(Module *module);
  [[nodiscard]] const Layout &GetLayout() const;
  [[nodiscard]] const ModuleSettings &GetModuleSettings() const;
  [[nodiscard]] Manager *GetManager() const;

 protected:
  Manager *manager_{nullptr};
  ModuleSettings module_settings_{};
  Layout layout_{};
  std::vector<Module *> submodule_list_;
};
}  // namespace grassland::vulkan::gui
