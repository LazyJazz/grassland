#include "grassland/vulkan/gui/module.h"

namespace grassland::vulkan::gui {

Module::Module(Manager *manager,
               const Layout &layout,
               const ModuleSettings &module_settings) {
  manager_ = manager;
  layout_ = layout;
  module_settings_ = module_settings;
}

void Module::AddSubmodule(Module *module) {
  submodule_list_.push_back(module);
}

const Layout &Module::GetLayout() const {
  return layout_;
}

const ModuleSettings &Module::GetModuleSettings() const {
  return module_settings_;
}

Manager *Module::GetManager() const {
  return manager_;
}

}  // namespace grassland::vulkan::gui
