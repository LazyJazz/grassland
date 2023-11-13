#include "grassland/d3d12/core/core.h"

namespace grassland::d3d12 {

Core::Core(const CoreSettings &settings) {
  // Create the factory
  DXGIFactorySettings factory_settings;
  factory_settings.enable_debug_layer = settings.enable_debug_layer;
  factory_ = std::make_unique<DXGIFactory>(factory_settings);
}

Core::~Core() {
  // Release all resources in reverse order of creation
  factory_.reset();
}

}  // namespace grassland::d3d12
