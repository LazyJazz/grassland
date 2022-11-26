#pragma once
#include "glm/glm.hpp"
#include "grassland/vulkan/gui/util.h"

namespace grassland::vulkan::gui {
struct Layout {
  int x;
  int y;
  int width;
  int height;
};

enum ModuleViewportSizeMode : uint32_t {
  MODULE_VIEWPORT_SIZE_MODE_FIXED = 0,
  MODULE_VIEWPORT_SIZE_MODE_SPACE = 1,
  MODULE_VIEWPORT_SIZE_MODE_FRAME = 2
};

enum ModuleFrameSizeMode : uint32_t {
  MODULE_FRAME_SIZE_MODE_VIEWPORT = 0,
  MODULE_FRAME_SIZE_MODE_SLIDER = 1,
  MODULE_FRAME_SIZE_MODE_DYNAMIC = 2
};

enum ModulePositionMode : uint32_t {
  MODULE_POSITION_MODE_CENTER = 0,
  MODULE_POSITION_MODE_ALIGN_LOW_BORDER = 1,
  MODULE_POSITION_MODE_ALIGN_HIGH_BORDER = 2
};

struct ModuleSettings {
  ModuleViewportSizeMode viewport_size_h{MODULE_VIEWPORT_SIZE_MODE_FIXED};
  ModuleViewportSizeMode viewport_size_v{MODULE_VIEWPORT_SIZE_MODE_FIXED};
  ModuleFrameSizeMode frame_size_h{MODULE_FRAME_SIZE_MODE_VIEWPORT};
  ModuleFrameSizeMode frame_size_v{MODULE_FRAME_SIZE_MODE_VIEWPORT};
  ModulePositionMode position_mode_h{MODULE_POSITION_MODE_CENTER};
  ModulePositionMode position_mode_v{MODULE_POSITION_MODE_CENTER};
  float weight{1.0f};
};

struct Offset {
  int x;
  int y;
};

struct Extent {
  int width;
  int height;
};

struct Vertex {
  glm::vec4 position;
  glm::vec4 extra;
};

struct GlobalUniformObject {
  glm::mat4 screen_to_frame;
};

struct ModelUniformObject {
  glm::mat4 local_to_screen;
  int extra_interpret_mode;
  int reserve[3];
};
}  // namespace grassland::vulkan::gui
