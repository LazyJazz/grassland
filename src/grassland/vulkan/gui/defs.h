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

typedef enum WindowFlag : uint32_t {
  WINDOW_FLAG_NONE = 0,
  WINDOW_FLAG_BAR_BIT = 1,
  WINDOW_FLAG_COMPRESS_BIT = 2,
  WINDOW_FLAG_CLOSE_BIT = 4,
} WindowFlag;

enum ModuleFrameSizeMode : uint32_t {
  MODULE_FRAME_SIZE_MODE_FIXED = 0,
  MODULE_FRAME_SIZE_MODE_SLIDER = 1
};

struct ModuleSettings {
  ModuleFrameSizeMode frame_size_mode_h{MODULE_FRAME_SIZE_MODE_FIXED};
  ModuleFrameSizeMode frame_size_mode_v{MODULE_FRAME_SIZE_MODE_FIXED};
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
  glm::mat4 local_to_screen{1.0f};
  float x{};
  float y{};
  float width{};
  float height{};
  uint32_t render_flag{};
  float round_radius{};
  float reserve0{};
  float reserve1{};
};

typedef enum ModelRenderFlagBits : uint32_t {
  MODEL_RENDER_FLAG_ROUNDED_RECT_BIT = 0x00000001,
  MODEL_RENDER_FLAG_TEXTURE_BIT = 0x00000002
} ModelRenderFlagBits;
}  // namespace grassland::vulkan::gui
