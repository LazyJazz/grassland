#pragma once
#include "grassland/font/font.h"
#include "grassland/vulkan/gui/defs.h"
#include "grassland/vulkan/gui/util.h"

namespace grassland::vulkan::gui {

class Model;

class Manager {
 public:
  explicit Manager(framework::Core *core);
  void BindFrameTexture(framework::TextureImage *frame_texture);
  [[nodiscard]] framework::Core *GetCore() const;
  [[nodiscard]] float GetUnitLength() const;
  void UpdateLayout();
  void RefreshLayout();
  void UpdatePosition();

  void BeginDraw();
  void EndDraw();

 private:
  void UpdateGlobalObject(int width, int height);
  void UpdateModelObjects();
  friend Model;
  int RegisterModel(Model *model);
  framework::Core *core_{nullptr};
  std::unique_ptr<framework::RenderNode> render_node_;
  std::unique_ptr<framework::TextureImage> frame_;
  std::unique_ptr<font::Factory> font_factory_;
  std::unique_ptr<framework::StaticBuffer<GlobalUniformObject>>
      global_uniform_buffer_;
  std::unique_ptr<framework::DynamicBuffer<ModelUniformObject>>
      model_uniform_buffer_;
  float unit_length_{50.0f};
  std::vector<Model *> models_;
};
}  // namespace grassland::vulkan::gui
