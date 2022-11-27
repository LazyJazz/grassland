#pragma once
#include "grassland/font/font.h"
#include "grassland/vulkan/gui/defs.h"
#include "grassland/vulkan/gui/util.h"

namespace grassland::vulkan::gui {

class Model;

class Window;

class Manager {
 public:
  explicit Manager(framework::Core *core);
  void BindFrameTexture(framework::TextureImage *frame_texture);
  [[nodiscard]] framework::Core *GetCore() const;
  [[nodiscard]] font::Factory *GetFontFactory() const;
  [[nodiscard]] int GetUnitLength() const;

  void Draw();

 private:
  friend Model;
  friend Window;

  void BeginDraw();
  void EndDraw();
  void UpdateGlobalObject(int width, int height);
  void UpdateModelObjects();
  int RegisterModel(Model *model);
  void RegisterWindow(Window *window);
  void SetScissorRect(const VkRect2D &scissor);
  void SetScissorRect(int x, int y, int width, int height);

  framework::Core *core_{nullptr};
  std::unique_ptr<framework::RenderNode> render_node_;
  std::unique_ptr<framework::RenderNode> output_render_node_;
  std::unique_ptr<framework::StaticBuffer<glm::vec2>> texture_vertex_buffer_;
  std::unique_ptr<framework::StaticBuffer<uint32_t>> texture_index_buffer_;
  std::unique_ptr<framework::TextureImage> frame_;
  std::unique_ptr<Sampler> sampler_;
  std::unique_ptr<font::Factory> font_factory_;
  std::unique_ptr<framework::StaticBuffer<GlobalUniformObject>>
      global_uniform_buffer_;
  std::unique_ptr<framework::DynamicBuffer<ModelUniformObject>>
      model_uniform_buffer_;
  std::vector<Model *> models_;

  int unit_length_{32};
  int super_sample_scale_{2};
  std::vector<Window *> windows_;
  Window *focus_window_{nullptr};
};
}  // namespace grassland::vulkan::gui
