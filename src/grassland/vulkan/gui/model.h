#pragma once
#include "grassland/vulkan/gui/defs.h"
#include "grassland/vulkan/gui/manager.h"

namespace grassland::vulkan::gui {
class Model {
 public:
  explicit Model(Manager *manager);
  void UploadMesh(const std::vector<Vertex> &vertices,
                  const std::vector<uint32_t> &indices);
  ModelUniformObject &GetModelObject();
  [[nodiscard]] const ModelUniformObject &GetModelObject() const;
  [[nodiscard]] int GetIndex() const;
  void Draw() const;

 private:
  Manager *manager_{nullptr};
  ModelUniformObject model_object_{};
  std::unique_ptr<framework::StaticBuffer<Vertex>> vertex_buffer_;
  std::unique_ptr<framework::StaticBuffer<uint32_t>> index_buffer_;
  int model_index_{0};
};
}  // namespace grassland::vulkan::gui
