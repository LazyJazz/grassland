#include "grassland/vulkan/gui/model.h"

namespace grassland::vulkan::gui {

Model::Model(Manager *manager) {
  manager_ = manager;
  model_index_ = manager_->RegisterModel(this);
  model_object_.render_flag = 0;
  model_object_.local_to_screen = glm::mat4{1.0f};
}

void Model::UploadMesh(const std::vector<Vertex> &vertices,
                       const std::vector<uint32_t> &indices) {
  vertex_buffer_ = std::make_unique<framework::StaticBuffer<Vertex>>(
      manager_->GetCore(), vertices.size());
  index_buffer_ = std::make_unique<framework::StaticBuffer<uint32_t>>(
      manager_->GetCore(), indices.size());
  vertex_buffer_->Upload(vertices.data());
  index_buffer_->Upload(indices.data());
}

ModelUniformObject &Model::GetModelObject() {
  return model_object_;
}

const ModelUniformObject &Model::GetModelObject() const {
  return model_object_;
}

int Model::GetIndex() const {
  return model_index_;
}

void Model::Draw() const {
  manager_->render_node_->DrawDirect(vertex_buffer_.get(), index_buffer_.get(),
                                     int(index_buffer_->Size()), model_index_);
}

void Model::UploadMesh(const std::vector<glm::vec2> &vertices,
                       const std::vector<uint32_t> &indices,
                       const glm::vec4 &color) {
  std::vector<Vertex> real_vertices;
  real_vertices.reserve(vertices.size());
  for (auto &vert : vertices) {
    real_vertices.push_back({{vert, 0.0f, 1.0f}, color});
  }
  UploadMesh(real_vertices, indices);
}
}  // namespace grassland::vulkan::gui
