#pragma once
#include "glm/glm.hpp"

struct CameraObject {
  glm::mat4 world_to_camera;
  glm::mat4 projection;
};

class Camera {
 public:
  Camera() = default;
  Camera(const glm::vec3 &eye,
         const glm::vec3 &center,
         const glm::vec3 &up = glm::vec3{0.0f, 1.0f, 0.0f});
  [[nodiscard]] CameraObject ComposeMatrix(float fovy,
                                           float aspect,
                                           float z_near,
                                           float z_far) const;
  void MoveLocal(glm::vec3 offset, const glm::vec3 &rotation);
  void MoveGlobal(const glm::vec3 &offset, const glm::vec3 &rotation);

 private:
  glm::vec3 rotation_{1.0f};
  glm::vec3 origin_{0.0f};
};
