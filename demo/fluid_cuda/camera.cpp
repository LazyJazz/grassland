#include "camera.h"

#include "glm/gtc/matrix_transform.hpp"
#include "util.h"

Camera::Camera(const glm::vec3 &eye,
               const glm::vec3 &center,
               const glm::vec3 &up) {
  auto view = glm::inverse(glm::lookAt(eye, center, up));
  rotation_ = DecomposeRotation(glm::mat3{view});
  origin_ = glm::vec3{view[3]};
}

CameraObject Camera::ComposeMatrix(float fovy,
                                   float aspect,
                                   float z_near,
                                   float z_far) const {
  auto R = ComposeRotation(rotation_);
  return {glm::inverse(glm::mat4{R[0], R[1], R[2], glm::vec4{origin_, 1.0f}}),
          glm::scale(glm::mat4{1.0f}, glm::vec3{1.0f, -1.0f, 1.0f}) *
              glm::perspective(fovy, aspect, z_near, z_far)};
}

void Camera::MoveLocal(glm::vec3 offset, const glm::vec3 &rotation) {
  offset = glm::vec3{ComposeRotation(rotation_) * glm::vec4{offset, 0.0f}};
  MoveGlobal(offset, rotation);
}

void Camera::MoveGlobal(const glm::vec3 &offset, const glm::vec3 &rotation) {
  origin_ += offset;
  rotation_ += rotation;
  rotation_.x = glm::clamp(rotation_.x, -glm::pi<float>() * 0.5f,
                           glm::pi<float>() * 0.5f);
  rotation_.y = glm::mod(rotation_.y, glm::pi<float>() * 2.0f);
  rotation_.z = glm::mod(rotation_.z, glm::pi<float>() * 2.0f);
}
