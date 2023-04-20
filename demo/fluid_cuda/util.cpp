#include "util.h"

glm::vec3 DecomposeRotation(glm::mat3 R) {
  return {
      std::atan2(-R[2][1], std::sqrt(R[0][1] * R[0][1] + R[1][1] * R[1][1])),
      std::atan2(R[2][0], R[2][2]), std::atan2(R[0][1], R[1][1])};
}

glm::mat4 ComposeRotation(glm::vec3 pitch_yaw_roll) {
  return glm::rotate(glm::mat4{1.0f}, pitch_yaw_roll.y,
                     glm::vec3{0.0f, 1.0f, 0.0f}) *
         glm::rotate(glm::mat4{1.0f}, pitch_yaw_roll.x,
                     glm::vec3{1.0f, 0.0f, 0.0f}) *
         glm::rotate(glm::mat4{1.0f}, pitch_yaw_roll.z,
                     glm::vec3{0.0f, 0.0f, 1.0f});
}
