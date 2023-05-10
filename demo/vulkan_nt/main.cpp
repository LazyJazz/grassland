#include "grassland/grassland.h"
#include "iostream"

using namespace grassland::vulkan;

int main() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwCreateWindow(800, 600, "Grassland", nullptr, nullptr);
  Instance instance{};
  auto handle = instance.Handle();
}
