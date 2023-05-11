#include "grassland/grassland.h"
#include "iostream"

using namespace grassland::vulkan;

int main() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  auto window = glfwCreateWindow(800, 600, "Grassland", nullptr, nullptr);
  Instance instance{InstanceSettings{true, true}};
  Surface surface(instance, window);
  for (auto device : instance.GetEnumeratePhysicalDevices()) {
    LAND_INFO("{}", device.DeviceName());
  }
  Device device(instance.PickDevice(), &surface);
  Swapchain swapchain(device);
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
  }
}
