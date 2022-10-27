#include <grassland/vulkan/framework/framework.h>
#include <grassland/vulkan/vulkan.h>

int main() {
  grassland::vulkan::framework::CoreSettings core_settings;
  core_settings.raytracing_pipeline_required = false;
  core_settings.window_title = "Hello, World!";
  grassland::vulkan::framework::Core core(core_settings);
  while (!glfwWindowShouldClose(core.GetWindow())) {
    glfwPollEvents();
  }
}
