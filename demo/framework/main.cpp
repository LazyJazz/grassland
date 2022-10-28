#include <grassland/vulkan/framework/framework.h>
#include <grassland/vulkan/vulkan.h>

#include "glm/glm.hpp"

int main() {
  grassland::vulkan::framework::CoreSettings core_settings;
  core_settings.raytracing_pipeline_required = false;
  core_settings.window_title = "Hello, World!";
  grassland::vulkan::framework::Core core(core_settings);
  std::unique_ptr<grassland::vulkan::framework::TextureImage> texture_image =
      std::make_unique<grassland::vulkan::framework::TextureImage>(
          &core, 256, 256, VK_FORMAT_B8G8R8A8_UNORM);
  texture_image->ReadImage("../textures/xor_grid.png");
  texture_image->WriteImage("../textures/xor_grid.jpg");
  while (!glfwWindowShouldClose(core.GetWindow())) {
    glfwPollEvents();
  }
}
