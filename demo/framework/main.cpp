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
          &core, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT);
  texture_image->ReadImage("../textures/air_museum_playground_2k.hdr");
  texture_image->WriteImage("../textures/air_museum_playground_2k.hdr.png");
  while (!glfwWindowShouldClose(core.GetWindow())) {
    glfwPollEvents();
  }
}
