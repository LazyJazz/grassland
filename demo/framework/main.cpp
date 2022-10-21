#include <grassland/vulkan/vulkan.h>

#include "app.h"

int main() {
  App app(1280, 720, "Rotating cube");
  app.Run();
}
