#include "application.h"
#include "grassland/grassland.h"
#include "iostream"

using namespace grassland::vulkan;

int main() {
  Application app("Application", 1280, 720);
  app.Run();
}
