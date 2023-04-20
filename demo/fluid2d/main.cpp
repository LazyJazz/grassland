#include "fluid2d.h"
#include "iostream"

int main() {
  auto fluid_app = std::make_unique<Fluid2D>(8192 / 2);
  fluid_app->Run();
}
