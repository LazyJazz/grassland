#include "fluid_large.cuh"
#include <memory>

void GUI(const PhysicSettings& settings) {
  FluidLarge fluid_large("Fluid Large", 1024, 1024, settings);
  fluid_large.Run();
}

[[noreturn]] void NoGUI(const PhysicSettings& settings) {
  PhysicSolver solver(settings);
  while (true) {
    solver.UpdateStep();
  }
}

int main() {
  PhysicSettings settings{};
  // GUI(settings);
  NoGUI(settings);
}
