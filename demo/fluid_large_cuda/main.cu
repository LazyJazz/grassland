#include <memory>

#include "fluid_large.cuh"

void GUI(const PhysicSettings &settings) {
  FluidLarge fluid_large("Fluid Large", 1024, 1024, settings);
  fluid_large.Run();
}

[[noreturn]] void NoGUI(const PhysicSettings &settings) {
  PhysicSolver solver(settings);
  auto start_tp = std::chrono::steady_clock::now();
  int frames = 4000;
  for (int i = 0; i < frames; i++) {
    solver.UpdateStep();
  }
  auto dur = std::chrono::steady_clock::now() - start_tp;
  std::cout << double(dur / std::chrono::milliseconds(1)) * 1e-3 / float(frames)
            << "s/frame avg" << std::endl;
}

int main() {
  PhysicSettings settings{};
  NoGUI(settings);
}
