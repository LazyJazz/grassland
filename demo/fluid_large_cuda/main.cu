#include "fluid_large.cuh"


int main() {
  PhysicSettings settings{};
  FluidLarge fluid_large("Fluid Large", 1024, 1024, settings);
  fluid_large.Run();
}
