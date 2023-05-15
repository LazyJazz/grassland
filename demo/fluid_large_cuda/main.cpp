#include "fluid_large.cuh"

int main() {
  FluidLarge fluid_large("Fluid Large", 1024, 1024);
  fluid_large.Run();
}
