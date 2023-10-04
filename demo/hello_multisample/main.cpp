#pragma once

#include "application.h"
#include "iostream"

using namespace grassland;

int main() {
  Application app("Hello Multisample", 800, 600, false);
  app.Run();
}
