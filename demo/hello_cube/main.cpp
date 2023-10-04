#pragma once

#include "application.h"
#include "iostream"

using namespace grassland;

int main() {
  Application app("Hello Cube", 800, 600, false);
  app.Run();
}
