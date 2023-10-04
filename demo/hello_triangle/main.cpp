#pragma once

#include "application.h"
#include "iostream"

using namespace grassland;

int main() {
  Application app("Hello Triangle", 800, 600, false);
  app.Run();
}
