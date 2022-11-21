#include "app.h"

int main() {
  FontViewer font_viewer(grassland::font::Mesh(
      {{0.1f, 0.1f}, {0.1f, 0.9f}, {0.9f, 0.1f}, {0.9f, 0.9f}},
      {0, 1, 2, 1, 2, 3}));
  font_viewer.Run();
}
