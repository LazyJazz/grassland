#include "app.h"
#include "grassland/font/font.h"

int main() {
  grassland::font::Factory factory("../fonts/NotoSansSC-Light.otf");
  FontViewer font_viewer(factory.GetChar(L'䉸'));
  font_viewer.Run();
}
