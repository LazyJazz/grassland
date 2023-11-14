#include "grassland/grassland.h"

using namespace grassland;

class Application {
 public:
  Application(int width, int height, const char *title);
  ~Application();
  void Run();

 private:
  void OnInit();
  void OnUpdate();
  void OnRender();
  void OnClose();
  GLFWwindow *window_;
  std::unique_ptr<d3d12::Core> core_;
  std::string title_;
};
