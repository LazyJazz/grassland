
struct GlobalUniformObject {
  mat4 screen_to_frame;
};

struct ModelUniformObject {
  mat4 local_to_screen;
  int extra_interpret_mode;
};
