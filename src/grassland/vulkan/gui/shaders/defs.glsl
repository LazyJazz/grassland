struct GlobalUniformObject {
  mat4 screen_to_frame;
};

struct ModelUniformObject {
  mat4 local_to_screen;
  float x;
  float y;
  float width;
  float height;
  uint render_flag;
  float round_radius;
  float reserve0;
  float reserve1;
};

#define MODEL_RENDER_FLAG_ROUNDED_RECT_BIT 0x00000001
#define MODEL_RENDER_FLAG_TEXTURE_BIT 0x00000002
