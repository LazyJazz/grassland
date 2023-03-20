#version 450

layout(location = 0) in vec2 pos;
layout(location = 0) out vec2 frag_tex_coord;

void main() {
  gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
  frag_tex_coord = pos;
  gl_Position.y *= -1.0;
}
