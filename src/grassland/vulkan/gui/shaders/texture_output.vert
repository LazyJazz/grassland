#version 450

layout(location = 0) in vec2 vert_position;
layout(location = 0) out vec2 frag_texcoord;

void main() {
  gl_Position =
      vec4(vert_position * vec2(2.0, 2.0) + vec2(-1.0, -1.0), 0.0, 1.0);
  frag_texcoord = vert_position;
}
