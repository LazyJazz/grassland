#version 450

layout(location = 0) in flat uint instance_index;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 position;

layout(location = 0) out vec4 out_color;

void main() {
  if (length(position) > 1.0)
    discard;
  out_color = color;
  //  float scale = instance_index / (400.0 * 800.0 * 400.0);
  //  out_color = color * (1.0 - scale) + vec4(scale);
}
