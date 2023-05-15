#version 450

layout(location = 0) in flat uint instance_index;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 position;

layout(location = 0) out vec4 out_color;

void main() {
  if (length(position) > 1.0)
    discard;
  out_color = color;
}
