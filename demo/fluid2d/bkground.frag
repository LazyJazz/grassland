#version 450

layout(location = 0) in vec2 tex_coord;
layout(location = 0) out vec4 color_out;

layout(binding = 0) uniform sampler2D color_texture;

void main() {
  color_out = texture(color_texture, tex_coord);
}
