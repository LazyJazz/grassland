#version 450
layout(location = 0) in vec2 frag_texcoord;
layout(location = 0) out vec4 color_out;

layout(binding = 0) uniform sampler2D tex;

void main() {
  color_out = texture(tex, frag_texcoord);
}
