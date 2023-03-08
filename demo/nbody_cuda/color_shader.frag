#version 450

layout(location = 0) in vec2 frag_v;

layout(location = 0) out vec4 out_color;

void main() {
  float scale = max(1.0 - length(frag_v), 0.0);
  out_color = vec4(vec3(0.5, 0.2, 0.1) * scale * scale * scale, 0.0);
}
