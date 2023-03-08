#version 450

layout(binding = 1) uniform sampler2D tex;

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
  outColor = texture(tex, fragColor.xy);
}
