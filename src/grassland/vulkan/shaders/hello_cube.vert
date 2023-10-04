#version 450
// Hello, Cube!

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 fragColor;

layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
}
ubo;

void main() {
  gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
  fragColor = color;
}
