#version 450

layout(location = 0) in vec3 position;

void main() {
  // Hello, World!
  gl_Position = vec4(position, 1.0);
}
