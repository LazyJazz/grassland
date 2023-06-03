#version 450

struct InstanceInfo {
  vec3 position;
  float size;
  vec4 color;
};

layout(location = 0) in vec2 position;
layout(location = 0) out uint instance_id;
layout(location = 1) out vec4 color;
layout(location = 2) out vec2 frag_position;

layout(std140, binding = 0) uniform GlobalUniformObject {
  mat4 world;
  mat4 camera;
};
layout(std140, binding = 1) buffer InstanceInfoArray {
  InstanceInfo instance_infos[];
};

void main() {
  InstanceInfo instance_info = instance_infos[gl_InstanceIndex];
  vec4 pos = vec4(instance_info.position, 1.0);
  pos = world * pos;
  pos += vec4(position, 0.0, 0.0) * instance_info.size;
  gl_Position = camera * pos * vec4(1.0, -1.0, 1.0, 1.0);
  color = instance_info.color;
  instance_id = gl_InstanceIndex;
  frag_position = position;
}
