#version 450

struct RenderObjectInfo {
  mat4 transform;
  vec4 color;
};

layout(location = 0) in vec2 vertex;
layout(location = 0) out vec4 frag_color;

layout(binding = 0) uniform global_transform {
  mat4 transform;
};
layout(binding = 1) buffer render_object_info_buffer {
  RenderObjectInfo render_object_infos[];
};

void main() {
  RenderObjectInfo render_object_info = render_object_infos[gl_InstanceIndex];
  frag_color = render_object_info.color;
  gl_Position =
      transform * render_object_info.transform * vec4(vertex, 0.0, 1.0);
  gl_Position.y *= -1.0;
}
