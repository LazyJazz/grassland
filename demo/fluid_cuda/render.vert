#version 450 core

struct RenderInfo {
  mat4 model;
  vec4 color;
};

layout(binding = 0) uniform global_uniform_object {
  mat4 camera;
  mat4 proj;
};
layout(binding = 1) buffer render_info_buffer {
  RenderInfo render_infos[];
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 0) out vec3 frag_normal;
layout(location = 1) out vec4 frag_color;

void main() {
  gl_Position = proj * camera * render_infos[gl_InstanceIndex].model *
                vec4(position, 1.0);
  frag_normal = vec3(render_infos[gl_InstanceIndex].model * vec4(normal, 0.0));
  frag_color = render_infos[gl_InstanceIndex].color;
}
