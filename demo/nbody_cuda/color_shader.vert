#version 460

layout(location = 0) in vec2 v;
layout(location = 0) out vec2 frag_v;

layout(binding = 0) uniform global_uniform_object {
  mat4 world_to_screen;
  mat4 camera_to_world;
  float particle_size;
};
layout(binding = 1) buffer particles_array {
  vec4 positions[];
};

void main() {
  vec4 pos = positions[gl_InstanceIndex];
  gl_Position = world_to_screen *
                (pos + camera_to_world * vec4(v, 0.0, 0.0) * particle_size);
  frag_v = v;
}
