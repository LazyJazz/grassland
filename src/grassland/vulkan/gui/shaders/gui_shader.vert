#version 450
#extension GL_GOOGLE_include_directive : require
#include "defs.glsl"

layout(location = 0) in vec4 vert_position;
layout(location = 1) in vec4 vert_extra;
layout(location = 0) out vec4 frag_extra;

layout(binding = 0) readonly uniform global_uniform_object {
  GlobalUniformObject global_object;
};

layout(binding = 1) readonly buffer local_uniform_objects {
  ModelUniformObject model_objects[];
};

void main() {
  ModelUniformObject model_object = model_objects[gl_InstanceIndex];
  gl_Position = global_object.screen_to_frame * model_object.local_to_screen *
                vert_position;
  frag_extra = vert_extra;
}
