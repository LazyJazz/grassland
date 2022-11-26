#version 450
#extension GL_GOOGLE_include_directive : require
#include "defs.glsl"

layout(location = 0) in vec4 vert_position;
layout(location = 1) in vec4 vert_extra;
layout(location = 0) out vec4 frag_extra;
layout(location = 1) out vec4 frag_scissor_rect;
layout(location = 2) out uint frag_render_flag;
layout(location = 3) out float frag_round_radius;

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
  frag_scissor_rect = vec4(model_object.x, model_object.y, model_object.width,
                           model_object.height);
  frag_render_flag = model_object.render_flag;
  frag_round_radius = model_object.round_radius;
}
