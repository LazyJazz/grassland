#version 450
#extension GL_GOOGLE_include_directive : require
#include "defs.glsl"
layout(location = 0) in vec4 frag_extra;
layout(location = 0) out vec4 out_color;

void main() {
  out_color = frag_extra;
}
