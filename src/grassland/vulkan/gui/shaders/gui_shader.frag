#version 450
#extension GL_GOOGLE_include_directive : require
#include "defs.glsl"
layout(location = 0) in vec4 frag_extra;
layout(location = 1) in vec4 frag_scissor_rect;
layout(location = 2) flat in uint frag_render_flag;
layout(location = 3) in float frag_round_radius;
layout(location = 4) in float frag_inv_super_sample_scale;
layout(location = 0) out vec4 out_color;

void main() {
  float alpha = 1.0f;
  out_color = frag_extra;
  if ((frag_render_flag & MODEL_RENDER_FLAG_ROUNDED_RECT_BIT) != 0) {
    float distx = 0.0f, disty;
    vec2 frag_coord = gl_FragCoord.xy * frag_inv_super_sample_scale;
    distx = max(distx, frag_scissor_rect.x - frag_coord.x + frag_round_radius);
    distx = max(distx, frag_coord.x - frag_scissor_rect.x -
                           frag_scissor_rect.z + frag_round_radius);
    disty = max(disty, frag_scissor_rect.y - frag_coord.y + frag_round_radius);
    disty = max(disty, frag_coord.y - frag_scissor_rect.y -
                           frag_scissor_rect.w + frag_round_radius);
    alpha = min(max(1.0 -
                        pow(distx * distx * distx * distx +
                                disty * disty * disty * disty,
                            1.0 / 4.0) +
                        frag_round_radius,
                    0.0),
                1.0);
  }
  out_color.w *= alpha;
}
