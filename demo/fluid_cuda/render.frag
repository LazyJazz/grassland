#version 450 core

layout(location = 0) in vec3 normal;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 color0;

const vec3 lighting_dir = normalize(vec3(1.0, 1.0, 1.0));
const float lighting_scale = 0.7;

void main() {
  color0 = vec4(color.rgb * ((1.0 - lighting_scale) +
                             lighting_scale *
                                 max(dot(lighting_dir, normalize(normal)), 0)),
                color.a);
}
