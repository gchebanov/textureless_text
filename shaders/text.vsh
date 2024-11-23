#version 330 core

layout (location=0) in vec2 in_position;
layout (location=1) in vec2 in_scale;
layout (location=2) in vec3 in_color;
layout (location=3) in uint in_idx;

out vec2 vertex_centers;
out vec2 vertex_scale;
out vec3 vertex_color;
out uint vertex_idx;

void main() {
    vertex_centers = in_position;
    vertex_scale = in_scale;
    vertex_color = in_color;
    vertex_idx = in_idx;
}
