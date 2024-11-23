#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec2 vertex_centers[];
in vec2 vertex_scale[];
in vec3 vertex_color[];
in uint vertex_idx[];

out vec2 g_uv;
out vec3 g_color;
flat out uint g_idx;

void main() {
    vec2 c = vertex_centers[0];
    vec2 h = vertex_scale[0];
    g_color = vertex_color[0].rgb;
    g_idx = vertex_idx[0];

    gl_Position = vec4(c + vec2(-h.x, h.y), 0.0, 1.0);
    g_uv = vec2(1, 1);
    EmitVertex();

    gl_Position = vec4(c + vec2(-h.x, -h.y), 0.0, 1.0);
    g_uv = vec2(1, 0);
    EmitVertex();

    gl_Position = vec4(c + vec2(h.x, h.y), 0.0, 1.0);
    g_uv = vec2(0, 1);
    EmitVertex();

    gl_Position = vec4(c + vec2(h.x, -h.y), 0.0, 1.0);
    g_uv = vec2(0, 0);
    EmitVertex();

//    gl_Position = vec4(c + vec2(0, h.y), 0.0, 1.0);
//    g_uv = vec2(0.5, 1);
//    EmitVertex();

    EndPrimitive();
}

