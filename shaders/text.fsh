#version 330 core

uniform uvec4 chars[512];

in vec2 g_uv;
in vec3 g_color;
flat in uint g_idx;

out vec4 out_color;

void main() {
    uvec4 char8x16 = chars[g_idx];
    uvec2 ji = uvec2(floor(g_uv.xy * vec2(8 - 1, 16)));
    ji = min(uvec2(8 - 1, 16 - 1), ji);
    uint j = ji.x;
    uint i = ji.y;
    uint char8x4 = char8x16[i / 4u];
    if ((char8x4 & (1u << (j + (i % 4u) * 8u))) == 0u) {
        discard;
    }
    out_color = vec4(g_color, 1.0);
}
