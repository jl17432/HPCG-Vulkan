#version 450

layout(set = 1, binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 outColor;

void main() {

    float level = textureQueryLod(tex, texCoord).x;

    vec4 c0 = vec4(0.0 , 0.0 , 0.0 , 1.0);   // black
    vec4 c1 = vec4(1.0 , 0.0 , 0.0 , 1.0);   // red
    vec4 c2 = vec4(1.0 , 0.5 , 0.0 , 1.0);   // orange
    vec4 c3 = vec4(1.0 , 1.0 , 0.0 , 1.0);   // yellow
    vec4 c4 = vec4(0.0 , 1.0 , 0.0 , 1.0);   // green
    vec4 c5 = vec4(0.0 , 1.0 , 1.0 , 1.0);   // cyan
    vec4 c6 = vec4(0.0 , 0.0 , 1.0 , 1.0);   // blue
    vec4 c7 = vec4(1.0 , 0.0 , 1.0 , 1.0);   // purple
    vec4 c8 = vec4(1.0 , 1.0 , 1.0 , 1.0);   // white



    vec4 color;


    if (level >= 0.0 && level < 1.0) color = mix(c0, c1, level);
    else if (level >= 1.0 && level < 2.0) color = mix(c1, c2, level - 1.0);
    else if (level >= 2.0 && level < 3.0) color = mix(c2, c3, level - 2.0);
    else if (level >= 3.0 && level < 4.0) color = mix(c3, c4, level - 3.0);
    else if (level >= 4.0 && level < 5.0) color = mix(c4, c5, level - 4.0);
    else if (level >= 5.0 && level < 6.0) color = mix(c5, c6, level - 5.0);
    else if (level >= 6.0 && level < 7.0) color = mix(c6, c7, level - 6.0);
    else if (level >= 7.0 && level < 11.0) color = mix(c7, c8, level - 7.0);
    else if (level >= 11.0 && level <= 12.0) color = mix(c8, c0, level - 11.0);


    outColor = color;
}