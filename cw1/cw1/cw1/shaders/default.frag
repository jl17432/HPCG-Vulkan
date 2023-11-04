#version 450

layout( location = 0 ) in  vec3 FragCol;
layout( location = 0 ) out vec4 oColor;

void main()
{
        oColor = vec4(FragCol,1.0f);

}