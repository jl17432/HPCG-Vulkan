#version 450 

layout( location = 0 ) in vec3 v2fLightPos; 
layout( location = 1 ) in vec3 v2fPosition;


layout( set = 1, binding = 0 ) uniform sampler2D uTexColor; 
layout( set = 1, binding = 1 ) uniform sampler2D uMetalColor; 
layout( set = 1, binding = 2 ) uniform sampler2D uRoughColor; 

layout( location = 0 ) out vec4 oColor; 

void main() 
{ 	
	vec3 temp = normalize(v2fLightPos - v2fPosition);
	oColor = vec4(temp, 1.f);
} 