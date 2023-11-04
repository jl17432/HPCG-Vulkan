#version 450 

layout( location = 0 ) in vec2 v2fTexCoord; 

layout( set = 1, binding = 0 ) uniform sampler2D uTexColor; 
layout( set = 1, binding = 1 ) uniform sampler2D uMetalColor; 
layout( set = 1, binding = 2 ) uniform sampler2D uRoughColor; 

layout( location = 0 ) out vec4 oColor; 

void main() 
{ 
	oColor = vec4( texture( uTexColor, v2fTexCoord ).rgb, 1.f ); 
} 