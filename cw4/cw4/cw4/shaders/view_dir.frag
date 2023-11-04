#version 450 

layout( location = 0 ) in vec3 myColor; 

layout( set = 1, binding = 0 ) uniform sampler2D uTexColor; 

layout( location = 0 ) out vec4 oColor; 

void main() 
{ 	
	oColor = vec4(normalize(myColor), 1.f); 
} 