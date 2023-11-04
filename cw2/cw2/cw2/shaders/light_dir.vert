#version 450 

layout( location = 0 ) in vec3 iPosition; 
layout( location = 1 ) in vec2 iTexCoord; 
layout( location = 2 ) in vec3 iNmlCoord; 

layout( set = 0, binding = 0 ) uniform UScene 
{ 
	mat4 camera; 
	mat4 projection; 
	mat4 projCam; 
	vec3 lightPos;
	vec3 lightCol;
} uScene; 

layout( location = 0 ) out vec3 v2fLightPos; 
layout( location = 1 ) out vec3 v2fPosition;

void main() 
{ 
	//myColor = normalize(vec4(uScene.lightPos - iPosition, 1.f)); 
	v2fLightPos = uScene.lightPos; 
	v2fPosition = iPosition;
	gl_Position = uScene.projCam * vec4( iPosition, 1.f ); 
} 