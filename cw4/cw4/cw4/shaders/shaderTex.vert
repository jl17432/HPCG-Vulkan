#version 450 

layout( location = 0 ) in vec3 iPosition; 
layout( location = 1 ) in vec2 iTexCoord; 
layout( location = 2 ) in vec3 iNmlCoord; 
layout( location = 3 ) in vec4 iTangent; 

layout( set = 0, binding = 0 ) uniform UScene 
{ 
	mat4 camera; 
	mat4 projection; 
	mat4 projCam; 
	vec3 lightPos;
	mat4 lightProjection;
	mat4 lightViewMat;
	float shadowMapRes;
	float swapchainRes;
} uScene; 

layout( location = 0 ) out vec2 v2fTexCoord; 

void main() 
{ 
	v2fTexCoord = iTexCoord; 
	gl_Position = uScene.projCam * vec4( iPosition, 1.f ); 
} 