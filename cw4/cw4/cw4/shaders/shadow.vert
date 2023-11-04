#version 450 

layout( location = 0 ) in vec3 iPosition; 

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

	

void main() 
{ 
	vec4 vPosition = uScene.lightProjection * uScene.lightViewMat * vec4( iPosition, 1.f );
	gl_Position = vPosition;
} 