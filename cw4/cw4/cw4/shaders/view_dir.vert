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

layout( location = 0 ) out vec3 myColor; 


void main() 
{ 
	vec4 cameraPosition = vec4(0.f, 0.f, 0.f, 1.f);
	mat4 camMatInverse = inverse(uScene.camera);
	vec3 camPos = vec3(camMatInverse * cameraPosition);
	
	myColor = camPos - iPosition;	
	gl_Position = uScene.projCam * vec4( iPosition, 1.f ); 
} 