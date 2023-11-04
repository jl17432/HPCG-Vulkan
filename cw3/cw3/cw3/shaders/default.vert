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

layout( location = 0 ) out vec3 v2fPosition; 
layout( location = 1 ) out vec3 v2fNormal; 


void main()
{

	vec4 cameraPosition = vec4(0.f, 0.f, 0.f, 1.f);
	mat4 camMatInverse = inverse(uScene.camera);
	vec3 camPos = vec3(camMatInverse * cameraPosition);

	v2fPosition = iPosition;
	v2fNormal	= iNmlCoord;


	gl_Position = uScene.projCam * vec4( iPosition, 1.f ); 

}
