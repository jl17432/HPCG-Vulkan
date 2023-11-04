#version 450 


layout( set = 0, binding = 0 ) uniform UScene 
{ 
	mat4 camera; 
	mat4 projection; 
	mat4 projCam; 
	vec3 lightPos;
	vec3 lightCol;

} uScene; 

const vec2 kVertexPositions[3] = vec2[3](
	vec2(-1.f, 1.f),
	vec2( 3.f, 1.f),
	vec2(-1.f, -3.f)
	);


layout(push_constant) uniform PushConstants {

	float width;
	float height;
	
} pc;

layout( location = 0 ) out vec2 v2fUV;
layout( location = 1 ) out vec2 v2fBoundary;

void main() 
{ 	
	v2fUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	//gl_Position = vec4(v2fUV * 2.0f - 1.0f, 0.f, 1.0f);

	const vec2 xy = kVertexPositions[gl_VertexIndex]; 
	gl_Position = vec4( xy, 0.f, 1.f );

	v2fBoundary = vec2(pc.width, pc.height);
} 