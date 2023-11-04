#version 450 

layout( location = 0 ) in vec2 v2fUV;
layout( location = 1 ) in vec2 v2fBoundary;



layout( set = 1, binding = 0 ) uniform sampler2D sam_pbr; 
layout( set = 1, binding = 1 ) uniform sampler2D sam_filter; 




layout( location = 0 ) out vec4 oColor; 

void main() 
{ 

	vec2 viewport_offset = vec2(0.f, 0.f);

	vec2 fliped_uv = vec2(v2fUV.x, 1.0 - v2fUV.y);

	vec3 o_pbr = texture(sam_pbr, fliped_uv).rgb;
	vec4 o_filter = texture(sam_filter, fliped_uv).rgba;

	// final output color
	oColor = vec4(o_pbr, 1.f);
	//oColor = vec4(o_filter);
} 