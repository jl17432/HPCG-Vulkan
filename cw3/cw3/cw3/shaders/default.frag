#version 450 

layout( location = 0 ) in vec3 v2fPosition; 
layout( location = 1 ) in vec3 v2fNormal; 



layout(push_constant) uniform PushConstants {

	float baseColor_r;
	float baseColor_g;
	float baseColor_b;

	float emissiveColor_r;
	float emissiveColor_g;
	float emissiveColor_b;

    float roughness;
    float metalness;
} pc;

layout( location = 0 ) out vec4 oColor; 
layout( location = 1 ) out vec4 oPosition; 
layout( location = 2 ) out vec4 oNormal; 
layout( location = 3 ) out vec4 oMaterial; 
layout( location = 4 ) out vec4 oEmissive; 

void main()
{
	vec3 inColor	= vec3(pc.baseColor_r, pc.baseColor_g, pc.baseColor_b);
	vec3 emissive	= vec3(pc.emissiveColor_r, pc.emissiveColor_g, pc.emissiveColor_b);
	
	oColor = vec4(inColor, 1.f);
	oPosition = vec4(v2fPosition, 1.f);
	oNormal = vec4(normalize(v2fNormal), 1.f);
	oMaterial = vec4(pc.roughness, pc.metalness, 1.f, 1.f);
	oEmissive = vec4(pc.emissiveColor_r, pc.emissiveColor_g, pc.emissiveColor_b, 1.f);
}
