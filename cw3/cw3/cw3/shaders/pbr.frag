#version 450 

layout( location = 0 ) in vec2 v2fUV;
layout( location = 1 ) in vec3 v2fCamPos;
layout( location = 2 ) in vec3 v2fLightPos;
layout( location = 3 ) in vec2 v2fBoundary;



layout( set = 1, binding = 0 ) uniform sampler2D sam_baseColor; 
layout( set = 1, binding = 1 ) uniform sampler2D sam_position; 
layout( set = 1, binding = 2 ) uniform sampler2D sam_normal; 
layout( set = 1, binding = 3 ) uniform sampler2D sam_material; 
layout( set = 1, binding = 4 ) uniform sampler2D sam_emissive; 




layout( location = 0 ) out vec4 oColor; 
layout( location = 1 ) out vec4 oFilter; 

void main() 
{ 

	vec2 viewport_offset = vec2(0.f, 0.f);

	if (gl_FragCoord.x < viewport_offset.x ||
        gl_FragCoord.x > v2fBoundary.x ||
        gl_FragCoord.y < viewport_offset.y ||
        gl_FragCoord.y > v2fBoundary.y )
    {
        discard;
    }


	// epsilon constant
	const float epsilon = 1e-12;
	const float PI = 3.14159f;

	vec2 fliped_uv = vec2(v2fUV.x, 1.0 - v2fUV.y);
	vec3 o_base_col = texture(sam_baseColor, fliped_uv).rgb;
	o_base_col = o_base_col / (o_base_col + 1.f);
	vec3 o_position = texture(sam_position, fliped_uv).rgb;
	vec3 o_normal = normalize(texture(sam_normal, fliped_uv).rgb);
	vec2 o_material = texture(sam_material, fliped_uv).rg;
	float o_roughness = o_material.x;
	float o_metalness = o_material.y;
	vec3 o_emissive = texture(sam_emissive, fliped_uv).rgb;

	vec3 o_lightDir = normalize(v2fLightPos - o_position);
	vec3 o_viewDir = normalize(v2fCamPos - o_position);
	vec3 o_halfVec = normalize((o_viewDir + o_lightDir) / 2.f);


	////////////////////////////////////////////
	//		  calculation of F				  //
	////////////////////////////////////////////

	vec3 F0 = (1.f - o_metalness) * vec3(0.04f) + o_base_col * o_metalness;			// base reflectivity
	float oneMinusHV = (1.f - dot(o_halfVec, o_viewDir));		// (1 - h * v)
	float oneMinusHV_pow5 = oneMinusHV * oneMinusHV * oneMinusHV * oneMinusHV * oneMinusHV;		// (1 - h * v) ^5

	vec3 oFresnel = F0 + (1.f - F0) * oneMinusHV_pow5;


	///////////////////////////////////////////
	//		  calculation of D		         //
	///////////////////////////////////////////


	// calculate shininess
	float oRoughness_pow2 = o_roughness * o_roughness;			// pow of 2
	float oRoughness_pow4 = oRoughness_pow2 * oRoughness_pow2;	// pow of 4
	float oShininess = (2.f / (oRoughness_pow4 + epsilon) ) - 2.f;

	float D_first_term  = (oShininess + 2.f) / (2.f * PI);		// (ap + 2) / 2 * PI
	float D_second_term = pow(max(0.f, dot(o_normal, o_halfVec)), oShininess);

	float oDist = D_first_term * D_second_term;			// D(h), normal distribution function term


	///////////////////////////////////////////
	//		  calculation of G		         //
	///////////////////////////////////////////


	float n_dot_h = max(0.f, dot(o_normal, o_halfVec));
	float n_dot_v = max(0.f, dot(o_normal, o_viewDir));
	float n_dot_l = max(0.f, dot(o_normal, o_lightDir));
	float v_dot_h = dot(o_viewDir, o_halfVec);

	float oMask = min( 1.f, min( 2.f * n_dot_h * n_dot_v / (v_dot_h + epsilon), 2.f * n_dot_h * n_dot_l / (v_dot_h + epsilon) ) );		// G(l,v)


	///////////////////////////////////////////
	//		  calculation of BRDF		     //
	///////////////////////////////////////////

	
	vec3 oDiffuse = o_base_col / PI * (vec3(1.f) - oFresnel) * (1.f - o_metalness);
	vec3 oBRDF = oDiffuse + oDist * oFresnel * oMask / (4.f * n_dot_v * n_dot_l + epsilon);

	vec3 oAmbient = vec3(0.02f) * o_base_col;
	vec3 oBlinnPhong = o_emissive + oAmbient + oBRDF * n_dot_l;

	// final output color
	oColor = vec4(oBlinnPhong, 1.f);
	//oColor = vec4(vec3(oBaseColor), 1.f);

	if (oColor.x > 1.f || oColor.y > 1.f || oColor.z > 1.f)
	{
		oFilter = oColor;
	}
	else
	{
		oFilter = vec4(0.f, 0.f ,0.f, 0.f);
	}
} 