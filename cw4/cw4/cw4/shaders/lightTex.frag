#version 450 

layout( location = 0 ) in vec2 v2fTexCoord;
layout( location = 1 ) in vec3 v2fViewDir;
layout( location = 2 ) in vec3 v2fLightPos;
layout( location = 3 ) in vec3 v2fNormal;
layout( location = 4 ) in vec3 v2fPosition;
layout( location = 5 ) in vec4 v2fTangent;
layout( location = 6 ) in vec4 v2fPositionLightSpace;




layout( set = 1, binding = 0 ) uniform sampler2D uTexColor; 
layout( set = 1, binding = 1 ) uniform sampler2D uMetalColor; 
layout( set = 1, binding = 2 ) uniform sampler2D uRoughColor; 
layout( set = 1, binding = 3 ) uniform sampler2D uNormalMap; 
layout( set = 1, binding = 4 ) uniform sampler2D uShadowMap; 

layout( location = 0 ) out vec4 oColor; 

void main() 
{ 
	// epsilon constant
	const float epsilon = 1e-12;
	const float PI = 3.14159f;

	// calculation of vec/mat
	vec3 lightDirection = normalize(v2fLightPos - v2fPosition);			// light direction
	vec3 ViewDir		= normalize(v2fViewDir);						// view direction
	vec3 halfVec		= normalize((ViewDir + lightDirection));	// half vector
	vec3 surfaceNormal	= normalize(v2fNormal);							// normalized normal
	vec3 oTangent		= normalize(vec3(v2fTangent));					// tangent
	vec3 oBaseColor		= texture( uTexColor, v2fTexCoord ).rgb;		// base color	
	float oMetalness	= texture( uMetalColor, v2fTexCoord ).r;		// material metalness 
	float oRoughness	= texture( uRoughColor, v2fTexCoord ).r;		// material roughness

	vec3 oNormalSamp	= texture( uNormalMap, v2fTexCoord ).xyz;		// material normal sample
	oNormalSamp = normalize(oNormalSamp * 2.f - 1.f);					//	transform to [-1, 1]

	vec3 biTangent		= normalize(v2fTangent.w * cross(v2fNormal, oTangent));		// biTangent

	mat3 mat_TBN = mat3(	oTangent, biTangent, surfaceNormal	);		// TBN mat
	vec3 finalNormal = normalize(mat_TBN * oNormalSamp);				// calculate normal for BRDF


	// shadow mapping
	vec3 projCoords = v2fPositionLightSpace.xyz / v2fPositionLightSpace.w;
	projCoords = projCoords * 0.5f + 0.5f;
	float closestDepth = texture(uShadowMap, projCoords.xy).r;


	////////////////////////////////////////////
	//		  calculation of F				  //
	////////////////////////////////////////////

	vec3 F0 = (1.f - oMetalness) * vec3(0.04f) + oBaseColor * oMetalness;			// base reflectivity
	float oneMinusHV = (1.f - dot(halfVec, ViewDir));		// (1 - h * v)
	float oneMinusHV_pow5 = oneMinusHV * oneMinusHV * oneMinusHV * oneMinusHV * oneMinusHV;		// (1 - h * v) ^5

	vec3 oFresnel = F0 + (1.f - F0) * oneMinusHV_pow5;


	///////////////////////////////////////////
	//		  calculation of D		         //
	///////////////////////////////////////////


	// calculate shininess
	float oRoughness_pow2 = oRoughness * oRoughness;			// pow of 2
	float oRoughness_pow4 = oRoughness_pow2 * oRoughness_pow2;	// pow of 4
	float oShininess = (2.f / (oRoughness_pow4 + epsilon) ) - 2.f;

	float D_first_term  = (oShininess + 2.f) / (2.f * PI);		// (ap + 2) / 2 * PI
	float D_second_term = pow(max(0.f, dot(finalNormal, halfVec)), oShininess);

	float oDist = D_first_term * D_second_term;			// D(h), normal distribution function term


	///////////////////////////////////////////
	//		  calculation of G		         //
	///////////////////////////////////////////


	float n_dot_h = max(0.f, dot(finalNormal, halfVec));
	float n_dot_v = max(0.f, dot(finalNormal, ViewDir));
	float n_dot_l = max(0.f, dot(finalNormal, lightDirection));
	float v_dot_h = dot(ViewDir, halfVec);

	float oMask = min( 1.f, min( 2.f * n_dot_h * n_dot_v / (v_dot_h + epsilon), 2.f * n_dot_h * n_dot_l / (v_dot_h + epsilon) ) );		// G(l,v)


	///////////////////////////////////////////
	//		  calculation of BRDF		     //
	///////////////////////////////////////////

	
	vec3 oDiffuse = oBaseColor / PI * (vec3(1.f) - oFresnel) * (1.f - oMetalness);
	vec3 oBRDF = oDiffuse + oDist * oFresnel * oMask / (4.f * n_dot_v * n_dot_l + epsilon);

	vec3 oAmbient = vec3(0.02f) * oBaseColor;
	vec3 oBlinnPhong = oAmbient + oBRDF * n_dot_l;

	//oBlinnPhong = (oBlinnPhong / (1.f + oBlinnPhong));

//	float currentDepth = projCoords.z;
//	if (currentDepth > closestDepth+epsilon) {
//		oBlinnPhong = oBlinnPhong * 0.1f;
//	} 
//
	// final output color
	oColor = vec4(oBlinnPhong, 1.f);
	//oColor = vec4(vec3(oShadow), 1.f);
} 