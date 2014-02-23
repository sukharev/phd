uniform vec3 c;
uniform vec3 s;
uniform mat4 projInverse;
uniform mat4 viewInverse;
uniform mat4 modelInverse;
uniform float flickerFactor;
#ifdef _VERTEX_

void main() {
	//vec4 clipCoord = gl_ModelViewProjectionMatrix * gl_Vertex;
	//vec4 clipCoord =  = (modelInverse*vec4(viewInverse * vec4((projInverse * gl_Vertex).xyz, 0.0)).xyz,0.0).xyz;
    vec4 clipCoord =  = vec4(viewInverse * vec4((projInverse * gl_Vertex).xyz, 0.0)).xyz;
    gl_Position = gl_Vertex;
    gl_Position = clipCoord;
    gl_FrontColor = gl_Color;
    vec4 ndc = vec4(clipCoord.xyz, 0) / clipCoord.w;
    gl_FrontSecondaryColor = (ndc * 0.5) + 0.5;
}

#else



void main() {
	// Mix primary and secondary colors, 50/50
    vec4 temp = mix(gl_Color, vec4(vec3(gl_SecondaryColor), 1.0), 0.5);
    // Multiply by flicker factor
    gl_FragColor = temp * flickerFactor;
    vec4(vec3(dot(sample[4].rgb, vec3(0.3, 0.59, 0.11))), 1.0);
}
#endif
