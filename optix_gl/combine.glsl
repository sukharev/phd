// combine.glsl
//
// take N incoming textures and
// tile them

uniform sampler2D sampler0;
uniform sampler2D sampler1;
//uniform sampler2D sampler2;
//uniform sampler2D sampler3;

varying vec2 coords;
#ifdef _VERTEX_

void main() {
	gl_Position = gl_Vertex;
	coords = gl_Vertex.xy * 0.5 + 0.5;
}

#else


void main(void)
{
    if (coords.s < 0.5)
    {
            // left
            //gl_FragColor = vec4(texture2D(sampler0, coords).rgb,1.0);
            gl_FragColor = vec4(texture2D(sampler1, coords).rgb,1.0);
            //gl_FragColor = vec4(1.0,0.0,0.0,1.0);
    }
    else
    {
			// right
			gl_FragColor = texture2D(sampler1, coords.st);
	}
	//gl_FragColot = texture2D(sampler0, gl_TexCoord[0].st);
	//gl_FragColor = vec4(0.0,1.0,0.0,1.0);
}

#endif