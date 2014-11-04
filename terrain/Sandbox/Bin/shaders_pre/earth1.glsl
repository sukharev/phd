/**
 * Precomputed Atmospheric Scattering
 * Copyright (c) 2008 INRIA
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holders nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Author: Eric Bruneton
 */

#define FIX

const float ISun = 100.0;

uniform vec3 c;
uniform vec3 s; //sun position
uniform mat4 projInverse;
uniform mat4 viewInverse;
uniform float exposure;
uniform float gwidth;
uniform float gheight;

uniform sampler2D reflectanceSampler;//ground reflectance texture
uniform sampler2D irradianceSampler;//precomputed skylight irradiance (E table)
uniform sampler3D inscatterSampler;//precomputed inscattered light (S table)

varying vec2 coords;
varying vec3 ray;

float PI_OVER_360 = M_PI / 360.;
#ifdef _VERTEX_

vec4 CalcUV(vec3 p)
{
   float r;
   float phi;

   r = atan(sqrt(p.x*p.x+p.y*p.y),p.z);
   r = r / M_PI; 
   phi = atan(p.y,p.x);

   vec4 rv = vec4(0.0);
   rv.x = r * cos(phi) + 0.5;
   rv.y = r * sin(phi) + 0.5;
   rv.z = r;
   rv.w = phi;
   return rv;
}

vec3 fisheye_projectionExt()
{
	float x = gl_Vertex.x *2.0/gwidth - 1.0;
	float y = gl_Vertex.y /gheight;
	
	int N = 20;
	float scale = M_PI / (float)N;
	//for (j=0;j<N/2;j++) {
	//for (i=-N/2;i<N/2;i++) {
	int j = (int)(y * (float)(N/2));
    int i = (int)(x * (float)(N/2));
    
    float theta = j * scale;
    float phi = i * 2 * scale;
    // Rotate about the z axis
    float sy = sin(theta);
    vec3 p = vec3(0.0);
    p.x = sy * sin(phi);
    p.y = sy * cos(phi);
    p.z = cos(theta);
    
    float r = p.x*p.x+p.y*p.y;
    if (r > 1 || r > 1 ||
         r > 1 || r > 1)
         return vec3(0);
    p.z = sqrt(1 - r);
    return p;
}

vec3 fisheye_projection()
{
	float x = gl_Vertex.x *2.0/gwidth - 1.0;
	float y = gl_Vertex.y *2.0/gheight - 1.0;
	//float y = gl_Vertex.y/gheight;
	float r = sqrt(x*x + y*y);
	float phi = 0.0;
	vec3 vx = vec3(0.0);
	float aperture = M_PI;
	float theta = 0.0;
	float scale = M_PI/10;///(20.0*gheight);
/*
	if(r == 0.0){
		phi = 0.0;
	}
	else if(x < 0.0){
		phi = M_PI - asin(y/r);
	}
	else if(x>= 0.0) {
		phi = asin(y/r);
	}
	*/
	phi = x * 2.0 * scale;
	theta = y * scale;
	//theta = r * aperture/2.0;
	
	vx.x = sin(theta)*cos(phi);
	vx.y = sin(theta)*sin(phi);
	vx.z = cos(theta);
	return vx;
}

void main() {
/*
    coords = gl_Vertex.xy * 0.5 + 0.5;
    //vec4 coords4 = vec4(gl_Vertex.xy,1.0,1.0);
    //vec3 coords4 = vec4(coords.x,coords.y,1.0,1.0);
    ray = (viewInverse * vec4((projInverse * gl_Vertex).xyz, 0.0)).xyz;
    //ray = (viewInverse * vec4((gl_Vertex).xyz, 0.0)).xyz;
    gl_Position = gl_Vertex;
*/    
    //coords = gl_Vertex.xy * 0.5 + 0.5;
    //ray = (viewInverse * vec4((projInverse * gl_Vertex).xyz, 0.0)).xyz;
    //vec3 vx = fisheye_projection();
    //ray = (viewInverse * vec4(vx,0.0)).xyz;
    
    //vec4 vx = CalcUV(gl_Vertex.xyy);
    //coords = vx.xy;
    //ray = (viewInverse * vec4(vx.xy,0.0,0.0)).xyz;
    
    coords = gl_Vertex.xy * 0.5 + 0.5;
    vec3 vx = fisheye_projectionExt();
    ray = (viewInverse * vec4(vx,0.0)).xyz;
    
    gl_Position = gl_Vertex;
 
}

void main1(){
	// perspective parameters:
	float fov = 270.0;
	float aspect = 1.0;
	float znear = 0.1;
	float zfar = 100.0;
	float eyesep = 0.01;
	float focal = 1.0;

	float omnipan = 2.0*M_PI;	// 360' panoramic

	// derive unit vectors / position from current modelview:
	//modelview = gl_ModelViewMatrix;
	// in actual uses, this could be derived from the modelview:
	vec3 ux, uy, uz, pos;
	vec4 vertex = gl_Vertex;
	
	float top = znear * tan(fov * PI_OVER_360);	// height of view at distance = near
	float bottom = -top;
	float left = -aspect*top;// + shift;
	float right = aspect*top;// + shift;
	float W = right-left;	
	float W2 = right+left;
	float H = top-bottom;	
	float H2 = top+bottom;
	float D = zfar-znear;	
	float D2 = zfar+znear;
	float n2 = znear*2.;
	float fn2 = zfar*n2;
	
	
	///////////////////////////////////////////////////////////////
	// standard stuff:
	///////////////////////////////////////////////////////////////
	// get coordinate in eye-space:

	vec4 ecv = vertex;
	//vec4 ecv = modelview * vertex;

	// get polar coordinates:
	float azimuth = atan(ecv.x, -ecv.z); 
	float elevation = atan(ecv.y, length(ecv.xz));

	// standard perspective projection:
	//gl_Position = projection * modelview * vertex;
	gl_Position.x = ecv.x*n2/W;	
	gl_Position.y = ecv.y*n2/H;	
	gl_Position.z = ecv.z*-D2/D + ecv.w*-fn2/D;
	gl_Position.w = ecv.z*-1.;
	float ofovx = fov * M_PI/180.;
	float ofovy = ofovx;

	// map azimuth/elevation to the XY clipspace
	gl_Position.x = azimuth/M_PI; 
	gl_Position.y = elevation/M_PI; 

	// map (adjusted) depth z to clip space:
	gl_Position.z = (length(ecv)-znear)/(zfar-znear);
	gl_Position.w = 1.;
	ray = (/*viewInverse **/ vec4((projInverse*vec4(gl_Position.x,gl_Position.y,1.0,0.0)).xyz,0.0)).xyz;
    //ray = (/*viewInverse **/ vec4((/*projInverse **/ gl_Vertex).xyz, 0.0)).xyz;
/*
	coords = gl_Vertex.xy * 0.5 + 0.5;
    vec4 coords4 = vec4(gl_Vertex.xy,1.0,1.0);
    //vec3 coords4 = vec4(coords.x,coords.y,1.0,1.0);
    //ray = (viewInverse * vec4((projInverse * gl_Vertex).xyz, 0.0)).xyz;
    ray = (viewInverse * vec4(( coords4).xyz, 0.0)).xyz;
    gl_Position = gl_Vertex;
*/
}

void main_bad() {
	coords = gl_Vertex.xy * 0.5 + 0.5;
    vec4 coords4 = vec4(gl_Vertex.xy,1.0,1.0);
    
    float aperture = 360.0;
    float apertureHalf = 0.5 * aperture * (M_PI / 180.0);

	// This factor ajusts the coordinates in the case that
	// the aperture angle is less than 180 degrees, in which
	// case the area displayed is not the entire half-sphere.
	float maxFactor = sin(apertureHalf);
	float xv = coords4.x/gwidth;
	float yv = coords4.y/gheight;  
	
	//vec2 pos = 2.0 * gl_TexCoord[0].st - 1.0;  
	//float l = length(pos);
	//if (l > 1.0) {
	//	gl_FragColor = vec4(0, 0, 0, 1);  
	//} else {
	float x = maxFactor * xv;
	float y = maxFactor * yv;

	float n = length(vec2(x, y));

	float z = sqrt(1.0 - n * n);

	float r = atan(n, z) / M_PI; 

	float phi = atan(y, x);

	float u = r * cos(phi) + 0.5;
	float v = r * sin(phi) + 0.5;
	//}

    //vec3 coords4 = vec4(coords.x,coords.y,1.0,1.0);
    //ray = (viewInverse * vec4((projInverse * gl_Vertex).xyz, 0.0)).xyz;
    //ray = (viewInverse * vec4((projInverse * coords4).xyz, 0.0)).xyz;
    ray = (/*viewInverse * */vec4(xv*gwidth,yv*gheight,1.0,0.0)).xyz;
    gl_Position = gl_Vertex;
}

#else

//inscattered light along ray x+tv, when sun in direction s (=S[L]-T(x,x0)S[L]|x0)
vec3 inscatter(inout vec3 x, inout float t, vec3 v, vec3 s, out float r, out float mu, out vec3 attenuation) {
    vec3 result;
    r = length(x);
    mu = dot(x, v) / r;
    float d = -r * mu - sqrt(r * r * (mu * mu - 1.0) + Rt * Rt);
    if (d > 0.0) { // if x in space and ray intersects atmosphere
        // move x to nearest intersection of ray with top atmosphere boundary
        x += d * v;
        t -= d;
        mu = (r * mu + d) / Rt;
        r = Rt;
    }
    
    if (r <= Rt) { // if ray intersects atmosphere
        float nu = dot(v, s);
        float muS = dot(x, s) / r;
        float phaseR = phaseFunctionR(nu);
        float phaseM = phaseFunctionM(nu);
        vec4 inscatter = max(texture4D(inscatterSampler, r, mu, muS, nu), 0.0);
        if (t > 0.0) {
            vec3 x0 = x + t * v;
            float r0 = length(x0);
            float rMu0 = dot(x0, v);
            float mu0 = rMu0 / r0;
            float muS0 = dot(x0, s) / r0;
#ifdef FIX
            // avoids imprecision problems in transmittance computations based on textures
            attenuation = analyticTransmittance(r, mu, t);
#else
            attenuation = transmittance(r, mu, v, x0);
#endif
            if (r0 > Rg + 0.01) {
                // computes S[L]-T(x,x0)S[L]|x0
                inscatter = max(inscatter - attenuation.rgbr * texture4D(inscatterSampler, r0, mu0, muS0, nu), 0.0);
#ifdef FIX
                // avoids imprecision problems near horizon by interpolating between two points above and below horizon
                const float EPS = 0.004;
                float muHoriz = -sqrt(1.0 - (Rg / r) * (Rg / r));
                if (abs(mu - muHoriz) < EPS) {
                    float a = ((mu - muHoriz) + EPS) / (2.0 * EPS);

                    mu = muHoriz - EPS;
                    r0 = sqrt(r * r + t * t + 2.0 * r * t * mu);
                    mu0 = (r * mu + t) / r0;
                    vec4 inScatter0 = texture4D(inscatterSampler, r, mu, muS, nu);
                    vec4 inScatter1 = texture4D(inscatterSampler, r0, mu0, muS0, nu);
                    vec4 inScatterA = max(inScatter0 - attenuation.rgbr * inScatter1, 0.0);

                    mu = muHoriz + EPS;
                    r0 = sqrt(r * r + t * t + 2.0 * r * t * mu);
                    mu0 = (r * mu + t) / r0;
                    inScatter0 = texture4D(inscatterSampler, r, mu, muS, nu);
                    inScatter1 = texture4D(inscatterSampler, r0, mu0, muS0, nu);
                    vec4 inScatterB = max(inScatter0 - attenuation.rgbr * inScatter1, 0.0);

                    inscatter = mix(inScatterA, inScatterB, a);
                }
#endif
            }
        }
        else{
			//result = vec3(1.0,0.0,0.0);
			//return result * ISun;
        }
#ifdef FIX
        // avoids imprecision problems in Mie scattering when sun is below horizon
        inscatter.w *= smoothstep(0.00, 0.02, muS);
#endif
        //result = max(inscatter.rgb * phaseR + getMie(inscatter) * phaseM, 0.0);
        //if(mu < 0.5 && mu > -0.9){
			//inscatter.rgb = vec3(0.0,0.0,1.0);

			if(nu >= 0.9){
				inscatter.rgb = vec3(0.0,0.0,1.0);
			}else if(nu < 0.5 && nu > 0.3){
				inscatter.rgb = vec3(0.0,1.0,0.0);
			}
			else if(nu <= 0.1 && nu > -0.2){
				inscatter.rgb = vec3(1.0,0.0,0.0);
			}
			else if(nu <= -0.2 && nu > -0.9){
				inscatter.rgb = vec3(1.0,1.0,0.0);
			}

		//}
		result = max(inscatter.rgb * phaseR, 0.0);
    } else { // x in space and ray looking in space
        result = vec3(0.0);
    }
    
    return result * ISun;
}

//ground radiance at end of ray x+tv, when sun in direction s
//attenuated bewteen ground and viewer (=R[L0]+R[L*])
vec3 groundColor(vec3 x, float t, vec3 v, vec3 s, float r, float mu, vec3 attenuation)
{
    vec3 result;
    if (t > 0.0) { // if ray hits ground surface
        // ground reflectance at end of ray, x0
        vec3 x0 = x + t * v;
        float r0 = length(x0);
        vec3 n = x0 / r0;
        vec2 coords = vec2(atan(n.y, n.x), acos(n.z)) * vec2(0.5, 1.0) / M_PI + vec2(0.5, 0.0);
        vec4 reflectance = texture2D(reflectanceSampler, coords) * vec4(0.2, 0.2, 0.2, 1.0);
        if (r0 > Rg + 0.01) {
            reflectance = vec4(0.4, 0.4, 0.4, 0.0);
        }

        // direct sun light (radiance) reaching x0
        float muS = dot(n, s);
        vec3 sunLight = transmittanceWithShadow(r0, muS);

        // precomputed sky light (irradiance) (=E[L*]) at x0
        vec3 groundSkyLight = vec3(0.0); //irradiance(irradianceSampler, r0, muS);

        // light reflected at x0 (=(R[L0]+R[L*])/T(x,x0))
        vec3 groundColor = reflectance.rgb * (max(muS, 0.0) * sunLight + groundSkyLight) * ISun / M_PI;

        // water specular color due to sunLight
        if (reflectance.w > 0.0) {
            vec3 h = normalize(s - v);
            float fresnel = 0.02 + 0.98 * pow(1.0 - dot(-v, h), 5.0);
            float waterBrdf = fresnel * pow(max(dot(h, n), 0.0), 150.0);
            groundColor += reflectance.w * max(waterBrdf, 0.0) * sunLight * ISun;
        }

        result = attenuation * groundColor; //=R[L0]+R[L*]
    } else { // ray looking at the sky
        result = vec3(0.0);
    }
    return result;
}

// direct sun light for ray x+tv, when sun in direction s (=L0)
vec3 sunColor(vec3 x, float t, vec3 v, vec3 s, float r, float mu) {
    if (t > 0.0) {
        return vec3(0.0);
    } else {
        vec3 transmittance = r <= Rt ? transmittanceWithShadow(r, mu) : vec3(1.0); // T(x,xo)
        float isun = step(cos(M_PI / 180.0), dot(v, s)) * ISun; // Lsun
        return transmittance * isun; // Eq (9)
    }
}

vec3 HDR(vec3 L) {
    L = L * exposure;
    L.r = L.r < 1.413 ? pow(L.r * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.r);
    L.g = L.g < 1.413 ? pow(L.g * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.g);
    L.b = L.b < 1.413 ? pow(L.b * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.b);
    return L;
}

void main() {
    vec3 x = c;
    vec3 v = normalize(ray);

    float r = length(x);
    float mu = dot(x, v) / r;
    float t = -r * mu - sqrt(r * r * (mu * mu - 1.0) + Rg * Rg);
/*
    vec3 g = x - vec3(0.0, 0.0, Rg + 10.0);
    float a = v.x * v.x + v.y * v.y - v.z * v.z;
    float b = 2.0 * (g.x * v.x + g.y * v.y - g.z * v.z);
    float c = g.x * g.x + g.y * g.y - g.z * g.z;
    float d = -(b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
    bool cone = d > 0.0 && abs(x.z + d * v.z - Rg) <= 10.0;

    if (t > 0.0) {
        if (cone && d < t) {
            t = d;
        }
    } else if (cone) {
        t = d;
    }
*/

    vec3 attenuation;
	vec3 inscatterColor = inscatter(x, t, v, s, r, mu, attenuation); //S[L]-T(x,xs)S[l]|xs
    vec3 groundColor = groundColor(x, t, v, s, r, mu, attenuation); //R[L0]+R[L*]
    vec3 sunColor = sunColor(x, t, v, s, r, mu); //L0
    gl_FragColor = vec4(HDR(sunColor + groundColor + inscatterColor), 1.0); // Eq (16)


    //gl_FragColor = texture3D(inscatterSampler,vec3(coords,(s.x+1.0)/2.0));
    //gl_FragColor = vec4(texture2D(irradianceSampler,coords).rgb*5.0, 1.0);
    //gl_FragColor = texture2D(transmittanceSampler,coords);
}

#endif
