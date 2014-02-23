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

uniform sampler1D SRSampler;
uniform sampler1D SMSampler;
uniform float minray;
uniform float maxray;
uniform float minmie;
uniform float maxmie;
const float boundary = 0.000001;

#ifdef _VERTEX_

void main() {
    gl_Position = gl_Vertex;
}

#else

float check(float test, float realval)
{
	if(test <= realval + boundary){
		if(test >= realval - boundary){
			return 0.0;
		}
	} 
	return test - realval;
}

float findInSRTexture(float muS)
{
	float diff = 0.0;
	float ind = (float)(PHASE_DISTR_RES-1)/(float)PHASE_DISTR_RES;
	float tryout = texture1D(SRSampler, ind);
	diff = check(tryout, muS);
	if(diff < 0.0)
		return ind;

	float right = ind;
	float left = 0.0;
	int i = 0;
	while(i < 1000 && diff != 0.0){
		ind = (right + left) / 2.0;
		tryout = texture1D(SRSampler, ind);
		diff = check(tryout, muS);
		if(diff > 0.0){ // go left
			right = ind;
		}
		else if (diff < 0.0){ // go right
			left = ind;
		}
		i++;
	}

	return ind;
}

float findInSMTexture(float muS)
{
	float diff = 0.0;
	float ind = (float)(PHASE_DISTR_RES-1)/(float)PHASE_DISTR_RES;
	float tryout = texture1D(SMSampler, ind);
	diff = check(tryout, muS);
	if(diff < 0.0)
		return ind;
		
	float right = ind;
	float left = 0.0;
	int i = 0;
	while(i < 1000 && diff != 0.0){
		ind = (right + left) / 2.0;
		tryout = texture1D(SMSampler, ind);
		diff = check(tryout, muS);
		if(diff > 0){ // go left
			right = ind;
		}
		else if (diff < 0){ // go right
			left = ind;
		}
		i++;
	}
	return ind;
}

void main() {
    
    float ray;
    float mie;
    float muS = (gl_FragCoord.x-0.5)/float(PHASE_DISTR_RES); // muS = cos(Theta) = x.s
    // store separately Rayleigh and Mie contributions, WITHOUT the phase function factor
    // (cf "Angular precision")
    // Rayleigh phase function
	ray = findInSRTexture(muS*(maxray-minray) + minray);
	mie = findInSMTexture(muS*(maxmie-minmie) + minmie);
    gl_FragData[0].rgb = vec3(ray,muS*(maxray-minray) + minray,0.0);
    gl_FragData[1].rgb = vec3(mie,muS*(maxmie-minmie) + minmie,0.0);
    
}

#endif