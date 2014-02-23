
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#pragma once

#include <optixu/optixu_math_namespace.h>
#include <optix_world.h>

using namespace optix;
#define TRANSMITTANCE_NON_LINEAR

// Convert a float3 in [0,1)^3 to a uchar4 in [0,255]^4 -- 4th channel is set to 255
#ifdef __CUDACC__
__device__ __inline__ optix::uchar4 make_color(const optix::float3& c)
{
    return optix::make_uchar4( static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
                               static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
                               static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
                               255u);                                                 /* A */
}
#endif

// Sample Phong lobe relative to U, V, W frame
__host__ __device__ __inline__ optix::float3 sample_phong_lobe( optix::float2 sample, float exponent, 
                                                                optix::float3 U, optix::float3 V, optix::float3 W )
{
  const float power = expf( logf(sample.y)/(exponent+1.0f) );
  const float phi = sample.x * 2.0f * (float)M_PIf;
  const float scale = sqrtf(1.0f - power*power);
  
  const float x = cosf(phi)*scale;
  const float y = sinf(phi)*scale;
  const float z = power;

  return x*U + y*V + z*W;
}

// Create ONB from normal.  Resulting W is parallel to normal
__host__ __device__ __inline__ void create_onb( const optix::float3& n, optix::float3& U, optix::float3& V, optix::float3& W )
{
  using namespace optix;

  W = normalize( n );
  U = cross( W, optix::make_float3( 0.0f, 1.0f, 0.0f ) );

  if ( fabs( U.x ) < 0.001f && fabs( U.y ) < 0.001f && fabs( U.z ) < 0.001f  )
    U = cross( W, make_float3( 1.0f, 0.0f, 0.0f ) );

  U = normalize( U );
  V = cross( W, U );
}

// Create ONB from normalized vector
__device__ __inline__ void create_onb( const optix::float3& n, optix::float3& U, optix::float3& V)
{
  using namespace optix;

  U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );

  if ( dot( U, U ) < 1e-3f )
    U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );

  U = normalize( U );
  V = cross( n, U );
}

// Compute the origin ray differential for transfer
__host__ __device__ __inline__ optix::float3 differential_transfer_origin(optix::float3 dPdx, optix::float3 dDdx, float t, optix::float3 direction, optix::float3 normal)
{
  float dtdx = -optix::dot((dPdx + t*dDdx), normal)/optix::dot(direction, normal);
  return (dPdx + t*dDdx)+dtdx*direction;
}

// Compute the direction ray differential for a pinhole camera
__host__ __device__ __inline__ optix::float3 differential_generation_direction(optix::float3 d, optix::float3 basis)
{
  float dd = optix::dot(d,d);
  return (dd*basis-optix::dot(d,basis)*d)/(dd*sqrtf(dd));
}

// Compute the direction ray differential for reflection
__host__ __device__ __inline__ 
optix::float3 differential_reflect_direction(optix::float3 dPdx, optix::float3 dDdx, optix::float3 dNdP, 
                                             optix::float3 D, optix::float3 N)
{
  using namespace optix;

  float3 dNdx = dNdP*dPdx;
  float dDNdx = dot(dDdx,N) + dot(D,dNdx);
  return dDdx - 2*(dot(D,N)*dNdx + dDNdx*N);
}

// Compute the direction ray differential for refraction
__host__ __device__ __inline__ 
optix::float3 differential_refract_direction(optix::float3 dPdx, optix::float3 dDdx, optix::float3 dNdP, 
                                             optix::float3 D, optix::float3 N, float ior, optix::float3 T)
{
  using namespace optix;

  float eta;
  if(dot(D,N) > 0.f) {
    eta = ior;
    N = -N;
  } else {
    eta = 1.f / ior;
  }

  float3 dNdx = dNdP*dPdx;
  float mu = eta*dot(D,N)-dot(T,N);
  float TN = -sqrtf(1-eta*eta*(1-dot(D,N)*dot(D,N)));
  float dDNdx = dot(dDdx,N) + dot(D,dNdx);
  float dmudx = (eta - (eta*eta*dot(D,N))/TN)*dDNdx;
  return eta*dDdx - (mu*dNdx+dmudx*N);
}

// Color space conversions
__host__ __device__ __inline__ optix::float3 Yxy2XYZ( const optix::float3& Yxy )
{
  return optix::make_float3(  Yxy.y * ( Yxy.x / Yxy.z ),
                              Yxy.x,
                              ( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x / Yxy.z ) );
}

__host__ __device__ __inline__ optix::float3 XYZ2rgb( const optix::float3& xyz)
{
  const float R = optix::dot( xyz, optix::make_float3(  3.2410f, -1.5374f, -0.4986f ) );
  const float G = optix::dot( xyz, optix::make_float3( -0.9692f,  1.8760f,  0.0416f ) );
  const float B = optix::dot( xyz, optix::make_float3(  0.0556f, -0.2040f,  1.0570f ) );
  return optix::make_float3( R, G, B );
}

__host__ __device__ __inline__ optix::float3 Yxy2rgb( optix::float3 Yxy )
{
  using namespace optix;

  // First convert to xyz
  float3 xyz = make_float3( Yxy.y * ( Yxy.x / Yxy.z ),
                            Yxy.x,
                            ( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x / Yxy.z ) );

  const float R = dot( xyz, make_float3(  3.2410f, -1.5374f, -0.4986f ) );
  const float G = dot( xyz, make_float3( -0.9692f,  1.8760f,  0.0416f ) );
  const float B = dot( xyz, make_float3(  0.0556f, -0.2040f,  1.0570f ) );
  return make_float3( R, G, B );
}

__host__ __device__ __inline__ optix::float3 rgb2Yxy( optix::float3 rgb)
{
  using namespace optix;

  // convert to xyz
  const float X = dot( rgb, make_float3( 0.4124f, 0.3576f, 0.1805f ) );
  const float Y = dot( rgb, make_float3( 0.2126f, 0.7152f, 0.0722f ) );
  const float Z = dot( rgb, make_float3( 0.0193f, 0.1192f, 0.9505f ) );

  // convert xyz to Yxy
  return make_float3( Y, 
                      X / ( X + Y + Z ),
                      Y / ( X + Y + Z ) );
}




// ----------------------------------------------------------------------------
// UTILITY FUNCTIONS
// ----------------------------------------------------------------------------

__host__ __device__ __inline__ float4 max(float4 a, float4 b)
{
    return make_float4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

__host__ __device__ __inline__ float3 max(float3 a, float3 b)
{
    return make_float3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

// nearest intersection of ray r,mu with ground or top atmosphere boundary
// mu=cos(ray zenith angle at ray origin)
__host__ __device__ __inline__ float limit(float r, float mu, float RL, float Rg) {
    float dout = -r * mu + sqrt(r * r * (mu * mu - 1.0) + RL * RL);
    float delta2 = r * r * (mu * mu - 1.0) + Rg * Rg;
    if (delta2 >= 0.0) {
        float din = -r * mu - sqrt(delta2);
        if (din >= 0.0) {
            dout = min(dout, din);
        }
    }
    return dout;
}

__host__ __device__ __inline__ float opticalDepth(float H, float r, float mu, 
												float Rg, float Rt, float RL,
												float TRANSMITTANCE_INTEGRAL_SAMPLES) {
    float result = 0.0;
    float dx = limit(r, mu, RL, Rg) / float(TRANSMITTANCE_INTEGRAL_SAMPLES);
    float xi = 0.0;
    float yi = exp(-(r - Rg) / H);
    for (int i = 1; i <= TRANSMITTANCE_INTEGRAL_SAMPLES; ++i) {
        float xj = float(i) * dx;
        float yj = exp(-(sqrt(r * r + xj * xj + 2.0 * xj * r * mu) - Rg) / H);
        result += (yi + yj) / 2.0 * dx;
        xi = xj;
        yi = yj;
    }
    return mu < -sqrt(1.0 - (Rg / r) * (Rg / r)) ? 1e9 : result;
}

__host__ __device__ __inline__ optix::float2 sign(optix::float2 val)
{
	optix::float2 res = make_float2(0.0);
	if(val.x >= 0)
		res.x = 1;
	else
		res.x = -1;

	if(val.y >= 0)
		res.y = 1;
	else
		res.y = -1;
	return res;
}

__host__ __device__ __inline__ optix::float2 abs2(optix::float2 val)
{
	return make_float2(abs(val.x), abs(val.y));
}

__host__ __device__ __inline__ optix::float2 sqrt2(optix::float2 val)
{
	return make_float2(sqrt(val.x), sqrt(val.y));
}

// optical depth for ray (r,mu) of length d, using analytic formula
// (mu=cos(view zenith angle)), intersections with ground ignored
// H=height scale of exponential density function
__host__ __device__ __inline__ float  opticalDepthAnalytic(float H, float r, float mu, float d,
												   float Rg, float Rt, float RL) {
    float a = sqrt((0.5/H)*r);
    optix::float2 a01 = a*make_float2(mu, mu + d / r);
    optix::float2 a01s = sign(a01);
    optix::float2 a01sq = a01*a01;
    float x = a01s.y > a01s.x ? exp(a01sq.x) : 0.0;
    optix::float2 y = a01s / (2.3193*abs2(a01) + sqrt2(1.52*a01sq + 4.0)) * make_float2(1.0, expf(-d/H*(d/(2.0*r)+mu)));
    return sqrt((6.2831*H)*r) * exp((Rg-r)/H) * (x + dot(y, make_float2(1.0, -1.0)));
}

// transmittance(=transparency) of atmosphere for ray (r,mu) of length d
// (mu=cos(view zenith angle)), intersections with ground ignored
// uses analytic formula instead of transmittance texture
__host__ __device__ __inline__ optix::float3  TransmittanceAtmosphereAnalytic(float r, float mu, float d,
																	float Rg, float Rt, float RL,
																	optix::float3 betaR, optix::float3 betaMEx,
																	float HR, float HM) {
    return expf(- betaR * opticalDepthAnalytic(HR, r, mu, d, Rg, Rt, RL) - betaMEx * opticalDepthAnalytic(HM, r, mu, d, Rg, Rt, RL));
}



__host__ __device__ __inline__ optix::float3 opticalDepth(optix::float3 & x, optix::float3 & x0, optix::float3 sigmaE){
	optix::float3 d = x - x0;
	float dist = sqrtf(optix::dot(d,d));
	return dist * (sigmaE);
}

__host__ __device__ __inline__ optix::float3 trans(optix::float3 & x, optix::float3 & x0, optix::float3 sigmaE)
{
	optix::float3 tau = opticalDepth(x, x0, sigmaE);
	return optix::expf(-tau);
}


__host__ __device__ __inline__ optix::float3 tranAtmIntgr(float r, float mu, float d,
																	float Rg, float Rt, float RL,
																	optix::float3 betaR, optix::float3 betaMEx,
																	float HR, float HM, float TRANSMITTANCE_INTEGRAL_SAMPLES) 
{
	float opt_depthR = opticalDepth(HR, r, mu, Rg, Rt, RL, TRANSMITTANCE_INTEGRAL_SAMPLES);
	float opt_depthMEx = opticalDepth(HM, r, mu, Rg, Rt, RL, TRANSMITTANCE_INTEGRAL_SAMPLES);

	optix::float3 tau = betaR * opt_depthR + betaMEx * opt_depthMEx;
	return optix::expf(-tau);
}



// approximated single Mie scattering (cf. approximate Cm in paragraph "Angular precision")
//optix::float3 getMie(optix::float4 rayMie) { // rayMie.rgb=C*, rayMie.w=Cm,r
//	return rayMie.rgb * rayMie.w / max(rayMie.r, 1e-4) * (betaR.r / betaR);
//}


/*
__host__ __device__ __inline__ optix::float3 get_sigma_a()
{
	using namespace optix;
	return make_float3( 0.05, 0.05, 0.05);
}

__host__ __device__ __inline__ optix::float3 get_sigma_s()
{
	using namespace optix;
	return make_float3( 0.0058, 0.0135, 0.0331);
}

__host__ __device__ __inline__ float PhaseRayleigh(const optix::float3 &w, const optix::float3 &wp) {
	float costheta = dot(w, wp);
	return  3.f/(16.f*M_PIf) * (1 + costheta * costheta);
}
*/
//"color sigma_a" [.05 .05 .05 ]
//"color sigma_s" [.0058 .0135 .0331]

// Rayleigh phase function
__host__ __device__ __inline__ float phaseFunctionR(float mu) 
{
    return (3.0 / (16.0 * M_PI)) * (1.0 + mu * mu);
}

// Mie phase function
__host__ __device__ __inline__ float phaseFunctionM(float mu, float mieG) 
{	
	if(1.0 + (mieG*mieG) - 2.0*mieG*mu < 0.0)
		return 0.0;
	return 1.5 * 1.0 / (4.0 * M_PI) * (1.0 - mieG*mieG) * powf(1.0 + (mieG*mieG) - 2.0*mieG*mu, -3.0/2.0) * (1.0 + mu * mu) / (2.0 + mieG*mieG);
}

// approximated single Mie scattering (cf. approximate Cm in paragraph "Angular precision")
__host__ __device__ __inline__ float3 getMie(float4 rayMie, optix::float3 betaR) 
{ // rayMie.rgb=C*, rayMie.w=Cm,r
	return make_float3(rayMie.x,rayMie.y,rayMie.z) * rayMie.w / max(rayMie.x, 1e-4) * (betaR.x / betaR);
}

__host__ __device__ __inline__ float analyticOpticalDepth(float H, float r, float mu, float d, float Rg) {
    float a = sqrt((0.5/H)*r);
    float2 a01 = a*make_float2(mu, mu + d / r);
    float2 a01s = sign(a01);
    float2 a01sq = a01*a01;
    float x = a01s.y > a01s.x ? exp(a01sq.x) : 0.0;
    float2 y = a01s / (2.3193*abs2(a01) + sqrt2(1.52*a01sq + 4.0)) * make_float2(1.0, exp(-d/H*(d/(2.0*r)+mu)));
    return sqrt((6.2831*H)*r) * exp((Rg-r)/H) * (x + dot(y, make_float2(1.0, -1.0)));
}

// transmittance(=transparency) of atmosphere for ray (r,mu) of length d
// (mu=cos(view zenith angle)), intersections with ground ignored
// uses analytic formula instead of transmittance texture
__host__ __device__ __inline__ float3 analyticTransmittance(float r, float mu, float d, 
						   float HR, float HM, float Rg,
						   optix::float3 betaR, optix::float3 betaMEx) {
    return expf(- betaR * analyticOpticalDepth(HR, r, mu, d, Rg) - betaMEx * analyticOpticalDepth(HM, r, mu, d, Rg));
}


__host__ __device__ __inline__ void coordinateSystem(const optix::float3 v1, 
													 optix::float3 *v2,
													 optix::float3 *v3) 
{
	if(fabsf(v1.x) > fabsf(v1.y)){
		float invLen = 1.0f / sqrtf(v1.x*v1.x + v1.z*v1.z);
		*v2 = optix::make_float3(-v1.z*invLen, 0.0f, v1.x*invLen);
	}
	else{
		float invLen = 1.0f / sqrtf(v1.y*v1.y + v1.z*v1.z);
		*v2 = optix::make_float3(0.0f, v1.z*invLen, -v1.y*invLen);
	}
	*v3 = cross(v1, *v2);
}

__host__ __device__ __inline__ optix::float3 sphericalDirection(
								 float stheta, 
								 float ctheta, 
								 float phi, 
								 optix::float3 x,
								 optix::float3 y, 
								 optix::float3 z)
{
	return stheta*cosf(phi)*x + stheta*sinf(phi)*y+ ctheta*z;
}



__device__ __inline__ float2 transmittanceUV(float r, float mu, float Rg, float Rt) 
{
    float uR, uMu;
    float angle = 0.15;//0.15;
#ifdef TRANSMITTANCE_NON_LINEAR
	uR = sqrt((r - Rg) / (Rt - Rg));
	uMu = atan((mu + angle) / (1.0 + angle) * tan(angle*10.0)) / (angle*10.0);
#else
	uR = (r - Rg) / (Rt - Rg);
	uMu = (mu + angle) / (1.0 + angle);
#endif
    return make_float2(uMu, uR);
}

__device__ __inline__ float3 transmittanceR_MU(float r, float mu, texture<float4, 2> sampler, float Rg, float Rt) 
{
	float2 uv = transmittanceUV(r, mu, Rg, Rt);
	//int2 uv_int = make_int2(uv.x*TRANSMITTANCE_W,uv.y*TRANSMITTANCE_H);
	float4 tr = tex2D(sampler, uv.x, uv.y);
    return make_float3(tr.x,tr.y,tr.z);
}


// transmittance(=transparency) of atmosphere between x and x0
// assume segment x,x0 not intersecting ground
// d = distance between x and x0, mu=cos(zenith angle of [x,x0) ray at x)
__device__ __inline__ optix::float3 transmittance_RMuD(float r, float mu, float d, texture<float4, 2> sampler, float Rg, float Rt) {
    float3 result;
    float r1 = sqrt(r * r + d * d + 2.0 * r * mu * d);
    float mu1 = (r * mu + d) / r1;
    if (mu > 0.0) {
        result = fminf(transmittanceR_MU(r, mu, sampler, Rg, Rt) / transmittanceR_MU(r1, mu1, sampler, Rg, Rt), make_float3(1.0));
    } else {
        result = fminf(transmittanceR_MU(r1, -mu1, sampler, Rg, Rt) / transmittanceR_MU(r, -mu, sampler, Rg, Rt), make_float3(1.0));
    }
    return result;
}


// transmittance(=transparency) of atmosphere between x and x0
// assume segment x,x0 not intersecting ground
// r=||x||, mu=cos(zenith angle of [x,x0) ray at x), v=unit direction vector of [x,x0) ray
__device__ __inline__ optix::float3 tranAtmNew(float r, float mu, texture<float4, 2> sampler,
																	 optix::float3 x0, optix::float3 v,
																float Rg, float Rt, float RL,
																optix::float3 betaR, optix::float3 betaMEx,
																float HR, float HM, float TRANSMITTANCE_INTEGRAL_SAMPLES) 
{
	
	optix::float3 result;
    float r1 = sqrtf(dot(x0,x0));
    float mu1 = dot(x0, v) / r1;
    
	if (mu > 0.0) {
        result = fminf(transmittanceR_MU(r, mu, sampler, Rg, Rt) / transmittanceR_MU(r1, mu1, sampler, Rg, Rt), make_float3(1.0));
    } else {
        result = fminf(transmittanceR_MU(r1, -mu1, sampler, Rg, Rt) / transmittanceR_MU(r, -mu, sampler, Rg, Rt), make_float3(1.0));
    }
    return result;
}

// transmittance(=transparency) of atmosphere for infinite ray (r,mu)
// (mu=cos(view zenith angle)), or zero if ray intersects ground
__device__ __inline__ optix::float3 transmittanceWithShadow(float r, float mu, texture<float4, 2> sampler, float Rg, float Rt) {
    return mu < -sqrtf(1.0 - (Rg / r) * (Rg / r)) ? make_float3(0.0) : transmittanceR_MU(r, mu, sampler, Rg, Rt); // transmittanceR_MU(r, mu);
}
/*
__device__ __inline__ float limit(float r, float mu) {
    float dout = -r * mu + sqrt(r * r * (mu * mu - 1.0) + RL * RL);
    float delta2 = r * r * (mu * mu - 1.0) + Rg * Rg;
    if (delta2 >= 0.0) {
        float din = -r * mu - sqrt(delta2);
        if (din >= 0.0) {
            dout = min(dout, din);
        }
    }
    return dout;
}
*/


__device__ __inline__ void ConcentricSampleDisk(float u1, float u2,
		float *dx, float *dy) {
	float r, theta;
	// Map uniform random numbers to $[-1,1]^2$
	float sx = 2 * u1 - 1;
	float sy = 2 * u2 - 1;
	// Map square to $(r,\theta)$
	// Handle degeneracy at the origin
	if (sx == 0.0 && sy == 0.0) {
		*dx = 0.0;
		*dy = 0.0;
		return;
	}
	if (sx >= -sy) {
		if (sx > sy) {
			// Handle first region of disk
			r = sx;
			if (sy > 0.0)
				theta = sy/r;
			else
				theta = 8.0f + sy/r;
		}
		else {
			// Handle second region of disk
			r = sy;
			theta = 2.0f - sx/r;
		}
	}
	else {
		if (sx <= sy) {
			// Handle third region of disk
			r = -sx;
			theta = 4.0f - sy/r;
		}
		else {
			// Handle fourth region of disk
			r = -sy;
			theta = 6.0f + sx/r;
		}
	}
	theta *= M_PI / 4.f;
	*dx = r*cosf(theta);
	*dy = r*sinf(theta);
}
__device__ __inline__ float3 CosineSampleHemisphere(float u1, float u2) {
	float3 ret;
	ConcentricSampleDisk(u1, u2, &ret.x, &ret.y);
	ret.z = sqrtf(max(0.f,
	                  1.f - ret.x*ret.x - ret.y*ret.y));
	return ret;
}



__device__ __inline__ void Sample_f(float3 wo, float3 point, float3 & wi,
		float u1, float u2, float *pdf) {
	// Cosine-sample the hemisphere, flipping the direction if necessary
	wi = CosineSampleHemisphere(u1, u2);
	//if(wi.z < 0){
	//	cout << "Sample_f(): ERROR!!!" << endl;
	//}
	//normal == point (because we are dealing with lambertian sphere)
	float3 v1 = cross(point, -wo);
	float3 v2 = cross(point, v1);
	//if(length(point + v1) < 1.0 || length(point + v2) < 1.0) {
	//	cout << "Sample_f(): ERROR!!!!! len(): " << length(point + v1) << ", " << length(point + v2)  << endl;
	//}
	//wi = vec3<float>(00.0,0.0,10.0);
	//if(wi.z * wo.z < 0.0)
	//	wi.z *= -1.0;
	float3 wi_old = normalize(wi);
	float z = dot(wi_old, point); 	
	float y= dot(wi_old, v1); 
	float x = dot(wi_old, v2); 
	//if(z * wi_old.z < 0.0)
	//	z *= -1.0;
	wi = normalize(make_float3(x,y,z));
	//if (wo.z < 0.) wi->z *= -1.f;
	//*wi = normalize(*wi);
	//*pdf = Pdf(wo, *wi);
	//return f(wo, *wi);
}