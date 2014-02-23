
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

#include <optixu/optixu_math_namespace.h>
#include "ppm.h"

struct ParallelogramLight
{
  optix::float3 corner;
  optix::float3 v1, v2;
  optix::float3 normal;
  optix::float3 emission;
};

__device__ __inline__ void mapToDisk( optix::float2& sample )
{
  float phi, r;
  float a = 2.0f * sample.x - 1.0f;      // (a,b) is now on [-1,1]^2 
  float b = 2.0f * sample.y - 1.0f;      // 
  if (a > -b) {                           // reg 1 or 2 
    if (a > b) {                          // reg 1, also |a| > |b| 
      r = a;
      phi = (M_PIf/4.0f) * (b/a);
    } else {                              // reg 2, also |b| > |a| 
      r = b;
      phi = (M_PIf/4.0f) * (2.0f - a/b);
    }
  } else {                                // reg 3 or 4 
    if (a < b) {                          // reg 3, also |a|>=|b| && a!=0 
      r = -a;
      phi = (M_PIf/4.0f) * (4.0f + b/a);
    } else {                              // region 4, |b| >= |a|,  but 
      // a==0 and  b==0 could occur. 
      r = -b;
      phi = b != 0.0f ? (M_PIf/4.0f) * (6.0f - a/b) :
        0.0f;
    }
  }
  float u = r * cosf( phi );
  float v = r * sinf( phi );
  sample.x = u;
  sample.y = v;
}


// Uniformly sample the surface of a Parallelogram.  Return probability
// of the given sample
__device__ __inline__ void sampleParallelogram( const ParallelogramLight& light, 
                                                const optix::float3& hit_point,
                                                const optix::float2& sample,
                                                optix::float3& w_in,
                                                float& dist,
                                                float& pdf )
{
  using namespace optix;

  float3 on_light = light.corner + sample.x*light.v1 + sample.y*light.v2;
  w_in = on_light - hit_point;
  float dist2 = dot( w_in, w_in );
  dist  = sqrt( dist2 ); 
  w_in /= dist;
  
  float3 normal = cross( light.v1, light.v2 );
  float area    = length( normal );
  normal /= area;
  float cosine  = -dot( normal, w_in );
  pdf = dist2 / ( area * cosine );
}


// Create ONB from normal.  Resulting W is Parallel to normal
__device__ __inline__ void createONB( const optix::float3& n,
                                      optix::float3& U,
                                      optix::float3& V,
                                      optix::float3& W )
{
  using namespace optix;

  W = normalize( n );
  U = cross( W, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( fabsf( U.x) < 0.001f && fabsf( U.y ) < 0.001f && fabsf( U.z ) < 0.001f  )
    U = cross( W, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( W, U );
}

// Create ONB from normalalized vector
__device__ __inline__ void createONB( const optix::float3& n,
                                      optix::float3& U,
                                      optix::float3& V)
{
  using namespace optix;

  U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( dot(U, U) < 1.e-3f )
    U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( n, U );
}

// sample hemisphere with cosine density
__device__ __inline__ void sampleUnitHemisphere( const optix::float2& sample,
                                                 const optix::float3& U,
                                                 const optix::float3& V,
                                                 const optix::float3& W,
                                                 optix::float3& point )
{
    using namespace optix;

    float phi = 2.0f * M_PIf*sample.x;
    float r = sqrt( sample.y );
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = 1.0f - x*x -y*y;
    z = z > 0.0f ? sqrt(z) : 0.0f;

    point = x*U + y*V + z*W;
}

// needed by volume photon mapper
__device__ __inline__ optix::float3 get_sigma_a()
{
	using namespace optix;
	return make_float3( 0.0001, 0.0001, 0.0001);
}

__device__ __inline__ optix::float3 get_sigma_s()
{
	using namespace optix;
	return make_float3( 0.058, 0.0135, 0.0331);
}

//emission
__device__ __inline__ optix::float3 Lve()
{
	using namespace optix;
	return make_float3(0.0f);//make_float3( 0.05, 0.0135, 0.331);
	//return make_float3(0.05f,0.05f,0.05);//make_float3( 0.05, 0.0135, 0.331);
}

__device__ __inline__ optix::float3 get_sigma_t(const optix::float3& s_a, const optix::float3& s_s)
{
	using namespace optix;
	return make_float3(s_a.x + s_s.x, s_a.y + s_s.y, s_a.z + s_s.z);
}

__device__ __inline__ optix::float3 PhaseRayleigh(const optix::float3 &w, const optix::float3 &wp) {
	using namespace optix;
	float costheta = dot(w, wp);
	float t = 3.f/(16.f*M_PIf) * (1 + costheta * costheta);
	return  make_float3(t,t,t);
}


__device__ __inline__ optix::float3 Tau(const optix::Ray &ray, const float t0, const float t1, const optix::float3 &sigma_a, const optix::float3 &sigma_s )
{
	using namespace optix;
	float3 p0 = ray.origin + t0*ray.direction;
	float3 p1 = ray.origin + t1*ray.direction;
	float3 pdiff = p0 - p1;
	float length = sqrtf(pdiff.x*pdiff.x + pdiff.y*pdiff.y + pdiff.z*pdiff.z);
	return length * (sigma_s + sigma_a);
}

__device__ __inline__  int Ceil2Int(double val) {

	return (int)ceil(val);
}

__device__ __inline__ optix::float3 Exp(const optix::float3 &s) 
{
	using namespace optix;
	return make_float3(expf(s.x), expf(s.y), expf(s.z));
}

__device__ __inline__ int isPointInsideBox(const optix::float3 &p, const optix::float3 &min, const optix::float3 &max) {
	using namespace optix;
	if(p.x >= min.x && p.y >= min.y && p.z >= min.z &&
		p.x <= max.x && p.y <= max.y && p.z <= max.z)
		return 1;
	return 0;
}

__device__ __inline__ optix::float3 UniformSampleSphere(float u1, float u2) {
	using namespace optix;
	float z = 1.f - 2.f * u1;
	float r = sqrtf(max(0.f, 1.f - z*z));
	float phi = 2.f * M_PI * u2;
	float x = r * cosf(phi);
	float y = r * sinf(phi);
	return make_float3(x, y, z);
}
__device__ __inline__ float UniformSpherePdf() {
	return 1.f / (4.f * M_PI);
}

__device__ __inline__ bool Black(const optix::float3 & ref){
	if(ref.x != 0.0 || ref.y != 0.0 || ref.z != 0.0)
		return false;
	return true;
}

//---------------------------------------------------------------
// Emission integrator

	//struct EmissionIntegrator
	//{ // : public VolumeIntegrator {
	// EmissionIntegrator Public Methods
	//__host__ __device__ CreateEmissionIntegrator(float ss) { stepSize = ss; }


	// EmissionIntegrator Private Data
	//float stepSize;
	//int tauSampleOffset, scatterSampleOffset;

	// EmissionIntegrator Method Definitions
	//__host__ __device__void EmissionIntegrator::RequestSamples(Sample *sample,
	//		const Scene *scene) {
	//	tauSampleOffset = sample->Add1D(1);
	//	scatterSampleOffset = sample->Add1D(1);
	//}
	
		//}
/////////////////////////////////////////////

// takes the sample and the pixel loc.
__device__ float evaluate_filter( float2 sample, uint2 pi, int filter_type, float filter_width, float gaussian_alpha ) {
  // need the .5 because we want to consider, e.g., 
  //the (0,0) pixel to be (.5, .5) in continuous sample space.
  
  float dx = fabs(sample.x - (pi.x + .5)); 
  float dy = fabs(sample.y - (pi.y + .5));
  
  if (dx > filter_width || dy > filter_width ) return 0;
  
  float gaussian_exp;
    
  switch( filter_type ) {
    case FILTER_BOX:
      return 1;
    case FILTER_TRIANGLE:
      return max(0.f,filter_width - dx) * max(0.f,filter_width - dy);
    case FILTER_GAUSSIAN:
      gaussian_exp = expf(-gaussian_alpha * filter_width * filter_width);
      return max(0.f, expf(-gaussian_alpha*dx*dx) - gaussian_exp) * max(0.f, expf(-gaussian_alpha*dy*dy) - gaussian_exp);
    //case FILTER_MITCHELL:
    //  return mitchell1D(dx/filter_width) * mitchell1D(dy/filter_width);
    //case FILTER_SINC:
    //  return sinc1D(dx/filter_width) * sinc1D(dy/filter_width);
	default:
	  gaussian_exp = expf(-gaussian_alpha * filter_width * filter_width);
      return max(0.f, expf(-gaussian_alpha*dx*dx) - gaussian_exp) * max(0.f, expf(-gaussian_alpha*dy*dy) - gaussian_exp);

  }
  
  return 1; // should not happen
}