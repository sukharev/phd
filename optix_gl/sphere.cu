
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

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float4,  sphere, , );
//rtDeclareVariable(float4,  sphere_inner, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 


template<bool use_robust_method>
__device__
void intersect_sphere(void)
{
  float3 center = make_float3(sphere);
  float3 O = ray.origin - center;
  float3 D = ray.direction;
  float radius = sphere.w;

  float b = dot(O, D);
  float c = dot(O, O)-radius*radius;
  float disc = b*b-c;
  if(disc > 0.0f){
    float sdisc = sqrtf(disc);
    float root1 = (-b - sdisc);

    bool do_refine = false;

    float root11 = 0.0f;

    if(use_robust_method && fabsf(root1) > 10.f * radius) {
      do_refine = true;
    }

    if(do_refine) {
      // refine root1
      float3 O1 = O + root1 * ray.direction;
      b = dot(O1, D);
      c = dot(O1, O1) - radius*radius;
      disc = b*b - c;

      if(disc > 0.0f) {
        sdisc = sqrtf(disc);
        root11 = (-b - sdisc);
      }
    }

    bool check_second = true;
    if( rtPotentialIntersection( root1 + root11 ) ) {
      shading_normal = geometric_normal = (O + (root1 + root11)*D)/radius;
      if(rtReportIntersection(0))
        check_second = false;
    } 
    if(check_second) {
      float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
      if( rtPotentialIntersection( root2 ) ) {
        shading_normal = geometric_normal = (O + root2*D)/radius;
        rtReportIntersection(0);
      }
    }
  }
}


RT_PROGRAM void intersect(int primIdx)
{
  intersect_sphere<false>();
}


RT_PROGRAM void robust_intersect(int primIdx)
{
  intersect_sphere<true>();
}


RT_PROGRAM void bounds (int, float result[6])
{
  float3 cen = make_float3( sphere );
  float3 rad = make_float3( sphere.w );

  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = cen - rad;
  aabb->m_max = cen + rad;
}

//////////////////////////////////////////////////////////////////

template<bool use_robust_method>
__device__
void intersect_atmosphere(void)
{
    float3 center = make_float3(sphere);
  float3 O = ray.origin - center;
  float3 D = ray.direction;
  float radius = sphere.w;

  float b = dot(O, D);
  float c = dot(O, O)-radius*radius;
  float disc = b*b-c;
  if(disc > 0.0f){
    float sdisc = sqrtf(disc);
    float root1 = (-b - sdisc);

    bool do_refine = false;

    float root11 = 0.0f;

    if(use_robust_method && fabsf(root1) > 10.f * radius) {
      do_refine = true;
    }

    if(do_refine) {
      // refine root1
      float3 O1 = O + root1 * ray.direction;
      b = dot(O1, D);
      c = dot(O1, O1) - radius*radius;
      disc = b*b - c;

      if(disc > 0.0f) {
        sdisc = sqrtf(disc);
        root11 = (-b - sdisc);
      }
    }

    bool check_second = true;
    if( rtPotentialIntersection( root1 + root11 ) ) {
      shading_normal = geometric_normal = (O + (root1 + root11)*D)/radius;
      texcoord = make_float3(0.0,0.0,0.0);
      if(rtReportIntersection(0))
        check_second = false;
    } 
    if(check_second) {
      float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
      if( rtPotentialIntersection( root2 ) ) {
        shading_normal = geometric_normal = (O + root2*D)/radius;
        texcoord = make_float3(0.0,0.0,0.0);
        rtReportIntersection(0);
      }
    }
  }
}

RT_PROGRAM void intersect_atmosphere(int primIdx)
{
  intersect_atmosphere<false>();
}


RT_PROGRAM void robust_intersect_atmosphere(int primIdx)
{
  intersect_atmosphere<true>();
}

RT_PROGRAM void bounds_atmosphere(int primIdx, float result[6])
{
  float3 cen = make_float3( sphere );
  float3 rad = make_float3( sphere.w );

  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = cen - rad;
  aabb->m_max = cen + rad;
  
  //float3 rad_inner = make_float3( sphere_inner.w );
  //aabb->m_min = cen - rad_inner;
}




RT_PROGRAM void tex_intersect(int primIdx)
{
  float3 matrix_row_0 = make_float3(1.0,0.0,0.0);
  float3 matrix_row_1 = make_float3(0.0,1.0,0.0);
  float3 matrix_row_2 = make_float3(0.0,0.0,1.0);
      
  float3 center = make_float3(sphere);
  float3 O = ray.origin - center;
  float3 D = ray.direction;
  float radius = sphere.w;

  float b = dot(O, D);
  float c = dot(O, O)-radius*radius;
  float disc = b*b-c;
  if(disc > 0.0f){
    float sdisc = sqrtf(disc);
    float root1 = (-b - sdisc);
    bool check_second = true;
    if( rtPotentialIntersection( root1 ) ) {
      shading_normal = geometric_normal = (O + root1*D)/radius;

      float3 polar;

      polar.x = dot(matrix_row_0, geometric_normal);
      polar.y = dot(matrix_row_1, geometric_normal);
      polar.z = dot(matrix_row_2, geometric_normal);
      polar = optix::cart_to_pol(polar);

      texcoord = make_float3( polar.x*0.5f*M_1_PIf, (polar.y+M_PI_2f)*M_1_PIf, polar.z/radius );

      if(rtReportIntersection(0))
        check_second = false;
    } 
    if(check_second) {
      float root2 = (-b + sdisc);
      if( rtPotentialIntersection( root2 ) ) {
        shading_normal = geometric_normal = (O + root2*D)/radius;

        float3 polar;
        polar.x = dot(matrix_row_0, geometric_normal);
        polar.y = dot(matrix_row_1, geometric_normal);
        polar.z = dot(matrix_row_2, geometric_normal);
        polar = optix::cart_to_pol(polar);

        texcoord = make_float3( polar.x*0.5f*M_1_PIf, (polar.y+M_PI_2f)*M_1_PIf, polar.z/radius );

        rtReportIntersection(0);
      }
    }
  }
}

RT_PROGRAM void tex_bounds (int, optix::Aabb* aabb)
{
  float3 cen = make_float3( sphere );
  float3 rad = make_float3( sphere.w );
  aabb->m_min = cen - rad;
  aabb->m_max = cen + rad;
}

////////////////////////////////////////////////////////////////////
//
// cone intersection and bounds kernels

rtDeclareVariable(float3, cone_data, ,);
rtDeclareVariable(int, up_cone, ,);
rtDeclareVariable(float, Rg, ,);

// radius, height, phiMax	:	cone_data.x, cone_data.y, cone_data.z

__device__ __inline__ void swap(float &t0, float &t1)
{
	float temp = t0;
	t0 = t1;
	t1 = temp;
}

__device__ __inline__ bool Quadratic(float A, float B, float C, float *t0,
		float *t1) {
	// Find quadratic discriminant
	float discrim = B * B - 4.f * A * C;
	if (discrim < 0.) return false;
	float rootDiscrim = sqrtf(discrim);
	// Compute quadratic _t_ values
	float q;
	if (B < 0) q = -0.5f * (B - rootDiscrim);
	else       q = -0.5f * (B + rootDiscrim);
	*t0 = q / A;
	*t1 = C / q;
	if (*t0 > *t1) swap(*t0, *t1);
	return true;
}

RT_PROGRAM void cone_intersect(int primIdx)
{
	float phi;
	float3 phit;
	float3 matrix_row_0 = make_float3(1.0,0.0,0.0);
	float3 matrix_row_1 = make_float3(0.0,1.0,0.0);
	float3 matrix_row_2 = make_float3(0.0,0.0,1.0);
  
	// Transform _Ray_ to object space
	// Compute quadratic cone coefficients
	float radius = cone_data.x;
	float height = cone_data.y;
	float phiMax = cone_data.z;
	float k = radius / height;
	k = k*k;
	
	int coef = 1;
	float A = ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y -
		k * ray.direction.z * ray.direction.z;
	float B = 2 * (ray.direction.x * ray.origin.x + ray.direction.y * ray.origin.y -
		k * ray.direction.z * (ray.origin.z*coef) );
	float C = ray.origin.x * ray.origin.x + ray.origin.y * ray.origin.y -
		k * (ray.origin.z) * (ray.origin.z*coef);
	// Solve quadratic equation for _t_ values
	float t0, t1;
	if (Quadratic(A, B, C, &t0, &t1)){
		// Compute intersection distance along ray
		if (t0 > ray.tmax || t1 < ray.tmin)
			return;
		
		float thit = t0;
		if (t0 < ray.tmin) {
			thit = t1;
			if (thit > ray.tmax) 
				return;
		}
		// Compute cone inverse mapping
		phit = ray.origin + ray.direction * thit;
		if(phit.z*phit.z + phit.x*phit.x + phit.y*phit.y< Rg*Rg)
			return;
		phi = atan2f(phit.y, phit.x);
		if (phi < 0.) phi += 2.f*M_PI;
		// Test cone intersection against clipping parameters
		if (phit.z < 0 || phit.z > height || phi > phiMax) {
			if (thit == t1) return;
			thit = t1;
			if (t1 > ray.tmax) return;
			// Compute cone inverse mapping
			phit = ray.origin + ray.direction*thit;
			phi = atan2f(phit.y, phit.x);
			if (phi < 0.) phi += 2.f*M_PI;
			if (phit.z < 0 || phit.z > height || phi > phiMax)
				return;
		}
		// Find parametric representation of cone hit
		float u = phi / phiMax;
		float v = phit.z / height;
		// Compute cone \dpdu and \dpdv
		float3 dpdu = make_float3(-phiMax * phit.y, phiMax * phit.x, 0);
		float3 dpdv = make_float3(-phit.x / (1.f - v),
					-phit.y / (1.f - v), height);
		// Compute cone \dndu and \dndv
		float3 d2Pduu = -phiMax * phiMax *
						make_float3(phit.x, phit.y, 0.);
		float3 d2Pduv = phiMax / (1.f - v) *
						make_float3(-phit.y, -phit.x, 0.);
		float3 d2Pdvv = make_float3(0, 0, 0);
		// Compute coefficients for fundamental forms
		float E = dot(dpdu, dpdu);  //jeffs <<<
		float F = dot(dpdu, dpdv);
		float G = dot(dpdv, dpdv);
		float3 N = normalize(cross(dpdu, dpdv));
		float e = dot(N, d2Pduu);
		float f = dot(N, d2Pduv);
		float g = dot(N, d2Pdvv);
		// Compute \dndu and \dndv from fundamental form coefficients
		float invEGF2 = 1.f / (E*G - F*F);
		float3 dndu = (f*F - e*G) * invEGF2 * dpdu +
			(e*F - f*E) * invEGF2 * dpdv;
		float3 dndv = (g*F - f*G) * invEGF2 * dpdu +
			(f*F - g*E) * invEGF2 * dpdv;
		// Initialize _DifferentialGeometry_ from parametric information
		//*dg = DifferentialGeometry(ObjectToWorld(phit),
		//						   ObjectToWorld(dpdu),
		//						   ObjectToWorld(dpdv),
		//						   ObjectToWorld(dndu),
		//						   ObjectToWorld(dndv),
		//						   u, v, this);
		// Update _tHit_ for quadric intersection
		//*tHit = thit;
		if( rtPotentialIntersection( thit ) ) {
			shading_normal = geometric_normal = N;
			float3 polar;
			polar.x = dot(matrix_row_0, geometric_normal);
			polar.y = dot(matrix_row_1, geometric_normal);
			polar.z = dot(matrix_row_2, geometric_normal);
			polar = optix::cart_to_pol(polar);
			//texcoord = make_float3( polar.x*0.5f*M_1_PIf, (polar.y+M_PI_2f)*M_1_PIf, polar.z/radius );
			rtReportIntersection(0);
		} 
		return;
	}
}

RT_PROGRAM void cone_bounds (int, optix::Aabb* aabb)
{ 
  float3 p1 = make_float3( -cone_data.x, -cone_data.x, -cone_data.y );
  float3 p2 = make_float3(  cone_data.x,  cone_data.x, cone_data.y );
  aabb->m_min = make_float3(fminf(p1.x, p2.x),
					 fminf(p1.y, p2.y),
					 fminf(p1.z, p2.z));
  aabb->m_max = make_float3(fmaxf(p1.x, p2.x),
					 fmaxf(p1.y, p2.y),
					 fmaxf(p1.z, p2.z));
}