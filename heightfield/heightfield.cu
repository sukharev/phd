
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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtDeclareVariable(float4,  sphere, , );
rtDeclareVariable(float3,  boxmin, , );
rtDeclareVariable(float3,  boxmax, , );
rtDeclareVariable(float3,  cellsize, , );
rtDeclareVariable(float3,  inv_cellsize, , );
rtDeclareVariable(int2,    ncells, , );

rtBuffer<float, 2>  data;
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 



static __device__ __forceinline__ void computeNormal(int Lu, int Lv, float3 hitpos, float ya, float yb, float yc, float yd)
{
  float2 C = make_float2((hitpos.x - boxmin.x) * inv_cellsize.x,
                         (hitpos.z - boxmin.z) * inv_cellsize.z);
  float2 uv = C - make_float2(Lu, Lv);
  float dudx = inv_cellsize.x;
  float dvdz = inv_cellsize.z;
  float px = dudx*(yb + uv.y*yd);
  float pz = dvdz*(yc + uv.x*yd);
  shading_normal = geometric_normal = normalize(make_float3(-px, 1.0f, -pz));

  texcoord = (hitpos - boxmin) / (boxmax - boxmin);
  texcoord.y = texcoord.z;
  texcoord.z = 0.0f;
}

RT_PROGRAM void intersect(int primIdx)
{
  // Step 1 is setup (handled in CPU code)

  // Step 2 - transform ray into grid space and compute ray-box intersection
  float3 t0   = (boxmin - ray.origin)/ray.direction;
  float3 t1   = (boxmax - ray.origin)/ray.direction;
  float3 near = fminf(t0, t1);
  float3 far  = fmaxf(t0, t1);
  float tnear = fmaxf( near );
  float tfar  = fminf( far );

  if(tnear >= tfar)
    return;
  if(tfar < 1.e-6f)
    return;
  tnear = max(tnear, 0.f);
  tfar  = min(tfar,  ray.tmax);

  // Step 3
  size_t2 nnodes = data.size();
  float3 L = (ray.origin + tnear * ray.direction - boxmin) * inv_cellsize;
  int Lu = min(__float2int_rz(L.x), (unsigned int) (nnodes.x-2));
  int Lv = min(__float2int_rz(L.z), (unsigned int) (nnodes.y-2));

  // Step 4
  float3 D = ray.direction * inv_cellsize;
  int diu = D.x>0?1:-1;
  int div = D.z>0?1:-1;
  int stopu = D.x>0?(int)(nnodes.x)-1:-1;
  int stopv = D.z>0?(int)(nnodes.y)-1:-1;

  // Step 5
  float dtdu = abs(cellsize.x/ray.direction.x);
  float dtdv = abs(cellsize.z/ray.direction.z);

  // Step 6
  float far_u = (D.x>0.0f?Lu+1:Lu) * cellsize.x + boxmin.x;
  float far_v = (D.z>0.0f?Lv+1:Lv) * cellsize.z + boxmin.z;

  // Step 7
  float tnext_u = (far_u - ray.origin.x)/ray.direction.x;
  float tnext_v = (far_v - ray.origin.z)/ray.direction.z;

  // Step 8
  float yenter = ray.origin.y + tnear * ray.direction.y;
  while(tnear < tfar){
    float texit = min(tnext_u, tnext_v);
    float yexit = ray.origin.y + texit * ray.direction.y;

    // Step 9
    float d00 = data[make_uint2(Lu,   Lv)  ];
    float d01 = data[make_uint2(Lu,   Lv+1)];
    float d10 = data[make_uint2(Lu+1, Lv)  ];
    float d11 = data[make_uint2(Lu+1, Lv+1)];
    float datamin = min(min(d00, d01), min(d10, d11));
    float datamax = max(max(d00, d01), max(d10, d11));
    float ymin = min(yenter, yexit);
    float ymax = max(yenter, yexit);
    if(ymin <= datamax && ymax >= datamin){
      // Step 10
      float3 EC = (ray.origin + tnear * ray.direction - boxmin) * inv_cellsize - make_float3(Lu, 0.0f, Lv);
      EC.y = ray.origin.y + tnear * ray.direction.y;
      
      float ya = d00;
      float yb = d10-d00;
      float yc = d01-d00;
      float yd = d11-d10-d01+d00;
      float a = D.x*D.z*yd;
      float b = -D.y + D.x*yb + D.z*yc + (EC.x*D.z + EC.z*D.x)*yd;
      float c = ya - EC.y + EC.x*yb + EC.z*yc + EC.x*EC.z*yd;
      if(abs(a) < 1.e-6f){
        // Linear
        float tcell = -fdividef(c, b);
        float t = tnear + tcell;
        if(tcell > 0.0f && t < texit){
          if(rtPotentialIntersection( t )){
            computeNormal(Lu, Lv, ray.origin+t*ray.direction, ya, yb, yc, yd);
            if(rtReportIntersection(0))
              return;
          }
        }
      } else {
        // Solve quadatric
        b = -0.5f * b;
        float disc = b*b-a*c;
        if(disc > 0.0f){
          float root = sqrtf(disc);
          float tcell1 = fdividef(b + root, a);
          float t1 = tnear + tcell1;
          bool done = false;
          if(tcell1 >= 0.0f && t1 <= texit){
            if( rtPotentialIntersection( t1 ) ){
              computeNormal(Lu, Lv, ray.origin+t1*ray.direction, ya, yb, yc, yd);
              if(rtReportIntersection(0))
                done = true;
            }
          }
          float tcell2 = fdividef(b - root, a);
          float t2 = tnear + tcell2;
          if( tcell2 >= 0.0f && t2 <= texit){
            if( rtPotentialIntersection( t2 ) ) {
              computeNormal(Lu, Lv, ray.origin+t2*ray.direction, ya, yb, yc, yd);
              if(rtReportIntersection(0))
                done = true;
            }
          }
          if(done)
            return;
        }
      }
    }

    // Step 11
    yenter = yexit;
    if(tnext_u < tnext_v){
      Lu += diu;
      if(Lu == stopu)
        break;
      tnear = tnext_u;
      tnext_u += dtdu;
    } else {
      Lv += div;
      if(Lv == stopv)
        break;
      tnear = tnext_v;
      tnext_v += dtdv;
    }
  }
}


RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->set(boxmin, boxmax);
}

RT_PROGRAM void intersect_test(int primIdx)
{
  // Step 1 is setup (handled in CPU code)

  // Step 2 - transform ray into grid space and compute ray-box intersection
  float3 t0   = (boxmin - ray.origin)/ray.direction;
  float3 t1   = (boxmax - ray.origin)/ray.direction;
  float3 near = fminf(t0, t1);
  float3 far  = fmaxf(t0, t1);
  float tnear = fmaxf( near );
  float tfar  = fminf( far );

  if(tnear >= tfar)
    return;
  if(tfar < 1.e-6f)
    return;
  tnear = max(tnear, 0.f);
  tfar  = min(tfar,  ray.tmax);

  // Step 3
  size_t2 nnodes = data.size();
  float3 L = (ray.origin + tnear * ray.direction - boxmin) * inv_cellsize;
  int Lu = min(__float2int_rz(L.x), (unsigned int) (nnodes.x-2));
  int Lv = min(__float2int_rz(L.z), (unsigned int) (nnodes.y-2));

  // Step 4
  float3 D = ray.direction * inv_cellsize;
  int diu = D.x>0?1:-1;
  int div = D.z>0?1:-1;
  int stopu = D.x>0?(int)(nnodes.x)-1:-1;
  int stopv = D.z>0?(int)(nnodes.y)-1:-1;

  // Step 5
  float dtdu = abs(cellsize.x/ray.direction.x);
  float dtdv = abs(cellsize.z/ray.direction.z);

  // Step 6
  float far_u = (D.x>0.0f?Lu+1:Lu) * cellsize.x + boxmin.x;
  float far_v = (D.z>0.0f?Lv+1:Lv) * cellsize.z + boxmin.z;

  // Step 7
  float tnext_u = (far_u - ray.origin.x)/ray.direction.x;
  float tnext_v = (far_v - ray.origin.z)/ray.direction.z;

  // Step 8
  float yenter = ray.origin.y + tnear * ray.direction.y;
  while(tnear < tfar){
    float texit = min(tnext_u, tnext_v);
    float yexit = ray.origin.y + texit * ray.direction.y;

    // Step 9
    float d00 = data[make_uint2(Lu,   Lv)  ];
    float d01 = data[make_uint2(Lu,   Lv+1)];
    float d10 = data[make_uint2(Lu+1, Lv)  ];
    float d11 = data[make_uint2(Lu+1, Lv+1)];
    float datamin = min(min(d00, d01), min(d10, d11));
    float datamax = max(max(d00, d01), max(d10, d11));
    float ymin = min(yenter, yexit);
    float ymax = max(yenter, yexit);
    if(ymin <= datamax && ymax >= datamin){
      // Step 10
      float3 EC = (ray.origin + tnear * ray.direction - boxmin) * inv_cellsize - make_float3(Lu, 0.0f, Lv);
      EC.y = ray.origin.y + tnear * ray.direction.y;
      
      float ya = d00;
      float yb = d10-d00;
      float yc = d01-d00;
      float yd = d11-d10-d01+d00;
      float a = D.x*D.z*yd;
      float b = -D.y + D.x*yb + D.z*yc + (EC.x*D.z + EC.z*D.x)*yd;
      float c = ya - EC.y + EC.x*yb + EC.z*yc + EC.x*EC.z*yd;
      if(abs(a) < 1.e-6f){
        // Linear
        float tcell = -fdividef(c, b);
        float t = tnear + tcell;
        if(tcell > 0.0f && t < texit){
          if(rtPotentialIntersection( t )){
            computeNormal(Lu, Lv, ray.origin+t*ray.direction, ya, yb, yc, yd);
            if(rtReportIntersection(0))
              return;
          }
        }
      } else {
        // Solve quadatric
        b = -0.5f * b;
        float disc = b*b-a*c;
        if(disc > 0.0f){
          float root = sqrtf(disc);
          float tcell1 = fdividef(b + root, a);
          float t1 = tnear + tcell1;
          bool done = false;
          if(tcell1 >= 0.0f && t1 <= texit){
            if( rtPotentialIntersection( t1 ) ){
              computeNormal(Lu, Lv, ray.origin+t1*ray.direction, ya, yb, yc, yd);
              if(rtReportIntersection(0))
                done = true;
            }
          }
          float tcell2 = fdividef(b - root, a);
          float t2 = tnear + tcell2;
          if( tcell2 >= 0.0f && t2 <= texit){
            if( rtPotentialIntersection( t2 ) ) {
              computeNormal(Lu, Lv, ray.origin+t2*ray.direction, ya, yb, yc, yd);
              if(rtReportIntersection(0))
                done = true;
            }
          }
          if(done)
            return;
        }
      }
    }

    // Step 11
    yenter = yexit;
    if(tnext_u < tnext_v){
      Lu += diu;
      if(Lu == stopu)
        break;
      tnear = tnext_u;
      tnext_u += dtdu;
    } else {
      Lv += div;
      if(Lv == stopv)
        break;
      tnear = tnext_v;
      tnext_v += dtdv;
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////

// input - posBlock: position within the local coordinate system of the block 
// output - return value: position after transformation to uvw space 
/*
static __device__ float3 getUVW(float3 posBlock) 
{ 
	float3 posUVW; 
	posUVW.x = posBlock.x * g_sizeBlock + g_uMin; 
	posUVW.y = 0.0f; 
	posUVW.z = posBlock.z * g_sizeBlock + g_vMin; 
	return posUVW; 
}
*/

RT_PROGRAM void sphere_intersect(int primIdx)
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

RT_PROGRAM void sphere_bounds (int, optix::Aabb* aabb)
{
  float3 cen = make_float3( sphere );
  float3 rad = make_float3( sphere.w );
  aabb->m_min = cen - rad;
  aabb->m_max = cen + rad;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////

rtDeclareVariable(float4, plane, , );
rtDeclareVariable(float3, v1, , );
rtDeclareVariable(float3, v2, , );
rtDeclareVariable(float3, anchor, , );
rtDeclareVariable(int, lgt_instance, , ) = {0};

rtDeclareVariable(int, lgt_idx, attribute lgt_idx, ); 

rtTextureSampler<float,  2> noise_texture;

// 3D solid noise texture, values in range [0, 1].
static __device__ __inline__ float Noise1f(float3 p)
{
  return tex2D(noise_texture, p.y, p.z);
}


/*
// input - posUnitSphere: position on the unit sphere 
// output: elevation value generated by fractal sum 
// of high and low frequency Perlin noise 
static __device__ float getElevation(in float3 posUnitSphere) 
{ 
	float noiseHeight = 0; 
	float g_lowFrequencyScale = 1.0;
	float g_highFrequencyHeight = 2.0;
	// low frequency noise 
	float n = Perlin(posUnitSphere * g_lowFrequencyScale); 
	n = shift(n); 
	noiseHeight += n * g_lowFrequencyHeight; 
	// high frequency noise 
	n = Perlin(posUnitSphere * g_highFrequencyScale); 
	n = shift(n); noiseHeight += n * g_highFrequencyHeight; 
	return noiseHeight; 
}
*/

RT_PROGRAM void parallelogram_intersect(int primIdx)
{
  float3 n = make_float3( plane );
  float dt = dot(ray.direction, n );
  float t = (plane.w - dot(n, ray.origin))/dt;
  if( t > ray.tmin && t < ray.tmax ) {
    float3 p = ray.origin + ray.direction * t;
    float3 vi = p - anchor;
    float a1 = dot(v1, vi);
    if(a1 >= 0 && a1 <= 1){
      float a2 = dot(v2, vi);
      if(a2 >= 0 && a2 <= 1){
        if( rtPotentialIntersection( t ) ) {
          shading_normal = geometric_normal = n;
          texcoord = make_float3(a1,a2,0);
          lgt_idx = lgt_instance;
          rtReportIntersection( 0 );
        }
      }
    }
  }
}

RT_PROGRAM void parallelogram_bounds (int, float result[6])
{
  // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
  const float3 tv1  = v1 / dot( v1, v1 );
  const float3 tv2  = v2 / dot( v2, v2 );
  const float3 p00  = anchor;
  const float3 p01  = anchor + tv1;
  const float3 p10  = anchor + tv2;
  const float3 p11  = anchor + tv1 + tv2;
  const float  area = length(cross(tv1, tv2));
  
  optix::Aabb* aabb = (optix::Aabb*)result;
  
  if(area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ) );
    aabb->m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ) );
  } else {
    aabb->invalidate();
  }
}
