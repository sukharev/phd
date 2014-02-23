
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
#include <optix_math.h>
#include "ppm.h"
#include "helpers.h"
#include "path_tracer.h"
#include "random.h"

using namespace optix;

//
// Ray generation program
//

rtDeclareVariable(rtObject,      top_object, , );
rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              debug_buffer;
rtBuffer<PackedPhotonRecord, 1>  photon_map;
rtBuffer<PackedPhotonRecord, 1>  vol_photon_map;
rtBuffer<PackedHitRecord, 2>     rtpass_output_buffer;
rtBuffer<uint2, 2>               image_rnd_seeds;
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(float,         alpha, , );
rtDeclareVariable(float,         total_emitted, , );
rtDeclareVariable(float,         frame_number , , );
rtDeclareVariable(float3,        ambient_light , , );
rtDeclareVariable(uint,          use_debug_buffer, , );
rtDeclareVariable(uint,          gather_ray_type, , );
rtDeclareVariable(PPMLight,      light , , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(ShadowPRD, shadow_prd, rtPayload, );
rtDeclareVariable(int, full_gather, ,);
rtDeclareVariable(int, hdrOn, ,);
rtDeclareVariable(float,         exposure, ,);



__device__ __inline__  float3 HDR(float3 L) {
	if(hdrOn == 0)
		return L;
    L = L * exposure;
    L.x = L.x < 1.413 ? pow(L.x * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.x);
    L.y = L.y < 1.413 ? pow(L.y * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.y);
    L.z = L.z < 1.413 ? pow(L.z * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.z);
    return L;
}

__device__ __inline__ 
void accumulatePhoton( const PackedPhotonRecord& photon,
                       const float3& rec_normal,
                       const float3& rec_atten_Kd,
                       uint& num_new_photons, float3& flux_M )
{
  float3 photon_energy = make_float3( photon.c.y, photon.c.z, photon.c.w );
  float3 photon_normal = make_float3( photon.a.w, photon.b.x, photon.b.y );
  float p_dot_hit = dot(photon_normal, rec_normal);
  if (p_dot_hit > 0.01f) { // Fudge factor for imperfect cornell box geom
    float3 photon_ray_dir = make_float3( photon.b.z, photon.b.w, photon.c.x );
    float3 flux = photon_energy * rec_atten_Kd; // * -dot(photon_ray_dir, rec_normal);
    num_new_photons++;
    flux_M += flux;
  }
}

__device__ __inline__ 
void accumulateVolumePhoton( const PackedPhotonRecord& photon,
                       const float3& rec_normal,
                       const float3& rec_atten_Kd,
                       uint& num_new_photons, float3& flux_VM )
{
	float3 photon_energy = make_float3( photon.c.y, photon.c.z, photon.c.w );
	float3 photon_ray_dir = make_float3( photon.b.z, photon.b.w, photon.c.x );

	//check out what the phase function thinks of your new direction
	float3 ref = PhaseRayleigh(photon_ray_dir,rec_normal); //scene->volumeRegion->p(interactPt,rn.d,direction);
	float3 flux = photon_energy * rec_atten_Kd *ref;
	num_new_photons++;
	flux_VM += flux;
}

#if 0
#define check( condition, color ) \
{ \
  if( !(condition) ) { \
    debug_buffer[index] = make_float4( stack_current, node, photon_map_size, 0 ); \
    output_buffer[index] = make_color( color ); \
    
    return; \
  } \
}
#else
#define check( condition, color )
#endif

rtDeclareVariable(float3,  sigma_s, , );
rtDeclareVariable(float3,  sigma_a, , );


#define MAX_DEPTH 20 // one MILLION photons
RT_PROGRAM void gather()
{

  
  
  
  clock_t start = clock();
  PackedHitRecord rec = rtpass_output_buffer[launch_index];
  float3 rec_position = make_float3( rec.a.x, rec.a.y, rec.a.z );
  float3 rec_normal   = make_float3( rec.a.w, rec.b.x, rec.b.y );
  float3 rec_atten_Kd = make_float3( rec.b.z, rec.b.w, rec.c.x );
  uint   rec_flags    = __float_as_int( rec.c.y );
  float  rec_radius2  = rec.c.z;
  
  //rtPrintf("gather!!!\n");
  //output_buffer[launch_index] = make_float4(rec_atten_Kd,1.0);//make_float4(1.0,0.0,0.0,1.0);
  //return;
  
  //if(rec_radius2 > 0.0f && rec_radius2 < 200.0)
  //	rtPrintf("------------------------------>old radius %f\n",rec_radius2);
  float  rec_photon_count = rec.c.w;
  float3 rec_flux     = make_float3( rec.d.x, rec.d.y, rec.d.z );
  float  rec_accum_atten = rec.d.w;
  size_t2 ob_size     = output_buffer.size();
  uint    pm_index = (launch_index.y * ob_size.x + launch_index.x); 
	

  // Check if this is hit point lies on an emitter or hit background 
  if( (!(rec_flags & PPM_HIT) && !(rec_flags & PPM_HIT_VOLUME)) || rec_flags & PPM_OVERFLOW ) {
	//output_buffer[launch_index] = make_float4(0.0,0.0,1.0,0.5);
    output_buffer[launch_index] = make_float4(rec_atten_Kd);
    return;
  }

  if(!full_gather){
	rec_atten_Kd = HDR(rec_atten_Kd);
	output_buffer[launch_index] = make_float4(rec_atten_Kd,1.0);
	return;
  }
  
  unsigned int stack[MAX_DEPTH];
  unsigned int stack_current = 0;
  unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

  push_node( 0 );

  int photon_map_size = photon_map.size(); // for debugging

  uint num_new_photons = 0u;
  float3 flux_M = make_float3( 0.0f, 0.0f, 0.0f );
  uint loop_iter = 0;
  //if(rec_flags & PPM_HIT){
  do {

    check( node < photon_map_size, make_float3( 1,0,0 ) );
    PackedPhotonRecord& photon = photon_map[ node ];

    uint axis = __float_as_int( photon.d.x );
    if( !( axis & PPM_NULL ) ) {

      float3 photon_position = make_float3( photon.a );
      float3 diff = rec_position - photon_position;
      float distance2 = dot(diff, diff);

      if (distance2 <= rec_radius2) {
        accumulatePhoton(photon, rec_normal, rec_atten_Kd, num_new_photons, flux_M);
        //if(launch_index.y ==  100 && launch_index.x == 260) {
			//rtPrintf("launch_index.x = %d\n", launch_index.x);
			//rtPrintf("accumulatePhoton surface! atten_kd=(%f,%f,%f), flux_M=(%f,%f,%f)\n", rec_atten_Kd.x, rec_atten_Kd.y, rec_atten_Kd.z, flux_M.x, flux_M.y, flux_M.z);
        //}
      }

      // Recurse
      if( !( axis & PPM_LEAF ) ) {
        float d;
        if      ( axis & PPM_X ) d = diff.x;
        else if ( axis & PPM_Y ) d = diff.y;
        else                      d = diff.z;

        // Calculate the next child selector. 0 is left, 1 is right.
        int selector = d < 0.0f ? 0 : 1;
        if( d*d < rec_radius2 ) {
          check( stack_current+1 < MAX_DEPTH, make_float3( 0,1,0) );
          push_node( (node<<1) + 2 - selector );
        }

        check( stack_current+1 < MAX_DEPTH, make_float3( 0,1,1) );
        node = (node<<1) + 1 + selector;
      } else {
        node = pop_node();
      }
    } else {
      node = pop_node();
    }
    loop_iter++;
  } while ( node );
  
  
  // Compute new N,R
  float R2 = rec_radius2;
  float N = rec_photon_count;
  float M = static_cast<float>( num_new_photons ) ;
  //float VM = static_cast<float>( num_new_vol_photons ) ; //volume photons
  float new_N = N + alpha*M;
  rec.c.w = new_N;  // set rec.photon_count

  float reduction_factor2 = 1.0f;
  float new_R2 = R2;
  if( M != 0 ) {
    reduction_factor2 = ( N + alpha*M ) / ( N + M );
    //reduction_factor2 = ( N + alpha*M) / ( N + M);
    new_R2 = R2*( reduction_factor2 ); 
    rec.c.z = new_R2; // set rec.radius2
    //rtPrintf("new radius %f\n",new_R2);
  }
 

  // Compute indirectflux
  float3 new_flux = make_float3(0.0);//JS:( rec_flux + flux_M ) * reduction_factor2;
  rec.d = make_float4( new_flux ); // set rec.flux
  
  
  float local_total_emitted = total_emitted;
  if(local_total_emitted <= 0.0f){
	local_total_emitted = 1.0f;
  }
  float3 indirect_flux = 1.0f / ( M_PIf * new_R2 ) * new_flux / local_total_emitted;

  // Compute direct
  float3 point_on_light;
  float dist_scale;
  
  if( light.is_area_light == 1) { //area light
    uint2  seed   = image_rnd_seeds[launch_index];
    float2 sample = make_float2( rnd( seed.x ), rnd( seed.y ) ); 
    image_rnd_seeds[launch_index] = seed;
    point_on_light = light.anchor + sample.x*light.v1 + sample.y*light.v2; 
    dist_scale = 1.0f;
  } else if( light.is_area_light == 0) { //point light
    point_on_light = light.position;
    dist_scale = light.radius / ( M_PIf * 0.5f); 
  } else if( light.is_area_light == 2) { //distant light
  
	uint2  seed   = image_rnd_seeds[launch_index];
    float2 sample = make_float2( rnd( seed.x ), rnd( seed.y ) ); 
    image_rnd_seeds[launch_index] = seed;
    point_on_light = light.anchor + sample.x*light.v1 + sample.y*light.v2; 
    dist_scale = light.worldRadius / ( M_PIf * 0.5f);//1.0f;
    
    //point_on_light = light.position;
    //dist_scale = light.worldRadius / ( M_PIf * 0.5f);
  }
  
  float3 to_light    = point_on_light - rec_position;
  float  light_dist  = length( to_light );
  to_light = to_light / light_dist;
  float  n_dot_l     = fmaxf( 0.0f, dot( rec_normal, to_light ) );
  float  light_atten = n_dot_l;
  
  // TODO Should clip direct light to photon emiting code -- but we will ignore this for demo 
  //if( !light.is_area_light && acosf( dot( -to_light, light.direction )  ) > light.radius ) {
  //  light_atten = 0.0f;
  //}

  // PPM_IN_SHADOW will be set if this is a point light and we have already performed an occluded shadow query 
  if( rec_flags & PPM_IN_SHADOW ) {
    light_atten = 0.0f;
  }
  if ( light_atten > 0.0f && !(rec_flags & PPM_HIT_VOLUME) ) {
    ShadowPRD prd;
    prd.attenuation = 1.0f;
    optix::Ray shadow_ray( rec_position, to_light, 2, scene_epsilon, light_dist - scene_epsilon );
    rtTrace( top_object, shadow_ray, prd );
    light_atten *= prd.attenuation * dot( -to_light, light.direction );
    rec.c.y = __int_as_float(  prd.attenuation == 0.0f && !light.is_area_light ? rec_flags|PPM_IN_SHADOW : rec_flags ); 
  } 
   
  light_atten /= dist_scale*light_dist*light_dist;
  if( light_atten < 0.0f ) light_atten = 0.0f;   // TODO Shouldnt be needed but we get acne near light w/out it
  rec.d.w = rec_accum_atten + light_atten;
  float avg_atten = rec.d.w / (frame_number+1.0f);
  float3 direct_flux = light.power * avg_atten *rec_atten_Kd;
  
  rtpass_output_buffer[launch_index] = rec;
  float3 final_color = make_float3(0.0,0.0,0.0);
  //if(bVM){
  //	final_color = make_float3(0.0,1.0,0.0);
  //}
  //else {
  //if( pm_index >= 262980 && pm_index <= 262986) {
  //	rtPrintf("indirect_flux (thread %d) = (%f,%f,%f)\n", launch_index.x, indirect_flux.x, indirect_flux.y, indirect_flux.z);
  //}
  final_color = direct_flux +indirect_flux + ambient_light*rec_atten_Kd;
  //}
  output_buffer[launch_index] = make_float4(HDR(final_color),1.0);
  if(use_debug_buffer == 1)
    debug_buffer[launch_index] = make_float4( loop_iter, new_R2, new_N, M );
}

RT_PROGRAM void gather_any_hit()
{
  //rtIgnoreIntersection();
  shadow_prd.attenuation = 0.0f;

  rtTerminateRay();
}


//
// Stack overflow program
//
rtDeclareVariable(float3, rtpass_bad_color, , );
RT_PROGRAM void gather_exception()
{
  output_buffer[launch_index] = make_float4(1.0f, 1.0f, 0.0f, 0.0f);
}


// Volume photon mapping
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );
rtDeclareVariable(float3, t_max, attribute t_max,);
rtDeclareVariable(float3, t_min, attribute t_min,); 
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(float3, shadow_attenuation, , );
rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
RT_PROGRAM void gather_vol_any_hit()
{

  shadow_prd.attenuation = 0.0f;
  rtTerminateRay();
}
