
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
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include "gathering.h"
#include "ppm.h"
#include "path_tracer.h"
#include "random.h"
#include "helpers.h"

using namespace optix;

//
// Scene wide variables
//
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(rtObject,      ppm_top_object, , );
rtDeclareVariable(int,	geometry_choice, ,);

rtTextureSampler<float4, 2> SRPhaseFuncSampler;
rtTextureSampler<float4, 2> SMPhaseFuncSampler;
rtTextureSampler<float4, 2> SRPhaseFuncIndSampler;
rtTextureSampler<float4, 2> SMPhaseFuncIndSampler;

rtTextureSampler<float4, 2> trans_texture;

rtDeclareVariable(float, Rt, ,);
rtDeclareVariable(float, Rg, ,);
rtDeclareVariable(float, RL, ,);
rtDeclareVariable(int, RES_DIR_BIN, ,);
rtDeclareVariable(int, RES_DIR_BIN_TOTAL, ,);
rtDeclareVariable(int, RES_R, ,);
rtDeclareVariable(int, RES_R_BIN, ,);
rtDeclareVariable(int, RES_MU_S, ,);
rtDeclareVariable(int, RES_MU_S_BIN, ,);

rtDeclareVariable(int, color_band, ,); //0-red, 1-green, 2-blue

rtDeclareVariable(float3, betaR, ,);
rtDeclareVariable(float3, betaMEx, ,);
rtDeclareVariable(float3, betaMSca, ,);
rtDeclareVariable(float, HR, ,);
rtDeclareVariable(float, HM, ,);
rtDeclareVariable(float, mieG, ,);
rtDeclareVariable(int, TRANSMITTANCE_INTEGRAL_SAMPLES, ,);


//
// Ray generation program
//
rtBuffer<PhotonCountRecord, 1>        ppass_vol_counts_output_buffer;
rtBuffer<PhotonRecord, 1>        ppass_output_buffer;
rtBuffer<PhotonRecord, 1>		 ppass_vol_output_buffer;
rtBuffer<PhotonRecord, 1>		 ppass_light_buffer;
rtBuffer<PhotonRecord, 1>		 ppass_lightplane_buffer;
rtBuffer<PhotonRecord, 1>		 ppass_vol_rotate_output_buffer; //duplicates with ppass_photon_table
rtBuffer<PhotonRecord, 1>		 ppass_photon_table;			 //duplicates with ppass_vol_rotate_output_buffer 
																 //2D: R (length to the point from the center of the Earth) X muS
																//	muS (cos of the angle between R and the sun direction)


rtDeclareVariable(int, INSCATTER_INTEGRAL_SAMPLES, ,);


rtBuffer<uint2, 2>               photon_rnd_seeds;
rtDeclareVariable(uint,          PHOTON_LAUNCH_WIDTH, , );
rtDeclareVariable(uint,          ppass_ray_type, , );
rtDeclareVariable(uint,          max_depth, , );
rtDeclareVariable(uint,          max_photon_count, , );
rtDeclareVariable(uint,          max_vol_photon_count, , );
rtDeclareVariable(PPMLight,      light , , );
rtDeclareVariable(float3,  sigma_s, , );
rtDeclareVariable(float3,  sigma_a, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );


__device__ __inline__ float2 rnd_from_uint2( uint2& prev )
{
  return make_float2(rnd(prev.x), rnd(prev.y));
}


		
//TODO: rewrite!!!
// utility functions used for for ray marching
__device__ __inline__ bool testForEvent(float3 x, /* in / out */ float3& x0, unsigned int seed,  uint pm_index, float3 dir, float dist, int numSteps)
{
	float3 sigmaE = sigma_a + sigma_s;
	//float3 T = trans(x, x0, sigmaE);
	float r0 = length(x0);
	float r = length(x);
	float d_length = length(x-x0);
	float mu0 = dot(dir,x0)/r0;
	float mu = dot(dir,x)/r;
	//float dx = limit(r, mu, RL, Rg) / float(INSCATTER_INTEGRAL_SAMPLES);
	
	//float3 T = transmittance_RMuD(r, mu, d_length, trans_texture, Rg, Rt);
	float3 T0 = transmittanceR_MU(r0, mu0, trans_texture, Rg, Rt); 
	float3 T = tranAtmNew(r, mu, trans_texture, x0, dir,  Rg, Rt ,RL, betaR, betaMEx,
							HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
	float u = rnd(seed);
	//if(pm_index == 289024 )
	//	rtPrintf("pm_index=%d, r=%f, r0=%f, T = (%f,%f,%f), T0 = (%f,%f,%f), dist:%f, color:%d, u:%f\n", pm_index, r,r0,T.x,T.y,T.z,T0.x,T0.y,T0.z, dist, color_band, u);
	
	
	//float avgT = (T.x+T.y+T.z)/3.0;
	
	float prob = 1.0;//1.0-T.y;//1.0- avgT; //0.8//1.0-T.y
	float sigma_t = 0.0;
	if(color_band == 0){//red
		prob = 1.0-T.x;
		sigma_t =  betaR.x;
		//if(prob > 0.3)
		//	prob = 0.3;
	}else if(color_band == 1){
		prob = 1.0-T.y;
		sigma_t =  betaR.y;
		//if(prob > 0.3)
		//	prob = 0.3;
	}else if(color_band == 2){
		prob = 1.0-T.z;
		sigma_t =  betaR.z;
	}
	//if(prob+0.5<=1.0){
	//	prob += 0.5;
	//}
	//else
	//	prob = 1.0;
	
	//prob = 0.5;
	//prob = 0.2;//1.0 - avgT;
	
	if(prob < 0.0)
		prob = 0.0;
	//if(u < prob && numSteps > 5)
	//	rtPrintf("mu=%f, prob = %f, u = %f, numSteps=%d, T(%f,%f,%f)\n", mu, prob, u, numSteps, T.x,T.y,T.z);
	
	//if(prob > 0.3 && prob < 0.5){//pm_index == 46980){
	//   rtPrintf("mu=%f, d_length=%f, prob = %f, u = %f\n", mu, float(INSCATTER_INTEGRAL_SAMPLES) * dx, prob, u);
	//}
	if(u < prob && u > 0.0 && prob > 0.0){
		//float newD = -lgammaf(1 - u)/sigma_t;
		//x0 = x + dir*newD;
		return true;
	}
	return false;
}

__device__ __inline__ bool isScattering(unsigned int seed, bool bChooseMie)
{
	float u = rnd(seed);
	if(bChooseMie){
		if(u < 0.9){
			return true;
		}
	}
	else{
		if(u < 1.0){	//TODO: change that
			return true;
		}
	}
	return false;
}


__device__ __inline__ float3 getNewScatteringDirection(const float3 new_hit_point, const float3 prev_dir, unsigned int seed1, unsigned int seed2, bool bChooseMie)
{
	//return prev_dir;
	
	//first figure out which particle was just hit (Raleigh-air molecule or Mie-aerosol)
	//bool bChooseMie = false;
	float r= length(new_hit_point);
	
	
	float stheta, ctheta;
	float u1 = rnd(seed1);
	float u2 = rnd(seed2);
	
	if(bChooseMie){
		ctheta = tex2D(SMPhaseFuncSampler, (tex2D(SMPhaseFuncIndSampler, u1, 0.0)).x, 0.0).x;
	}
	else{
		ctheta = tex2D(SRPhaseFuncSampler, (tex2D(SRPhaseFuncIndSampler, u1, 0.0)).x, 0.0).x;
	}
	stheta = sqrtf(fmaxf(0.0f, 1.0f - ctheta*ctheta));
	
	//rtPrintf("ctheta_sr: %f, ctheta_sm: %f\n",ctheta_sr, ctheta_sm);
	
	float3 v2,v3;
	coordinateSystem(prev_dir, &v2, &v3);
	float phi = 2.0f * M_PI * u2; 

	
	float3 prev_dir_n = normalize(prev_dir);
	//float3 new_dir = sphericalDirection(stheta_sm, ctheta_sm, phi, v2, v3, prev_dir_n);
	//float3 new_dir = sphericalDirection(stheta_sr, ctheta_sr, phi, v2, v3, prev_dir_n);
	float3 new_dir = sphericalDirection(stheta, ctheta, phi, v2, v3, prev_dir_n);
	new_dir = normalize(new_dir);
	//if(dot(new_dir, prev_dir) >= 0.01 || dot(new_dir, prev_dir) < -0.01){
	//	rtPrintf("dot(new_dir, prev_dir): %f\n",dot(new_dir, prev_dir));
	//}
	return new_dir;
}

__device__ __inline__ void generateLightPlane( const PPMLight& light, const float2& d_sample, float3& o, float3& d)
{
	float2 square_sample = d_sample; 
	mapToDisk( square_sample );
	
	float rad1 = dot((light.anchor-light.worldCenter)/2.0,(light.anchor-light.worldCenter)/2.0); 
	float rad2 = light.worldRadius/2.0;
	float3 pdisk = light.worldCenter + rad2 * light.plane_v1*square_sample.x + 
		rad2 * light.plane_v2*square_sample.y;
	o = pdisk;
	d = light.direction;
}

__device__ __inline__ void generateDistantLightPhoton( const PPMLight& light, const float2& d_sample, float3& o, float3& d)
{
/*
	Spectrum Sample_L(const Scene *scene,
		float u1, float u2, float u3, float u4,
		Ray *ray, float *pdf) const {
	// Choose point on disk oriented toward infinite light direction
	Point worldCenter;
	float worldRadius;
	scene->WorldBound().BoundingSphere(&worldCenter,
	                                   &worldRadius);
	Vector v1, v2;
	CoordinateSystem(lightDir, &v1, &v2);
	float d1, d2;
	ConcentricSampleDisk(u1, u2, &d1, &d2);
	Point Pdisk =
		worldCenter + worldRadius * (d1 * v1 + d2 * v2);
	// Set ray origin and direction for infinite light ray
	ray->o = Pdisk + worldRadius * lightDir;
	ray->d = -lightDir;
	*pdf = 1.f / (M_PI * worldRadius * worldRadius);
	return L;
	}
*/

  o = light.position;

  // Choose random dir by sampling disk of radius light.radius and projecting up to unit hemisphere
  float2 square_sample = d_sample; 
  mapToDisk( square_sample );
  
  //square_sample = square_sample * atanf( light.radius );
  //float x = square_sample.x;
  //float y = square_sample.y;
  //float z = sqrtf( fmaxf( 0.0f, 1.0f - x*x - y*y ) );

  // Now transform into light space
  float3 U, V, W;
  createONB(light.direction, U, V, W);
  //float3 org = light.worldCenter + (light.worldRadius) * (light.direction);
  //float3 pdisk = light.anchor + light.worldRadius * (U*square_sample.x + V*square_sample.y);  
  float3 pdisk = light.anchor + light.worldRadius * (light.v1*square_sample.x + light.v2*square_sample.y);
  
  //d =  x*U + y*V + z*W;
  /*
  float3 test_o;
  test_o.x = 0;
  test_o.y = 100.f;
  test_o.z = 0.0f;
  */
  o = pdisk;// + (light.worldRadius*1.3) * (light.direction);
  // Choose a random position on light
  
  //this would create a 2D light slice
  //o = light.anchor + ( square_sample.x* light.v1 + square_sample.y*light.v2);
  d = light.direction;

}

__device__ __inline__ void generateAreaLightPhoton( const PPMLight& light, const float2& d_sample, float3& o, float3& d)
{
  // Choose a random position on light
  o = light.anchor + 0.5f * ( light.v1 + light.v2);
  
  // Choose a random direction from light
  float3 U, V, W;
  createONB( light.direction, U, V, W);
  sampleUnitHemisphere( d_sample, U, V, W, d );
}

__device__ __inline__ void generateSpotLightPhoton( const PPMLight& light, const float2& d_sample, float3& o, float3& d)
{
  o = light.position;

/*
  // Choose random dir by sampling disk of radius light.radius and projecting up to unit hemisphere
  float r = atanf( light.radius) * sqrtf( d_sample.x );
  float theta = 2.0f * M_PIf * d_sample.y;

  float x = r*cosf( theta );
  float y = r*sinf( theta );
  float z = sqrtf( fmaxf( 0.0f, 1.0f - x*x - y*y ) );
*/

  // Choose random dir by sampling disk of radius light.radius and projecting up to unit hemisphere
  float2 square_sample = d_sample; 
  mapToDisk( square_sample );
  square_sample = square_sample * atanf( light.radius );
  float x = square_sample.x;
  float y = square_sample.y;
  float z = sqrtf( fmaxf( 0.0f, 1.0f - x*x - y*y ) );

  // Now transform into light space
  float3 U, V, W;
  createONB(light.direction, U, V, W);
  d =  x*U + y*V + z*W;
}


RT_PROGRAM void ppass_camera()
{
  size_t2 size     = photon_rnd_seeds.size();
  //rtPrintf("Hello from index index = %u, size = %u, max_photon_count = %u!\n", launch_index.x, size.x, max_photon_count);
  uint	  rndind = launch_index.y*PHOTON_LAUNCH_WIDTH + launch_index.x;
  size_t2 rnd_launch;
  rnd_launch.x = rndind;
  rnd_launch.y = 0;
  //uint	  old_rndind = launch_index.y * size.x + launch_index.x;
  uint    pm_index = (rndind) * max_photon_count;
  uint	  pm_vol_index = (rndind) * max_vol_photon_count;
  uint    index = (rndind);
  uint2   seed     = photon_rnd_seeds[rnd_launch]; // No need to reset since we dont reuse this seed
  //int j = 3;
  //for(unsigned int i = 0; i < j; ++i) {
	//j++;
  //}
  
  float2 direction_sample = make_float2(
      ( static_cast<float>( launch_index.x ) + rnd( seed.x ) ) / static_cast<float>( PHOTON_LAUNCH_WIDTH ),
      ( static_cast<float>( launch_index.y ) + rnd( seed.y ) ) / static_cast<float>( size.x/PHOTON_LAUNCH_WIDTH ) );
  float3 ray_origin, ray_direction;
  if( light.is_area_light == 1) {
    generateAreaLightPhoton( light, direction_sample, ray_origin, ray_direction );
  } else if( light.is_area_light == 0 ){
    generateSpotLightPhoton( light, direction_sample, ray_origin, ray_direction );
  } else { //if( light.is_area_light == 2 )
	generateDistantLightPhoton( light, direction_sample, ray_origin, ray_direction );
	//generateAreaLightPhoton( light, direction_sample, ray_origin, ray_direction );
  }

  
  PhotonRecord& rec = ppass_light_buffer[index];
  rec.position = ray_origin;
  rec.normal = ray_origin; //ffnormal;
  rec.ray_dir = ray_direction;
  rec.energy = make_float3(1.0,1.0,1.0);
  
  float3 pl_origin, pl_direction;
  generateLightPlane(light, direction_sample, pl_origin, pl_direction);
  PhotonRecord& rec_pl = ppass_lightplane_buffer[index];
  rec_pl.position = pl_origin;
  rec_pl.normal = pl_origin;
  rec_pl.ray_dir = pl_direction;
  rec_pl.energy = make_float3(1.0,1.0,1.0);
  
			    
  optix::Ray ray(ray_origin, ray_direction, ppass_ray_type, scene_epsilon );

  // Initialize our photons
  for(unsigned int i = 0; i < max_photon_count; ++i) {
    ppass_output_buffer[i+pm_index].energy = make_float3(0.0f);
  }
  
  // Initialize our volume photons
  for(unsigned int i = 0; i < max_vol_photon_count; ++i) {
    ppass_vol_output_buffer[i+pm_vol_index].energy = make_float3(0.0f);
  }
  
  // Initialize our volume photons
  for(unsigned int i = 0; i < max_vol_photon_count; ++i) {
    ppass_vol_rotate_output_buffer[i+pm_vol_index].energy = make_float3(0.0f);
  }
  
  
  PhotonPRD prd;
  //  rec.ray_dir = ray_direction; // set in ppass_closest_hit
  if( light.is_area_light == 2){
	prd.energy = light.power; //L * M_PI * worldRadius * worldRadius;
  } else {
	prd.energy = light.power;
  }
	
  prd.sample = seed;
  prd.pm_index = pm_index;
  prd.pm_vol_index = pm_vol_index;
  prd.num_deposits = 0;
  prd.num_voldeposits = 0;
  prd.ray_depth = 0;
  prd.prev_ray_length = 0;
  int old_num_voldeposits = prd.num_voldeposits;
  if(geometry_choice == 0)
		rtTrace( top_object, ray, prd );
  else
		rtTrace( ppm_top_object, ray, prd );
    
  if(prd.num_voldeposits > 0){
	  while((old_num_voldeposits != prd.num_voldeposits) && (prd.num_voldeposits < max_vol_photon_count-1)){
		prd.sample = seed;
		prd.pm_index = pm_index;
		prd.num_deposits = 0;
		old_num_voldeposits = prd.num_voldeposits;
		prd.ray_depth = 0;
		prd.prev_ray_length = 0;
		if(geometry_choice == 0)
			rtTrace( top_object, ray, prd );
		else
			rtTrace( ppm_top_object, ray, prd );
	  }
  }
  
}

//
// Closest hit material
//
rtDeclareVariable(float3,  Ks, , );
rtDeclareVariable(float3,  Kd, , );
rtDeclareVariable(float3,  emitted, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PhotonPRD, hit_record, rtPayload, );

rtDeclareVariable(float,  phong_exp, , );
rtDeclareVariable(float3, emissive, , );
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtTextureSampler<float4, 2> ambient_map;
rtTextureSampler<float4, 2> diffuse_map;
rtTextureSampler<float4, 2> specular_map;
//rtBuffer<TriangleLight, 1>  light_buffer;


RT_PROGRAM void ppass_closest_hit()
{
  //rtPrintf("ppass_closest_hit \n");
  //rtPrintf("Hello from index %u, %u!\n", launch_index.x, launch_index.y);
  // Check if this is a light source
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal     = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  float3 hit_point = ray.origin + t_hit*ray.direction;
  hit_record.prev_ray_length = t_hit;
  float3 new_ray_dir;
  
	//rtPrintf("some other type %u, %u!\n", launch_index.x, launch_index.y);
	  if( fmaxf( Kd ) > 0.0f ) {
		// We hit a diffuse surface; record hit if it has bounced at least once
		if( hit_record.ray_depth > 0 ) {
		  PhotonRecord& rec = ppass_output_buffer[hit_record.pm_index + hit_record.num_deposits];
		  rec.position = hit_point;
		  rec.normal = ffnormal;
		  rec.ray_dir = ray.direction;
		  rec.energy = hit_record.energy;
		  hit_record.num_deposits++;
		}

		hit_record.energy = Kd * hit_record.energy; 
		float3 U, V, W;
		createONB(ffnormal, U, V, W);
		sampleUnitHemisphere(rnd_from_uint2(hit_record.sample), U, V, W, new_ray_dir);

	  } else {
		hit_record.energy = Ks * hit_record.energy;
		// Make reflection ray
		new_ray_dir = reflect( ray.direction, ffnormal );
	  }

	  hit_record.ray_depth++;
	  if ( hit_record.num_deposits >= max_photon_count || hit_record.ray_depth >= max_depth)
		return;

	  optix::Ray new_ray( hit_point, new_ray_dir, ppass_ray_type, scene_epsilon );
	  rtTrace(top_object, new_ray, hit_record);
	  if(geometry_choice == 0)
		rtTrace( top_object, new_ray, hit_record);
	  else
		rtTrace( ppm_top_object, new_ray, hit_record);
		
	  //ignore other rays
	  hit_record.prev_ray_length = t_hit;

}


rtDeclareVariable(float, t_max, attribute t_max,);
rtDeclareVariable(float, t_min, attribute t_min,);
rtDeclareVariable(float3, bb_max, attribute bb_max,);
rtDeclareVariable(float3, bb_min, attribute bb_min,);
rtDeclareVariable(float, stepSizeP, ,);
rtDeclareVariable(float, frame_number, ,);







// Y *RES_DIR_BIN^2 + X*RES_DIR_BIN^1 + Z
// VERIFIED
__device__ __inline__ void save_countToPosBin(int muS_bound, int r_ind, optix::float3 photon_dir, uint cuda_index, int vol_photon_depth)
{
	//direction bin <--------- ppass_vol_counts_output_buffer[(muS_bound*RES_R_BIN + r_ind)*RES_DIR_BIN_TOTAL]
	float3 pdir = normalize(photon_dir);
	int x_ind = floor((int)(RES_DIR_BIN) *(float)(pdir.x +1.0)/2.0);
	int y_ind = floor((int)(RES_DIR_BIN) *(float)(pdir.y +1.0)/2.0);
	int z_ind = floor((int)(RES_DIR_BIN) *(float)(pdir.z +1.0)/2.0);
	if(x_ind == RES_DIR_BIN) x_ind = RES_DIR_BIN-1;
	if(y_ind == RES_DIR_BIN) y_ind = RES_DIR_BIN-1;
	if(z_ind == RES_DIR_BIN) z_ind = RES_DIR_BIN-1;
	int bin_id = y_ind * RES_DIR_BIN * RES_DIR_BIN + x_ind * RES_DIR_BIN + z_ind;
	
	//TODO: ? why not int index = (muS_bound*(RES_R_BIN-1) + r_ind)*RES_DIR_BIN_TOTAL + bin_id;
	int index = (muS_bound*RES_R_BIN + r_ind)*RES_DIR_BIN_TOTAL + bin_id;
	//if(pdir.x + pdir.y+pdir.z > 0.0)
	//	rtPrintf("PPass: pdir (%f,%f,%f)--> (%d,%d,%d)\n", pdir.x, pdir.y, pdir.z, x_ind, y_ind, z_ind);
	//rtPrintf("index = %d, muS_bound = %d,, r_ind = %d\n", index, muS_bound, r_ind);
	//if(index == 0)
	//rtPrintf("bin_id %d (x %d, y %d, z %d),  muS_bound %d, r_ind %d, bin_count index %d\n", bin_id, x_ind, y_ind, z_ind, muS_bound, r_ind, index);
	
	
	ppass_vol_counts_output_buffer[cuda_index].pos_dir_index = index;
	ppass_vol_counts_output_buffer[cuda_index].dir_count = vol_photon_depth;
	
	//ppass_vol_counts_output_buffer[cuda_index].cuda_index = cuda_index - vol_photon_depth;
	//ppass_vol_counts_output_buffer[cuda_index].dir_count = vol_photon_depth;
}
				
RT_PROGRAM void ppass_vol_closest_hit_counts()
{
  // 1. Perform ray marching until either event is found or reached the destination (next geometry piece or empty space
  // 2. If event found determine if it is a scattering event or absorbtion event
  // 3. if it is a scattering event go back to step 1 else terminate the random walk
  
  //rtPrintf("Hello from index %u, %u!\n", launch_index.x, launch_index.y);
  // Check if this is a light source
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal     = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  float3 hit_point = ray.origin + t_hit*ray.direction;
  float3 prev_ray_dir = ray.direction;
  float3 new_ray_dir = ray.direction;
  float3 new_hit_point = hit_point;
  float ray_depth = hit_record.ray_depth;
  
  size_t2 size     = photon_rnd_seeds.size();
  uint	  rndind = launch_index.y*PHOTON_LAUNCH_WIDTH + launch_index.x;
  size_t2 rnd_launch;
  rnd_launch.x = rndind;
  rnd_launch.y = 0;
  uint    pm_index = (rndind) * max_vol_photon_count;
  //uint    pm_index = (launch_index.y * size.x + launch_index.x) * max_vol_photon_count;
  
  if(pm_index > 47810 && pm_index < 47820) {
	//rtPrintf("fnum = %f,  depth=%d\n", fnum, hit_record.ray_depth);
  }
		
  //rtPrintf("LIGHT: pv1(%f,%f,%f), pv2(%f,%f,%f), light_dir(%f,%f,%f)\n", light.plane_v1.x, light.plane_v1.y, light.plane_v1.z,
  //light.plane_v2.x,light.plane_v2.y,light.plane_v2.z, light.direction.x,light.direction.y,light.direction.z);

  int count = 0; // number of events along a single ray
  bool bEvent = false;
  float sigma_t = 1.0;
  if(color_band == 0){//red
	sigma_t =  betaR.x;
  }else if(color_band == 1){
	sigma_t =  betaR.y;
  }else if(color_band == 2){
	sigma_t =  betaR.z;
  }
  
  int layer;
  float ray_length = 0.0;
  if(ray_depth == 0){
	while(hit_record.num_voldeposits < max_vol_photon_count-1){
		rnd_launch.y = hit_record.num_voldeposits;
		unsigned int seed1 = rot_seed( photon_rnd_seeds[ rnd_launch ].x, (frame_number+1)*count );
		unsigned int seed2 = rot_seed( photon_rnd_seeds[ rnd_launch ].y, (frame_number+1)*count );
		//hit_record.ray_depth++;
		
		sphere_planet_atm_boundary_dist(pm_index, make_float3(0.0), hit_point, new_ray_dir, layer, ray_length, true, Rg, Rt, RES_R_BIN);
		//TODO: replace this on the same test as in rtpass (and the same needs to be done with the one below).
		//optix::Ray new_ray( new_hit_point, new_ray_dir, ppass_ray_type, scene_epsilon, RT_DEFAULT_MAX );
		//if(geometry_choice == 0)
		//	rtTrace( top_object, new_ray, hit_record);
		//else
		//	rtTrace( ppm_top_object, new_ray, hit_record);
		//
		//hit_record.ray_depth--;

		float dist = stepSizeP*rnd(seed2);
		if(count == 0){
			dist = stepSizeP*rnd(seed2);//stepSizeP*3;
		}
		int numSteps = 0;
	
		new_hit_point = hit_point;
		bEvent = false;
		float3 curr_dir = new_ray_dir;
		float min_dist = 0.5;
		float u1 = 0.0;
		float u2 = 0.0;
		int countReachEarth = -1;
		while(dist < ray_length && !bEvent && numSteps < 1600){//30){
			new_hit_point = hit_point + dist*new_ray_dir;//*1.5;
			if(length(new_hit_point) > (Rt+0.1)){
				bEvent = false;
				break;
			}
			
			if(length(new_hit_point) >= (Rg - min_dist) && length(new_hit_point) <= (Rg + min_dist)){
				new_hit_point = hit_point + ray_length*new_ray_dir;
				u1 = rnd(seed1);
				u2 = rnd(seed2);
				Sample_f(normalize(curr_dir), normalize(new_hit_point), new_ray_dir, u1, u2, NULL) ;
				curr_dir = new_ray_dir;
				countReachEarth = numSteps;
				hit_point = new_hit_point;
				dist = 0.0;
				sphere_planet_atm_boundary_dist(pm_index, make_float3(0.0), new_hit_point, new_ray_dir, layer, ray_length, true, Rg, Rt, RES_R_BIN);

			}
			
			if(length(new_hit_point) > (Rt+0.1) || length(new_hit_point) < (Rg-5.0)){
				bEvent = false;
				break;
			}
			
			bEvent = testForEvent(hit_point, new_hit_point, seed1, pm_index, new_ray_dir, dist, numSteps);
			float rnd_s = rnd(seed1);
			dist = dist + /*rnd_s*/stepSizeP;
			//dist = dist + logf(rnd_s)/sigma_t;//stepSizeP;
			numSteps++;
		}
		
		//if(numSteps == 1600){//pm_index == 
		//	rtPrintf("pm_index: %d, numSteps=%d altitude: %f countReachEarth:%d\n", pm_index, numSteps, length(new_hit_point), countReachEarth);
		//}
		//if(bEvent && countReachEarth > -1){//pm_index == 
		//	rtPrintf("(Reached Earth) pm_index: %d, numSteps=%d altitude: %f countReachEarth:%d\n", pm_index, numSteps, length(new_hit_point), countReachEarth);
		//}
		
		
		if(bEvent){
			bEvent = false;
			hit_point = new_hit_point;
			
			bool bChooseMie = false;
			float mie_dist = expf(-(length(new_hit_point) - Rg) / HM);
			float ray_dist = expf(-(length(new_hit_point) - Rg) / HR);
			float mie_percent = mie_dist/(mie_dist+ray_dist);
			float ray_percent = ray_dist/(mie_dist+ray_dist);
			float rand_mie_ray = rnd(seed1);
			if(rand_mie_ray < mie_percent)
				bChooseMie = true;
		
			//determine if the event is a SCATTERING or ABSORBTION
			if(isScattering(seed2, bChooseMie)){
				//rtPrintf("ppm_ppass sun dir: (%f,%f,%f)\n", light.direction.x,light.direction.y,light.direction.z);
			    
			    /*
			    PhotonRecord& rec = ppass_vol_output_buffer[hit_record.pm_vol_index + hit_record.num_voldeposits];
			    rec.position = new_hit_point;
			    rec.normal = prev_ray_dir; //ffnormal;
			    rec.ray_dir = prev_ray_dir;
			    rec.energy = hit_record.energy;
				*/
			    
			    //PhotonRecord& rec_rt = ppass_vol_rotate_output_buffer[hit_record.pm_vol_index + hit_record.num_voldeposits];
			    float3 pos_rot = new_hit_point;
			    float3 dir_rot = prev_ray_dir;
			    rotateVectorToLightPlane(light, pos_rot, dir_rot);
			    dir_rot = normalize(dir_rot);
			    //rtPrintf("dir_rot(%f,%f,%f) \n", dir_rot.x,dir_rot.y,dir_rot.z);
			    /*
			    rec_rt.position = pos_rot;
			    rec_rt.normal = dir_rot; //ffnormal;
			    rec_rt.ray_dir = dir_rot;
			    rec_rt.energy = hit_record.energy;
			    */
		  
				new_ray_dir = getNewScatteringDirection(new_hit_point, prev_ray_dir, seed1, seed2, bChooseMie);
				prev_ray_dir = new_ray_dir;
				
				//save the photon into position bins
				int muS_bound=findPosBin(new_hit_point, light.direction, RES_MU_S_BIN);
				int r_len = findRBin(new_hit_point, Rt, Rg, RES_R_BIN);
				save_countToPosBin(muS_bound,r_len, dir_rot, hit_record.pm_vol_index + hit_record.num_voldeposits, hit_record.num_voldeposits );
				hit_record.num_voldeposits++;
			}
			else{
				break;
			}
		}
		else{
			break;
		}
		count++;
	}
	//if(pm_index  >= 46000 && pm_index <= 46980){
	//	rtPrintf("num_voldeposits=%d \n", hit_record.num_voldeposits);
	//}
  }
  else{
	hit_record.prev_ray_length = t_hit;
	return;
  }
}

/*
RT_PROGRAM void visit()
{
  unsigned int index = (unsigned int)( ray.direction.y < 0.0f );
  rtIntersectChild( index );
}
*/


