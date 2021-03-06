
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
#include "ppm.h"
#include "path_tracer.h"
#include "random.h"
#include "helpers.h"
#include "gathering.h"

using namespace optix;

__device__ __inline__ float3 photon_gather_contib_ext(int pm_index, float3 x_start, float3 x_end, float3 v_in, int index, int & num_photons, int ci, int si);
__device__ __inline__ float3 viewing_ray(uint pm_index, float r, float muS, float mu, float nu);

rtTextureSampler<float4, 2> trans_texture;
rtDeclareVariable(int, TRANSMITTANCE_INTEGRAL_SAMPLES, ,);
rtDeclareVariable(float, SCALE_FACTOR, ,);
rtDeclareVariable(int, TEST_RT_PASS, ,);
rtBuffer<uint3>  rtpass_global_photon_counts;
rtDeclareVariable(uint, pbinpass_ray_type, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(PPMLight, light, ,);

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

struct PerRayData_shadow
{
  float3 attenuation;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );


RT_PROGRAM void any_hit_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);

  rtTerminateRay();
}

RT_PROGRAM void closest_hit_radiance()
{
  prd_radiance.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Kernels for generating 4D table of photon bin counts contributions
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//rtBuffer<uint>  rtpass_global_photon_counts;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<float3, 2>         pbin_output_buffer;
//rtBuffer<PhotonRecord, 1> pbin_output_buffer;
//rtDeclareVariable(float4, dhdH, , );
//uniform float r;
//uniform vec4 dhdH;

rtDeclareVariable(int, gR_ind, ,);
rtDeclareVariable(float, Rt, ,);
rtDeclareVariable(float, Rg, ,);
rtDeclareVariable(float, RL, ,);
rtDeclareVariable(float3, betaR, ,);
rtDeclareVariable(float3, betaMEx, ,);
rtDeclareVariable(float3, betaMSca, ,);
rtDeclareVariable(float, HR, ,);
rtDeclareVariable(float, HM, ,);
rtDeclareVariable(float, mieG, ,);

rtDeclareVariable(int, RES_R, ,);
rtDeclareVariable(int, RES_MU, ,);
rtDeclareVariable(int, RES_MU_S, ,);
rtDeclareVariable(int, RES_NU, ,);
rtDeclareVariable(int, RES_MU_S_BIN, ,);
rtDeclareVariable(int, RES_R_BIN, ,);
rtDeclareVariable(int, RES_DIR_BIN_TOTAL, ,);
rtDeclareVariable(int, RES_DIR_BIN, ,);


#define INSCATTER_NON_LINEAR
__device__ __inline__ void getMuMuSNu(float r, float4 dhdH, /*out*/ float& mu, /*out*/ float& muS, /*out*/ float& nu) {
	size_t2 size  = pbin_output_buffer.size();
    //float x = (size.x/2 + launch_index.x) % size.x - 0.5;
    float x = launch_index.x - 0.5;
    float y = launch_index.y - 0.5;
#ifdef INSCATTER_NON_LINEAR
    if (y < float(RES_MU) / 2.0) {
        float d = 1.0 - y / (float(RES_MU) / 2.0 - 1.0);
        d = min(max(dhdH.z, d * dhdH.w), dhdH.w * 0.999);
        mu = (Rg * Rg - r * r - d * d) / (2.0 * r * d);
        mu = min(mu, -sqrtf(1.0 - (Rg / r) * (Rg / r)) - 0.001);
    } else {
        float d = (y - float(RES_MU) / 2.0) / (float(RES_MU) / 2.0 - 1.0);
        d = min(max(dhdH.x, d * dhdH.y), dhdH.y * 0.999);
        mu = (Rt * Rt - r * r - d * d) / (2.0 * r * d);
    }
    muS = ((int)x % (int)RES_MU_S) / (float(RES_MU_S) - 1.0);
    // paper formula
    //muS = -(0.6 + log(1.0 - muS * (1.0 -  exp(-3.6)))) / 3.0;
    // better formula
    muS = tan((2.0 * muS - 1.0 + 0.26) * 1.1) / tan(1.26 * 1.1);
    nu = -1.0 + floor(x / float(RES_MU_S)) / (float(RES_NU) - 1.0) * 2.0;
#else
    mu = -1.0 + 2.0 * y / (float(RES_MU) - 1.0);
    muS = ((int)x % (int)RES_MU_S) / (float(RES_MU_S) - 1.0);
    muS = -0.2 + muS * 1.2;
    nu = -1.0 + floor(x / float(RES_MU_S)) / (float(RES_NU) - 1.0) * 2.0;
#endif
}
/*
__device__ __inline__ int findPosBin(float muS)
{
	int ind = floor((float)((muS + 1.0)/2.0)*(int)(RES_MU_S_BIN-1));
	return ind;
}
*/



__device__ __inline__ void init_r_dhdH(int layer, float& r, float4& dhdH)
{
	r = layer / (RES_R - 1.0);
    r = r * r;
    r = sqrtf(Rg * Rg + r * (Rt * Rt - Rg * Rg)) + (layer == 0 ? 0.01 : (layer == RES_R - 1 ? -0.001 : 0.0));
    float dmin = Rt - r;
    float dmax = sqrtf(r * r - Rg * Rg) + sqrtf(Rt * Rt - Rg * Rg);
    float dminp = r - Rg;
    float dmaxp = sqrtf(r * r - Rg * Rg);
    dhdH.x = dmin;
    dhdH.y = dmax;
    dhdH.z = dminp;
    dhdH.w = dmaxp;
}


__device__ __inline__ float3 photon_gather(uint pm_index, int r_ind, float r, float muS, float mu, float nu)
//(float first, float3 last, float d, float mu, int type ) 
{
	r = clamp(r, Rg, Rt);
    mu = clamp(mu, -1.0, 1.0);
    muS = clamp(muS, -1.0, 1.0);
    float var = sqrtf(1.0 - mu * mu) * sqrtf(1.0 - muS * muS);
    nu = clamp(nu, muS * mu - var, muS * mu + var);
	
	float3 result = viewing_ray(pm_index, r, muS, mu, nu);
	return result;
    
}


RT_PROGRAM void pbin_camera()
{
	size_t2 size  = pbin_output_buffer.size();
	uint pm_index = (launch_index.y * size.x + launch_index.x);
	float4 dhdH = make_float4(0.0);
	float r = 0.0;
	float mu = 0.0, muS = 0.0, nu = 0.0; 
	//size_t3 ind;
	//ind.x = 0; 
	//ind.y = launch_index.x;
	//ind.z = launch_index.y;
	
	size_t2 ind;
	ind.x = (launch_index.x);
	ind.y = (launch_index.y);
	
	//for (int i = 0; i < RES_R; i++){
	int i = gR_ind;
	init_r_dhdH(i, r, dhdH);
	getMuMuSNu(r, dhdH, mu, muS, nu);
	//ind.x = i;
	pbin_output_buffer[ind] = photon_gather(pm_index, i, r, muS, mu, nu);
	//rtPrintf("pbin_output_buffer[%d,%d,%d]=test\n",ind.x,ind.y,ind.z);
	//rtPrintf("pbin_output_buffer[%d,%d,%d]=(%f,%f,%f)\n",i,ind.x,ind.y, pbin_output_buffer[ind].x, pbin_output_buffer[ind].y,pbin_output_buffer[ind].z);
	//}
}

/*
RT_PROGRAM void pbin_camera()
{ 
  float2 screen = make_float2( rtpass_output_buffer.size() );

  //uint   seed   = image_rnd_seeds[index];                       // If we start writing into this buffer here we will
  //float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );      // need to make it an INPUT_OUTPUT buffer.  For now it
  //image_rnd_seeds[index] = seed;                                // is just INPUT
  
  float2 sample = make_float2( 0.5f, 0.5f ); 

  float2 d = ( make_float2(launch_index) + sample ) / screen * 2.0f - 1.0f;
  
  float3 ray_origin;// = rtpass_eye;
  float3 ray_direction;// = normalize(d.x*rtpass_U + d.y*rtpass_V + rtpass_W);
  ray_dir_org(ray_direction, ray_origin, d);
  
  optix::Ray ray(ray_origin, ray_direction, rtpass_ray_type, scene_epsilon);

  HitRecord& rec = rtpass_output_buffer[ launch_index ];
  rec.attenuated_Kd = make_float3(0.0);
  rec.depth = 0;
  
  HitPRD prd;
  // rec.ray_dir = ray_direction; // set in rtpass_closest_hit
  prd.attenuation = make_float3( 1.0f );
  prd.ray_depth   = 0u; 
  prd.ray_vol_depth   = 0u; 
  prd.prev_ray_length = 0.0f;
  //prd.Lv = make_float3(0.0f);
  //prd.L0 = make_float3(1.0);
  prd.total_dist = 0.0;
  prd.bSurfaceReached = 0;
  prd.step = -1.0;
  prd.numSteps = 1;
  if(bruneton_single == 1)
	rtTrace( single_top_object, ray, prd );
  else
	rtTrace( top_object, ray, prd );
  
  //rtpass_output_buffer
}
*/



RT_PROGRAM void pbin_closest_hit()
{

}

__device__ __inline__ float3 viewing_ray(uint pm_index, float r, float muS, float mu, float nu)
{
	
	float3 final_result = make_float3(0.0); 
	float3 direction = make_float3(0.0);
	
	float3 light_dir = make_float3(0,0,-1.0);//make_float3(-1,0,0);
	// for light dir (0,0,-1.0)
	direction.z = -nu;
	direction.y = (mu - muS*nu)/sqrtf(1- muS*muS);
	direction.x = sqrtf(1.0 - direction.x*direction.x - direction.z*direction.z);
	
	//direction = make_float3(sqrt(1.0 - mu * mu), 0.0, mu);
	
	float3 origin       = make_float3(0.0);
	origin.y = sqrtf(1.0 -muS*muS);
	origin.x = 0.0;
	origin.z = -muS;
	origin = r * origin;
	
	//if( fabs(muS) < 0.1){
	//	final_result = make_float3(10000.0,0.0,0.0);
	//	return final_result;
	//}
	
	//if( fabs(mu) < 0.1){
	//	final_result = make_float3(0.0,10000.0,0.0);
	//	return final_result;
	//}

	
	/*
	float3 light_dir = make_float3(0,-1.0,0.0);//make_float3(-1,0,0);
	// for light dir (0,-1.0,0.0)
	direction.y = -nu;
	direction.x = (mu - muS*nu)/sqrtf(1- muS*muS);
	direction.z = sqrtf(1.0 - direction.x*direction.x - direction.z*direction.z);
	
	float3 origin       = make_float3(0.0);
	origin.x = sqrtf(1.0 -muS*muS);
	origin.z = 0.0;
	origin.y = -muS;
	origin = r * origin;
	*/
	optix::Ray ray(origin, direction, pbinpass_ray_type, scene_epsilon);
		
	float3 hit_point    = origin;// + t_hit*direction;
	
    // first point
	float3 x = origin;
	if(length(origin) <= Rt && length(origin) >= Rg){
		x = origin;
	}
	// view direction
	float3 v = normalize(direction);

	
	
	float t = 0.0;//-r * mu;
    
	if((r * r * (mu * mu - 1.0) + Rg * Rg) >= 0.0){
		t =  -r * mu - sqrtf(r * r * (mu * mu - 1.0) + Rg * Rg);
	}
		
	final_result = make_float3(0.0);

	
	
	float3 temp = make_float3(1.0);
	int count = 0;
	float arr[24];
	bool bt_hit = false;
	float total_dist = 0.0;

	if(/*length(origin) < Rt ||*/ muS < 0.0){
		final_result = make_float3(0.0,0.0,0.0);
		return final_result;
	}
	else{ //length(origin) > Rt
		// keep track of:
		// -current sphere (si) 
		// -current cone   (ci)
		// -current sphere_direction (si_dir)
		// -current cone_direction (ci_dir)
	
	
		if((r * r * (mu * mu - 1.0) + Rg * Rg) >= 0.0){
			t =  -r * mu - sqrtf(r * r * (mu * mu - 1.0) + Rg * Rg);
		}
		if(t <= 0.0)
		{
			if((r * r * (mu * mu - 1.0) + Rt * Rt) >= 0.0){
				t =  -r * mu - sqrtf(r * r * (mu * mu - 1.0) + Rt * Rt);
				//t_start = t;
			}
		}
		
		int si = RES_R_BIN;
		int ci = -1;//RES_MU_S_BIN;
		
		int si_old = -2;
		int ci_old = ci;
		int num_si = 0;
		int num_ci = 0;
		
		int si_dir_old = -2;
		int si_dir = -1; // -1: down, +1: up
		int ci_dir = -1; // -1: down, +1: up
		bool b_si_dir = false;
		bool b_ci_dir = false;
		bool b_ci_change = false;
		
		float next_t_hit_s = 0.0;
		float next_t_hit_c = 0.0;
		float dist = 0.0;
		int num_photons = 0; //debugging
	
		float3 p = x;
		//if(t < 0.0){
		//	final_result = make_float3(10.0,0.0,0.0);
		//	return;
		//}
	
	    int layer;
	    float final_dist = 0.0;
	    if(length(origin) >= Rt){
			sphere_planet_atm_boundary_dist(pm_index, light_dir, p, v, layer, t, true, Rg, Rt, RES_R_BIN);
			si_dir = sphere_planet_atm_boundary_dist_ext(pm_index, light_dir, p, v, t,  si, next_t_hit_s, true, Rg, Rt, RES_R_BIN); //bSpace = true because we are in space
			final_dist = t;
		}
		else{
			sphere_planet_atm_boundary_dist(pm_index, light_dir, p, v, layer, t, true, Rg, Rt, RES_R_BIN);
			final_dist = t;
			si_dir = searchall_spheres_dir(pm_index, si_dir, light.direction,
													p, v, final_dist, 
													si, next_t_hit_s,
													Rt, Rg, RES_R_BIN, b_si_dir);
			//rtPrintf("<AFTER SEARCH>pm_index %d, si:%d, si_old:%d, si_dir:%d, <count:%d>, next_t_hit_s:%f, total_dist:%f, final_dist:%f last_layer:%d\n",
			//											pm_index, si, si_old, si_dir, 
			//											count, next_t_hit_s, total_dist, final_dist, layer);
			
		}
		
		float3 attn = make_float3(1.0);
		float3 total_attn = make_float3(1.0);
		int index = 0;
		float3 orig_p = p;
		if(si_dir != 0){
			//si = si+si_dir;
			int ci_temp = -1;
			ci_temp =  findPosBin((p + next_t_hit_s*v), light.direction, RES_MU_S_BIN);
			int i_si = si;
			int i_ci = ci;
			if(length(origin) >= Rt){
				total_dist += next_t_hit_s;
				//attn = analyticTransmittance(length(p), mu, total_dist, HR, HM, Rg, betaR, betaMEx);
				//total_attn = total_attn * attn;//(1-attn);
				attn = tranAtmNew(r, mu, trans_texture, p+next_t_hit_s*v, v,  Rg, Rt ,RL, betaR, betaMEx,
								HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
				total_attn = total_attn * attn;//(1-attn);
			
				index = (int)(ci_temp*RES_R_BIN + (si))*RES_DIR_BIN_TOTAL; 
				final_result += next_t_hit_s*photon_gather_contib_ext(pm_index, p, p+dist*v, v, index, num_photons, ci_temp, si);
				p = p + next_t_hit_s*v;
				
				next_t_hit_s = 0.0;
				si = si+si_dir;	
			}
			/*
			if(length(origin) < Rt){
				if((r * r * (mu * mu - 1.0) + Rg * Rg) >= 0.0){
					t =  -r * mu - sqrtf(r * r * (mu * mu - 1.0) + Rg * Rg);
					if(t < 0){
							rtPrintf("<BEFORE LOOP>pm_index %d, si:%d, si_old:%d, si_dir:%d, <count:%d>, next_t_hit_s:%f, total_dist:%f, final_dist:%f last_layer:%d\n",
														pm_index, si, si_old, si_dir, 
														count, next_t_hit_s, total_dist, final_dist, layer);
							final_result = make_float3(1.0,0.0,0.0);
							return;
					}
				}
			}
			*/
		
			 
			while((si_dir != 0 || ci_dir != 0) && count < ((RES_R_BIN+RES_MU_S_BIN)*5) /*&& total_dist < final_dist*/){
				// find first sphere intersection
				// find first cone intersection
				// update all booking
				// choose the step (to the closest intersection)
				//rtPrintf("pm_index %d, count:%d\n",pm_index, count); 
				
				
				if(next_t_hit_s == 0.0 && si_dir != 0){
					//si_dir = 0;
					si_old = si;
					si_dir_old = si_dir;
					
					//if(si == 0)
					//if(final_dist < 200 && count == 11)
					//rtPrintf(">sphere>>pm_index %d, si:%d, si_dir:%d, <count:%d>, next_t_hit_s:%f, total_dist:%f, final_dist:%f\n",
					//								pm_index, si, si_dir, 
					//								count, next_t_hit_s, total_dist, final_dist); 
					si_dir = intersect_spheres_dir(pm_index, si_dir, light.direction,
													p, v, final_dist, 
													si, next_t_hit_s,
													Rt, Rg, RES_R_BIN, b_si_dir);
					if(si_dir != 0)
						num_si++;
					
					//if(pm_index == 3327){//(pm_index == 24971)//(pm_index == 24980) //(pm_index == 341886)
				
					//if(num_ci == 1 && si == -1 && si_dir == 0){
				
					
					//if(length(origin) < Rt && next_t_hit_s<0.0 && si_dir != 0){
					/*
					if(si == 2 && si_dir == 0){
							rtPrintf(">sphere>>pm_index %d, si:%d, si_old:%d, si_dir_old:%d, <count:%d>, next_t_hit_s:%f, total_dist:%f, final_dist:%f last_layer:%d\n",
														pm_index, si, si_old, si_dir_old, 
														count, next_t_hit_s, total_dist, final_dist, layer); 
					
							final_result  = make_float3(0.0,0.0,1.0);
							return;								
					}
					*/
					//}
						
				}
				
				if(next_t_hit_c == 0.0 && ci_dir != 0){
					//ci_dir = 0;
					
					ci_dir = intersect_cones_dir(pm_index, ci_dir, light.direction, 
													p, v, final_dist, 
													ci,  next_t_hit_c,
													Rt, Rg, RES_MU_S_BIN, ray, b_ci_dir);
					if(ci_dir != 0)
						num_ci++;
					//if(count > 11)
					//if(out_ci != ci && out_ci != ci+ 1)
					
					//if(pm_index == 60344)//if(pm_index == 24971) //(pm_index == 341886)
					//	rtPrintf(">cone>>pm_index %d, ci:%d, out_ci:%d, ci_dir:%d, <count:%d>, next_t_hit_c:%f, total_dist:%f, final_dist:%f\n",
					//								pm_index, ci, ci_dir, 
					//								count, next_t_hit_c, total_dist, final_dist); 
					
					
					//if(ci == -1){
					//	//ci = findPosBin((orig_p + (t/2.0)*v), light.direction, RES_MU_S_BIN);
					//}
					
					//if(pm_index ==3327){
					//if(count >= 0 && count <= 13 && num_ci== 1 ){
					/*
					if(length(origin) < Rt && ci_dir != 0){
						if(count >= 0){ //&& si_dir == 1){
							rtPrintf(">cone>>pm_index %d, ci:%d, ci_dir:%d, si:%d, si_dir:%d, <count:%d>, next_t_hit_c:%f, next_t_hit_s:%f, total_dist:%f, final_dist:%f\n",
														pm_index, ci, ci_dir, si, si_dir,
														count, next_t_hit_c, next_t_hit_s, total_dist, final_dist); 
							//final_result  = make_float3(1.0,0.0,1.0);
							//return;	
						}
						else{
							rtPrintf(">cone>>pm_index %d, ci:%d, ci_dir:%d, si:%d, si_dir:%d, <count:%d>, next_t_hit_c:%f, next_t_hit_s:%f, total_dist:%f, final_dist:%f\n",
													pm_index, ci, ci_dir, si, si_dir,
													count, next_t_hit_c, next_t_hit_s, total_dist, final_dist); 
							//final_result  = make_float3(0.0,1.0,1.0);
							//return;	
						}
					}
					*/
					
				}
				
				if(si_dir != 0 && ci_dir != 0){
					
				
					dist = 0.0;
					float minborder = 0.00001;
					
					
					if((next_t_hit_s <= next_t_hit_c && next_t_hit_s > 0.0) || (next_t_hit_c == 0.0 && next_t_hit_s >0.0)){
						total_dist += next_t_hit_s;
						if(next_t_hit_c >= next_t_hit_s){
							next_t_hit_c = next_t_hit_c - next_t_hit_s;
						}
						
						//si = si + si_dir;
						//if(b_si_dir){
						//	i_si = (si == 1) ? 0 : si - 2;
						//}
						//else{
						//	i_si = si - si_dir;
						//}
					
						if(b_si_dir){
							i_si = (si == 1) ? 0 : si - 1;
						}
						else{
							i_si = (si_dir == 1) ? si -1 : si;
						}
						si = si + si_dir;
						
						dist = next_t_hit_s;
						next_t_hit_s = 0.0;
					}
					else if((next_t_hit_c < next_t_hit_s && next_t_hit_c > 0.0) || (next_t_hit_s == 0.0 && next_t_hit_c > 0.0)){
						
						if(next_t_hit_s <= 0.0)
							next_t_hit_s = 0.0;
							
						total_dist += next_t_hit_c;
						if(next_t_hit_s >= next_t_hit_c){
							next_t_hit_s = next_t_hit_s - next_t_hit_c;
						}
						
						
						if(b_ci_dir){
							i_ci = (ci == 1) ? 0 : ci - 1;
						}
						else{
							i_ci = (ci_dir == 1) ? ci -1 : ci;
						}
						ci = ci + ci_dir;
						
						dist = next_t_hit_c;
						
						ci_temp =  findPosBin(p, light.direction, RES_MU_S_BIN);
						if(count > 3 && count < 6 && num_ci== 1 && ci_dir == -1 && si_dir == -1){
							//rtPrintf(">cone>>pm_index %d, ci:%d, ci_dir:%d, i_ci:%d, si:%d, si_dir:%d, i_si:%d, ci_temp:%d, num_si:%d, <count:%d>, next_t_hit_c:%f, next_t_hit_s:%f\n",
							//							pm_index, ci, ci_dir, i_ci, si, si_dir, i_si, ci_temp, num_si, 
							//							count, next_t_hit_c, next_t_hit_s); 
							//final_result = make_float3(0.0);
							//return;	
						}
						next_t_hit_c = 0.0;
					}
					else{
						rtPrintf(">else>>pm_index %d, ci:%d, ci_dir:%d, si:%d, si_dir:%d, <count:%d>, next_t_hit_c:%f, next_t_hit_s:%f, total_dist:%f, final_dist:%f\n",
													pm_index, ci, ci_dir, si, si_dir,
													count, next_t_hit_c, next_t_hit_s, total_dist, final_dist); 
						//final_result  = make_float3(10.0,0.0,0.0);
						//return;
					}
					
					//ci_temp =  findPosBin((orig_p + (final_dist/2.0)*v), light.direction, RES_MU_S_BIN);
					//ci_temp =  findPosBin((p + (dist/2.0)*v), light.direction, RES_MU_S_BIN);
					ci_temp =  findPosBin(p, light.direction, RES_MU_S_BIN);
					if(dist > 0){
						if(ci == -1){
							index = (int)(ci_temp*RES_R_BIN + (i_si))*RES_DIR_BIN_TOTAL; 
							//final_result  += dist*total_attn*photon_gather_contib_ext(pm_index, p, p+dist*v, v, index, num_photons, ci_temp, i_si);
							//final_result  = make_float3(1.0,0.0,0.0);
							//return;
							
						}
						else{
							index = (int)(ci_temp*RES_R_BIN + (i_si))*RES_DIR_BIN_TOTAL;
							//index = ((i_ci)*RES_R_BIN + (i_si))*RES_DIR_BIN_TOTAL;
							if(index <= ((RES_MU_S_BIN-1)*RES_R_BIN + RES_R_BIN-1)*RES_DIR_BIN_TOTAL){
								final_result  += dist*total_attn* photon_gather_contib_ext(pm_index, p, p+dist*v, v, index, num_photons, ci_temp, i_si);
								//if((ci == 8 || (ci>=10 && ci < 20)) && (si >= 3)){
								//if(count > 3 && count < 6 && num_ci== 1 && ci_dir == -1 && si_dir == -1 && ci_temp != i_ci){
								//	rtPrintf(">cone>>pm_index %d, ci:%d, ci_dir:%d, i_ci:%d, si:%d, si_dir:%d, i_si:%d, ci_temp:%d, num_si:%d, <count:%d>, next_t_hit_c:%f, next_t_hit_s:%f\n",
								//								pm_index, ci, ci_dir, i_ci, si, si_dir, i_si, ci_temp, num_si, 
								//								count, next_t_hit_c, next_t_hit_s); 
									//final_result += dist*total_attn* make_float3(0.0,0.0,0.0);
									//return;	
								//}
							}
							
							
						}
					}
					/*
					if(ci != -1 && num_photons == 0 && count == 0){
					
						if((ci-ci_dir) >= 20 && (ci-ci_dir) <= 20){
				
								rtPrintf("<<ci_ci_dir>>pm_index %d, index: %d, ci-ci_dir:%d, ci_temp:%d, <num_photons:%d> ci_dir:%d, <count:%d>, final_dist:%f dist:%f\n",
													pm_index, index, ci-ci_dir, ci_temp, num_photons, ci_dir, 
													count, final_dist, dist);
								final_result  += total_attn*make_float3(0.0,1.0,0.0);
						
						}
					}
					*/
					
					
				
				}
				else{
					
					dist = 0.0;
					
					if(ci_dir != 0){
						total_dist += next_t_hit_c;
						if(next_t_hit_s >= next_t_hit_c){
							next_t_hit_s = next_t_hit_s - next_t_hit_c;
						}
						
						if(b_ci_dir){
							i_ci = (ci == 1) ? 0 : ci - 1;
						}
						else{
							i_ci = (ci_dir == 1) ? ci -1 : ci;
						}
						ci = ci + ci_dir;
						dist = next_t_hit_c;
						next_t_hit_c = 0.0;
						
						
						
					}
					
					if(si_dir != 0){
						total_dist += next_t_hit_s;
						
						if(b_si_dir){
							i_si = (si == 1) ? 0 : si - 1;
						}
						else{
							i_si = (si_dir == 1) ? si -1 : si;
						}
						si = si + si_dir;
						dist = next_t_hit_s;
						next_t_hit_s = 0.0;
						
					}
					else{ //ERROR
						//final_result = make_float3(10.0,0.0,0.0);
						//return;
						//break;
					}
					
					if(dist > 0.0){
						//attn = analyticTransmittance(length(p), mu, dist, HR, HM, Rg, betaR, betaMEx);
						//total_attn = total_attn * attn;
						//index = ((ci-ci_dir)*RES_R_BIN + (si-si_dir))*RES_DIR_BIN_TOTAL;
						
						
						if(ci_dir == 0){//ci == -1){
						    //ci_temp =  findPosBin((orig_p + (final_dist/2.0)*v), light.direction, RES_MU_S_BIN);
						    ci_temp =  findPosBin(p, light.direction, RES_MU_S_BIN);
						    
						    index = (int)(ci_temp*RES_R_BIN + i_si)*RES_DIR_BIN_TOTAL;
							final_result  += dist*total_attn*photon_gather_contib_ext(pm_index, p, p+dist*v, v, index, num_photons, ci_temp, i_si);
							//final_result  += total_attn*make_float3(0.0,0.1,0.1);
							//if(pm_index == 146184)
							//rtPrintf("<<test gathering>pm_index %d, index: %d, si-dir:%d, ci_temp:%d, <num_photons:%d> si_dir:%d, <count:%d>, final_dist:%f dist:%f\n",
							//					pm_index, index, si-si_dir, ci_temp, num_photons, si_dir, 
							//					count, final_dist, dist); 
						
						}
						else {
							index = (int)(i_ci*RES_R_BIN + i_si)*RES_DIR_BIN_TOTAL;
							final_result  += dist*total_attn*photon_gather_contib_ext(pm_index, p, p+dist*v, v, index, num_photons, i_ci, i_si);
							//final_result = make_float3(0.0);
							//return;
						}
						
						
					}
					else{
						//if(pm_index == 23434)
						//	rtPrintf(">(sp and cone) pm_index:%d, count:%d, si:%d, ci:%d, si_dir:%d, ci_dir:%d, total_dist:%f, final_dist:%f\n",
						//								pm_index, count, si, ci, si_dir, ci_dir, total_dist, final_dist); 
						//final_result  = make_float3(1.0,0.0,0.0);
						//break;
					}
					
					
					
				}
				
				
				if(dist > 0){
					attn = tranAtmNew(r, mu, trans_texture, p+dist*v, v,  Rg, Rt ,RL, betaR, betaMEx,
							HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
					total_attn = total_attn * attn;//(1-attn);
					//total_attn = make_float3(1.0); 
					
					p += dist*v;
					r = length(p);
					mu = dot(p, v)/r;
					if((r * r * (mu * mu - 1.0) + Rg * Rg) >= 0.0){
						t =  -r * mu - sqrtf(r * r * (mu * mu - 1.0) + Rg * Rg);
					}
					if(t <= 0.0)
					{
						if((r * r * (mu * mu - 1.0) + Rt * Rt) >= 0.0){
							t =  -r * mu - sqrtf(r * r * (mu * mu - 1.0) + Rt * Rt);
						}
					}
				}
				count++;
			}
			
			/*
			if(total_dist < final_dist-1.0){
				rtPrintf(">GREEN>>pm_index %d, si:%d, si_dir:%d, si_old:%d, si_dir_old:%d, ci:%d, ci_dir:%d, <count:%d>, next_t_hit_s:%f, next_t_hit_c:%f, total_dist:%f, final_dist:%f\n",
													pm_index, si, si_dir, si_old, si_dir_old, ci, ci_dir, 
													count, next_t_hit_s, next_t_hit_c, total_dist, final_dist); 
				final_result = make_float3(0.0,10.0,0.0);
				return;
			}
			*/
			
		
			//final_result *= SCALE_FACTOR;//1000000.0;//0.01;//100.0;//1000000000.0;//;
			//if((pm_index == 219778)){
			//	rtPrintf("<<>>pm_index %d, count:%d, final_result(%f,%f,%f)\n",pm_index, count, final_result.x,
			//	final_result.y, final_result.z);
			//}
			//if(pm_index == 30271){
			//	final_result = make_float3(0.4,0.0,1.0);
			//}
			
			//if(pm_index == 22626){
			//	final_result = make_float3(0.0,4.0,1.0);
			//}
			//if(length(p) < (Rt-0.1) && length(p) > (Rg+0.2)){ 
			
			//}
			
			
			
		}
		else{// (si_dir == 0)
			//final_result = make_float3(165.0/255.0,42.0/255.0,42.0/255.0);
			//return final_result;
		}
		
		if(final_result.x <= 0.1 && final_result.y <= 0.1 && final_result.z <= 0.1){
				//if(count == 12){
				//	final_result  += inscatter(p, t, v, light.direction, 100.0, r, mu, temp, true);
				//final_result = make_float3(1.0,1.0,0.0);
				//rtPrintf("<RED>pm_index %d, count:%d,t:%f, total_dist:%f, final_dist:%f, len(p):%f\n",pm_index, count, t, total_dist, final_dist, length(p)); 
				//final_result = make_float3(10.0,00.0,0.0);
				//return;
		}
		//final_result *= 1000000000.0;
		
		
	}
	//if(count > 2)
	//rtPrintf("pm_index %d, count:%d,t:%f, total_dist:%f\n",pm_index, count, t, total_dist); 
	return final_result;
}


__device__ __inline__ float3 photon_gather_contib_ext(int pm_index, float3 x_start, float3 x_end, float3 v_in, int index, int & num_photons, int ci, int si)
{
    // Input: x_start (ray start poistion) x_end (ray primitive intersection) v_in (viewing direction)
	
	// Calculate mid coridnates x which is in the middle of the path through a bin
	// Calculate bin R id and mu_s id: use them to calculate Bin id
	float volume = posbin_volume(ci, si, Rt, Rg, RES_R_BIN, RES_MU_S_BIN);
	
	
	
	//if(si == 8){
	//	return make_float3(0.0,0.4,0.4);	
	//}else if(si == 4){
	//	return make_float3(0.2,0.3,0.0);
	//}
	
	if(TEST_RT_PASS>0){
	
		//return make_float3(1.0,0.0,0.0);
		if(ci == 5 || (ci >59 && ci < 100)){
			return make_float3(0.09,0.0,0.0);	
		}else if(ci == 4 || (ci > 99 && ci < 150)){
			return make_float3(0.0,0.09,0.0);
		}else if(ci == 9 || (ci > 34 && ci < 60)){
			return make_float3(0.0,0.0,0.09);
		}else if(ci > 19 && ci < 35){
		//	return make_float3(0.5,0.0,0.0);
		}else if(ci == 8 || (ci>=10 && ci < 20)){
			if(si < 20)
				return make_float3(0.03,0.004,0.0);
			else
				return make_float3(0.0,0.0,0.0);
		}
		return make_float3(0.0,0.1,0.1);
	}
	
	float3 x;
	//x = (x_end-x_start)/2.0;
	x = x_start;
	
	float3 v_rot = v_in;
    float3 x_rot = x;
    rotateVectorToLightPlane(light, x_rot, v_rot);
    
    float3 s = make_float3(-1.0,0.0,0.0);
	float nu = dot(v_rot, s);
	float mu = dot(normalize(x_rot),v_rot);
	float muS = dot(normalize(x_rot),s);
	float r = sqrtf(dot(x,x));
	
	r = clamp(r, Rg, Rt);
    mu = clamp(mu, -1.0, 1.0);
    muS = clamp(muS, -1.0, 1.0);
    float var = sqrtf(1.0 - mu * mu) * sqrtf(1.0 - muS * muS);
    nu = clamp(nu, muS * mu - var, muS * mu + var);

    float cthetamin = -sqrtf(1.0 - (Rg / r) * (Rg / r));

	float vy = (mu - muS*nu)/sqrtf(1-muS*muS);
    float3 v = make_float3(-nu, vy, sqrtf(1-nu*nu-vy*vy));
    
    //float sx = v.x == 0.0 ? 0.0 : (nu - muS * mu) / v.x;
    //float3 s = make_float3(sx, sqrtf(max(0.0, 1.0 - sx * sx - muS * muS)), muS);
    
	float3 result = make_float3(0.0);
	uint3 total_in_bin = make_uint3(0);;
	uint3 w_weight = make_uint3(0);
	
		
	//int index = 0;
	//uint val = rtpass_global_photon_counts[index];
	
	//(1/extinction)* sum of [ phase function*flux/(Volume of sectors)]
	int x_ind = 0;
	int y_ind = 0;
	int z_ind = 0;
	float3 w = make_float3(0.0);
	
	float nu1 = 0.0;
	float nu2 = 0.0;
	float pr2 = 0.0;
	float pm2 = 0.0;
	float pr1 = 0.0;
	float pm1 = 0.0;
	
	float3 raymie1 = make_float3(1.0);
	float3 rayDen = betaR*exp(-(r - Rg) / HR);
	float3 mieDen = betaMSca*exp(-(r - Rg) / HM);
	
	//uint3 testdata = rtpass_global_photon_counts[index];
	//if( testdata.x != 0 || testdata.y != 0 || testdata.z != 0)
 	//	rtPrintf("testdata[%d]: (%d,%d,%d) \n",index,testdata.x,testdata.y, testdata.z);	
	//testdata = rtpass_global_photon_counts[index+1];
	//rtPrintf("testdata[%d]: (%d,%d,%d) \n",index+1,testdata.x,testdata.y, testdata.z);	
	
	for(int i=0; i < RES_DIR_BIN_TOTAL; i++){
		if(TEST_RT_PASS>0){
			w_weight = make_uint3(0);
		}
		else{
			w_weight = rtpass_global_photon_counts[index+i];
		}
			
			
		//if(w_weight>0){
		if(w_weight.x+w_weight.y+w_weight.z > 0){
			w = directionFromBin(i,RES_DIR_BIN);
			
			nu2 = dot(normalize(v_rot), normalize(w));
			pr2 = phaseFunctionR(nu2);
			pm2 = phaseFunctionM(nu2, mieG);
			
			
			
			//result += w_weight*(pr2 * betaR * expf(-(r - Rg) / HR)   +   pm2* betaMSca * expf(-(r - Rg) / HM));
			result.x += (float)w_weight.x*(pr2 + pm2);//  +   pm2* expf(-(r - Rg) / HM));
			result.y += (float)w_weight.y*(pr2 + pm2);// * expf(-(r - Rg) / HR)   +   pm2* expf(-(r - Rg) / HM));
			result.z += (float)w_weight.z*(pr2 + pm2);// * expf(-(r - Rg) / HR)   +   pm2* expf(-(r - Rg) / HM));
			//rtPrintf("(ind:%d, total:%d, w(%f,%f,%f), v(%f,%f,%f), nu2:%f, pr2:%f, pm2:%f)\n", index, total_in_bin, w.x, w.y, w.z, v.x,v.y,v.z, nu2, pr2, pm2);	
				
			total_in_bin += w_weight;
		}
		
	}
	//if(index == 1040 || index == 1032 || index == 1048)
	//	rtPrintf("(ind:%d, total:%d, result(%f,%f,%f), r:%f)\n", index, total_in_bin, result.x, result.y, result.z, r);	
	//if(total_in_bin > 0.0)
	//	rtPrintf("(%d,%d) %d\n",muS_bound,r_ind, total_in_bin);		
	
	//float volume = 1000;//posbin_volume(x, s, Rt, Rg, RES_R_BIN, RES_MU_S_BIN);
	
	//if (volume == 0.0)
	//volume = 1000;
	//float3 attenuation = analyticTransmittance(r, mu, d, HR, HM, Rg, betaR, betaMEx);
	
	float3 attenuation = make_float3(0.0);
	//r_start = sqrtf(dot(x_start,x_start);
	//mu_start = dot(normalize(x_start),v_in);
	//float3 attenuation = tranAtm(r_start, mu_start,
	//							x_end, v_in,
	//							Rg, Rt, RL,
	//							betaR, betaMEx,
	//							HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
	//if(total_in_bin > 0.0)
	//	rtPrintf("volume :%f\n",volume);
		//rtPrintf("(%d,%d) res(%f,%f,%f) atten(%f,%f,%f) %f\n",muS_bound,r_ind, result.x,result.y,result.z,attenuation.x,attenuation.y,attenuation.z,volume);	
	
	
	
	
	//if(total_in_bin > 0 && pm_index == 30271){// && ci == 17){
		//rtPrintf(">>>>pm_index:%d, ind:%d, total_in_bin:%d, result(%f,%f,%f), ci:%d, si:%d\n", pm_index, index, total_in_bin, result.x, result.y, result.z, ci, si);	
		//return make_float3(0.0,4.0,1.0);
	//}
	//if(result.x > 0.0 || result.y > 0.0 || result.z > 0.0)
	//	rtPrintf(">>>>pm_index:%d, ind:%d, total_in_bin:(%d,%d,%d), %f\n", pm_index, index, total_in_bin.x,total_in_bin.y,total_in_bin.z, volume);
	
	//return SCALE_FACTOR*result/volume;
	//10000.0
	//return 14.0*result/volume;
	return 15.0*result/volume;
}

