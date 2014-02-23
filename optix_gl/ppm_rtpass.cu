
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
#include "gathering.h"
#include "ppm.h"
#include "path_tracer.h"
#include "random.h"
#include "helpers.h"

#define TRANSMITTANCE_NON_LINEAR
#define MAX_DEPTH 20 // one MILLION photons
using namespace optix;



//
// Scene wide variables
//
// hitpoint parameters
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
//rtBuffer<PackedPhotonRecord, 1>  photon_map;
//rtBuffer<PackedPhotonRecord, 1>  vol_photon_map;
rtBuffer<uint3>  rtpass_global_photon_counts;
rtDeclareVariable(int, RES_DIR_BIN, ,);
rtDeclareVariable(int, RES_DIR_BIN_TOTAL, ,);
rtDeclareVariable(float, SCALE_FACTOR, ,);
rtDeclareVariable(float,         viewStepSize, ,);
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(rtObject,      single_top_object, ,);
rtDeclareVariable(rtObject,          top_shadower, , );
//rtDeclareVariable(float,			numSteps, ,);
rtDeclareVariable(float,			stepSize, ,);
rtDeclareVariable(int,			bruneton_single, ,);
rtDeclareVariable(int,			jm_mult, ,);

rtDeclareVariable(float3, emissive, , );
rtDeclareVariable(uint,          max_vol_photon_count, , );
rtBuffer<uint2, 2>               photon_rnd_seeds;
rtDeclareVariable(float,		frame_number, ,);
rtBuffer<uint2, 2>               image_rnd_seeds;

rtDeclareVariable(float3,  sigma_s, , );
rtDeclareVariable(float3,  sigma_a, , );
rtTextureSampler<float4, 2> earth_texture;
rtTextureSampler<float4, 2> trans_texture;
rtTextureSampler<float4, 3> inscatterSampler;
rtTextureSampler<float4, 3> inscatterPhotonSampler;
rtTextureSampler<float4, 3> raySampler;
rtTextureSampler<float4, 3> mieSampler;
rtDeclareVariable(int, singleOnly, ,);

rtDeclareVariable(float, Rt, ,);
rtDeclareVariable(float, Rg, ,);
rtDeclareVariable(float, RL, ,);
rtDeclareVariable(float3, betaR, ,);
rtDeclareVariable(float3, betaMEx, ,);
rtDeclareVariable(float3, betaMSca, ,);
rtDeclareVariable(float, HR, ,);
rtDeclareVariable(float, HM, ,);
rtDeclareVariable(float, mieG, ,);
rtDeclareVariable(int, TEST_RT_PASS, ,);

rtDeclareVariable(int, RES_R, ,);
rtDeclareVariable(int, RES_MU, ,);
rtDeclareVariable(int, RES_MU_S, ,);
rtDeclareVariable(int, RES_NU, ,);
rtDeclareVariable(int, RES_MU_S_BIN, ,);
rtDeclareVariable(int, RES_R_BIN, ,);

rtDeclareVariable(int, TRANSMITTANCE_INTEGRAL_SAMPLES, ,);
rtDeclareVariable(int, INSCATTER_INTEGRAL_SAMPLES, ,);

rtDeclareVariable(float4, iview_matrix_row_0, , );
rtDeclareVariable(float4, iview_matrix_row_1, , );
rtDeclareVariable(float4, iview_matrix_row_2, , );
rtDeclareVariable(float4, iview_matrix_row_3, , );

rtDeclareVariable(float4, iproj_matrix_row_0, , );
rtDeclareVariable(float4, iproj_matrix_row_1, , );
rtDeclareVariable(float4, iproj_matrix_row_2, , );
rtDeclareVariable(float4, iproj_matrix_row_3, , );

rtDeclareVariable(float4, view_matrix_row_0, , );
rtDeclareVariable(float4, view_matrix_row_1, , );
rtDeclareVariable(float4, view_matrix_row_2, , );
rtDeclareVariable(float4, view_matrix_row_3, , );

rtDeclareVariable(float4, proj_matrix_row_0, , );
rtDeclareVariable(float4, proj_matrix_row_1, , );
rtDeclareVariable(float4, proj_matrix_row_2, , );
rtDeclareVariable(float4, proj_matrix_row_3, , );

rtDeclareVariable(float3,        new_eye, , );
rtTextureSampler<float4, 2>   diffuse_map;  
rtBuffer<float3, 3>         pbin_output_buffer;

__device__ __inline__ float3 photon_gather_contib(float3 x_start, float3 x_end, float3 v_in, int index);
__device__ __inline__ float3 photon_gather_contib_ext(int pm_index, float3 x_start, float3 x_end, float3 v_in, int index, int & num_photons , int ci, int si);
__device__ __inline__ float3 arial_perspective(float3 origin, float3 hit_point, float3 direction);
__device__ __inline__ float3 bincount_mult_scatter(float raylen, float3 hit_point, float3 point, float3 direction);
__device__ __inline__ float3 bincount_mult_scatterExt(float raylen, float3 hit_point, float3 point, float3 direction);
__device__ __inline__ float3 prepare_ss(float3 x, float3 v, float3 s, float sunPower, bool doAttenuation);
//__device__ __inline__ int sphere_intersection(	float3 x, float3 v, int updownsp, 
//												/* out*/ float &root1, /* out*/ float & root2, /*out*/int & radius_ind);
//__device__ __inline__ int cone_intersection(	float3 x, float3 v, float3 cone_data, 
//												/* out*/ float & t0, /* out*/ float & t1, /* out */ float & thit);		
//__device__ __inline__ int cone_next_boundary_dist(int pm_index, float3 x, float3 v, float curr_dist, float muS, float & out_dist, int & out_ind);										
__device__ __inline__ int shape_boundary_intersection(	int pm_index, float3 x, float3 v, float last_hit,  int & layer_sphere, int & layer_cone,
													  /* out*/ float &dist, /*out*/ bool & bSphere);
//#define FIX
#define INSCATTER_NON_LINEAR

__device__ __inline__ float3 atten_for_ground(float3 x, float t, float r, float mu, float3 v)
{
	float3 attenuation = make_float3(1.0);
	if (r <= Rt) { // if ray intersects atmosphere

		attenuation = analyticTransmittance(r, mu, t, HR, HM, Rg, betaR, betaMEx);
		if (t > 0.0) {
			float3 x0 = x + t * v;
			attenuation = tranAtmNew(r, mu, trans_texture, x0, v,  Rg, Rt ,RL, betaR, betaMEx,
				HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
		}
	}
	return attenuation;
}

//don't know how to pass texture as a parameter in CUDA/OptiX
__device__ __inline__ float4 tex4D_inscatter(texture<float4, 3> sampler, float r, float mu, float muS, float nu)
{
	float H = sqrtf(Rt * Rt - Rg * Rg);
	float rho = sqrtf(r * r - Rg * Rg);
#ifdef INSCATTER_NON_LINEAR
	float rmu = r * mu;
	float delta = rmu * rmu - r * r + Rg * Rg;
	float4 cst = rmu < 0.0 && delta > 0.0 ? make_float4(1.0, 0.0, 0.0, 0.5 - 0.5 / float(RES_MU)) : make_float4(-1.0, H * H, H, 0.5 + 0.5 / float(RES_MU));
	float uR = 0.5 / float(RES_R) + rho / H * (1.0 - 1.0 / float(RES_R));
	float uMu = cst.w + (rmu * cst.x + sqrtf(delta + cst.y)) / (rho + cst.z) * (0.5 - 1.0 / float(RES_MU));

	// paper formula
	float uMuS = 0.5 / float(RES_MU_S) + max((1.0 - exp(-3.0 * muS - 0.6)) / (1.0 - exp(-3.6)), 0.0) * (1.0 - 1.0 / float(RES_MU_S));
	// better formula
	//float uMuS = 0.5 / float(RES_MU_S) + (atan(max(muS, -0.1975) * tan(1.26 * 1.1)) / 1.1 + (1.0 - 0.26)) * 0.5 * (1.0 - 1.0 / float(RES_MU_S));
	//float uMuS = 0.5 / float(RES_MU_S) + ((max(muS, -0.1975) * tan(1.26 * 1.1)) / 1.1 + (1.0 - 0.26)) * 0.5 * (1.0 - 1.0 / float(RES_MU_S));
#else
	float uR = 0.5 / float(RES_R) + rho / H * (1.0 - 1.0 / float(RES_R));
	float uMu = 0.5 / float(RES_MU) + (mu + 1.0) / 2.0 * (1.0 - 1.0 / float(RES_MU));
	float uMuS = 0.5 / float(RES_MU_S) + max(muS + 0.2, 0.0) / 1.2 * (1.0 - 1.0 / float(RES_MU_S));
#endif
	float lerp = (nu + 1.0) / 2.0 * (float(RES_NU) - 1.0);
	float uNu = floor(lerp);
	lerp = lerp - uNu;
	float3 v1 = make_float3((uNu + uMuS) / float(RES_NU), uMu, uR);
	float3 v2 = make_float3((uNu + uMuS + 1.0) / float(RES_NU), uMu, uR);

	//float4 val1 = tex3D(inscatterSampler, v1.x, v1.y, v1.z);
	//float4 val2 = tex3D(inscatterSampler, v2.x, v2.y, v2.z);
	float4 val1 = tex3D(sampler, v1.x, v1.y, v1.z);
	float4 val2 = tex3D(sampler, v2.x, v2.y, v2.z);
	//rtPrintf("val1: v1.x=%f, v1.y=%f, v1.z=%f,-> (%f,%f,%f)\n",v1.x, v1.y, v1.z, val1.x,val1.y,val1.z);
	//rtPrintf("val2: v2.x=%f, v2.y=%f, v2.z=%f,-> (%f,%f,%f)\n",v2.x, v2.y, v2.z, val2.x,val2.y,val2.z);
	return val1 * (1.0 - lerp) + val2 * lerp;
}

__device__ __inline__ float4 tex4D_inscatterExt(texture<float4, 3> sampler, float r, float mu, float muS, float nu)
{
	float H = sqrtf(Rt * Rt - Rg * Rg);
	float rho = sqrtf(r * r - Rg * Rg);

	float uR = 0.5 / float(RES_R) + rho / H * (1.0 - 1.0 / float(RES_R));
	float uMu = 0.5 / float(RES_MU) + (mu + 1.0) / 2.0 * (1.0 - 1.0 / float(RES_MU));
	float uMuS = 0.5 / float(RES_MU_S) + max(muS + 0.2, 0.0) / 1.2 * (1.0 - 1.0 / float(RES_MU_S));

	float vlerp = (nu + 1.0) / 2.0 * (float(RES_NU) - 1.0);
	float uNu = floor(vlerp);
	vlerp = vlerp - uNu;
	float3 v1 = make_float3((uNu + uMuS) / float(RES_NU), uMu, uR);
	float3 v2 = make_float3((uNu + uMuS + 1.0) / float(RES_NU), uMu, uR);

	//float4 val1 = tex3D(inscatterSampler, v1.x, v1.y, v1.z);
	//float4 val2 = tex3D(inscatterSampler, v2.x, v2.y, v2.z);
	float4 val1 = tex3D(sampler, v1.x, v1.y, v1.z);
	float4 val2 = tex3D(sampler, v2.x, v2.y, v2.z);
	//rtPrintf("val1: v1.x=%f, v1.y=%f, v1.z=%f,-> (%f,%f,%f)\n",v1.x, v1.y, v1.z, val1.x,val1.y,val1.z);
	//rtPrintf("val2: v2.x=%f, v2.y=%f, v2.z=%f,-> (%f,%f,%f)\n",v2.x, v2.y, v2.z, val2.x,val2.y,val2.z);
	return val1 * (1.0 - vlerp) + val2 * vlerp;
}

__device__ __inline__ float2 transmittanceUV(float r, float mu) 
{
	float uR, uMu;
	float angle = 0.15;//0.15;
#ifdef TRANSMITTANCE_NON_LINEAR
	uR = sqrtf((r - Rg) / (Rt - Rg));
	uMu = atan((mu + angle) / (1.0 + angle) * tan(angle*10.0)) / (angle*10.0);
#else
	uR = (r - Rg) / (Rt - Rg);
	uMu = (mu + angle) / (1.0 + angle);
#endif
	return make_float2(uMu, uR);
}

__device__ __inline__ float3 transmittanceR_MU(float r, float mu) 
{
	float2 uv = transmittanceUV(r, mu);
	//int2 uv_int = make_int2(uv.x*TRANSMITTANCE_W,uv.y*TRANSMITTANCE_H);
	float4 tr = tex2D(trans_texture, uv.x, uv.y);
	return make_float3(tr.x,tr.y,tr.z);
}

// transmittance(=transparency) of atmosphere between x and x0
// assume segment x,x0 not intersecting ground
// r=||x||, mu=cos(zenith angle of [x,x0) ray at x), v=unit direction vector of [x,x0) ray
__device__ __inline__ optix::float3 tranAtm(float r, float mu,
											optix::float3 x0, optix::float3 v,
											float Rg, float Rt, float RL,
											optix::float3 betaR, optix::float3 betaMEx,
											float HR, float HM, float TRANSMITTANCE_INTEGRAL_SAMPLES) 
{

	optix::float3 result;
	float r1 = sqrtf(dot(x0,x0));
	float mu1 = dot(x0, v) / r1;

	if (mu > 0.0) {
		result = fminf(transmittanceR_MU(r, mu) / transmittanceR_MU(r1, mu1), make_float3(1.0));
	} else {
		result = fminf(transmittanceR_MU(r1, -mu1) / transmittanceR_MU(r, -mu), make_float3(1.0));
	}
	return result;
}


/*
// transmittance(=transparency) of atmosphere between x and x0
// assume segment x,x0 not intersecting ground
// d = distance between x and x0, mu=cos(zenith angle of [x,x0) ray at x)
__device__ __inline__ optix::float3 transmittance_RMuD(float r, float mu, float d) {
float3 result;
float r1 = sqrtf(r * r + d * d + 2.0 * r * mu * d);
float mu1 = (r * mu + d) / r1;
if (mu > 0.0) {
result = fminf(transmittanceR_MU(r, mu) / transmittanceR_MU(r1, mu1), make_float3(1.0));
} else {
result = fminf(transmittanceR_MU(r1, -mu1) / transmittanceR_MU(r, -mu), make_float3(1.0));
}
return result;
}
*/



//
// Ray generation program
//
rtBuffer<HitRecord, 2>           rtpass_output_buffer;
rtBuffer<zpSample, 2>			 rtpass_output_samples;
rtDeclareVariable(float,         rtpass_default_radius2, , );
rtDeclareVariable(uint,          rtpass_ray_type, , );
rtDeclareVariable(uint,          rtpass_shadow_ray_type, , );
rtDeclareVariable(float3,        rtpass_eye, , );
rtDeclareVariable(float3,        rtpass_U, , );
rtDeclareVariable(float3,        rtpass_V, , );
rtDeclareVariable(float3,        rtpass_W, , );
rtDeclareVariable(uint2,		 launch_index, rtLaunchIndex, );
rtDeclareVariable(uint,			 sqrt_samples_per_pixel, ,);

__device__ __inline__ void ray_dir_org(float3& ray_dir, float3& ray_org, float2 d)
{

	float4 d4= make_float4(d.x,d.y,1.0,1.0);
	float4 td = make_float4(0.0,0.0,0.0,0.0);
	td.x = dot(iproj_matrix_row_0, d4);
	td.y = dot(iproj_matrix_row_1, d4);
	td.z = dot(iproj_matrix_row_2, d4);
	td.w = 0.0;


	//float4 td = make_float4(0.0,0.0,0.0,0.0);
	ray_dir.x = dot(iview_matrix_row_0, td);
	ray_dir.y = dot(iview_matrix_row_1, td);
	ray_dir.z = dot(iview_matrix_row_2, td);

	/*
	ray_dir = d.x*make_float3(iview_matrix_row_0.x,iview_matrix_row_0.y,iview_matrix_row_0.z) 
	+ d.y*make_float3(iview_matrix_row_1.x,iview_matrix_row_1.y,iview_matrix_row_1.z) 
	+ make_float3(iview_matrix_row_2.x,iview_matrix_row_2.y,iview_matrix_row_2.z);
	*/

	ray_dir = normalize(ray_dir);
	ray_org = new_eye;
	//ray = (viewInverse * vec4((projInverse * gl_Vertex).xyz, 0.0)).xyz;
}

__device__ float2 get_new_sample( uint2 corner )
{
	float2 loc = make_float2( (corner.x + 0.5f) / sqrt_samples_per_pixel,
		(corner.y + 0.5f) / sqrt_samples_per_pixel ); 

	return loc;
}


RT_PROGRAM void rtpass_camera()
{ 
	float2 screen = make_float2( rtpass_output_buffer.size() );
	/*
	float2 loc = get_new_sample(launch_index);
	rtpass_output_samples[launch_index].x = loc.x;
	rtpass_output_samples[launch_index].y = loc.y;

	float2 screen = make_float2( rtpass_output_samples.size() );
	*/
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


rtDeclareVariable(int, filter_type, ,);
rtDeclareVariable(float, filter_width, ,);
rtDeclareVariable(float, gaussian_alpha, ,);


struct PerRayData_shadow
{
	float3 attenuation;
	float	 prev_ray_length;
	bool inShadow;
};

// 
// Closest hit material
// 
rtDeclareVariable(float3,  Ks, , );
rtDeclareVariable(float3,  Kd, , );
rtDeclareVariable(float3,  grid_color, , );
rtDeclareVariable(uint,    use_grid, , );
rtDeclareVariable(float3,  emitted, , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 

rtDeclareVariable(HitPRD, hit_prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );


rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(PPMLight, light, ,);


// kernel for the ground
RT_PROGRAM void rtpass_closest_hit()
{
	//HitRecord& rec = rtpass_output_buffer[launch_index]; 
	//rec.attenuated_Kd  += make_float3(1.0,0.0,0.0);
	//return;

	float3 direction    = ray.direction;
	float3 origin       = ray.origin;
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal     = faceforward( world_shading_normal, -direction, world_geometric_normal );
	float3 hit_point    = origin + t_hit*direction;
	hit_prd.bSurfaceReached = 1;

	//case of bin counts photon mapping

	if(bruneton_single == 0){
		HitRecord& rec = rtpass_output_buffer[launch_index];   
		hit_prd.total_dist += t_hit;
		//rec.attenuated_Kd += bincount_mult_scatterExt(t_hit, hit_point, origin, direction);
		//rec.attenuated_Kd += prepare_ss(origin, direction, light.direction, 100.0, false);
		rec.attenuated_Kd  += make_float3(1.0,0.0,0.0);
		hit_prd.total_dist -= t_hit;
		return;
	}


	if (t_hit > 0.0) { // if ray hits ground surface
		// ground reflectance at end of ray, x0
		float sunPower = 100.0;
		float3 s = light.direction;
		float3 v = direction; 
		float3 x0 = hit_point;
		float r0 = length(x0);
		float3 n = shading_normal;//x0 / r0;
		//vec2 coords = vec2(atan(n.y, n.x), acos(n.z)) * vec2(0.5, 1.0) / M_PI + vec2(0.5, 0.0);
		float2 coords = make_float2(atan2(n.y, n.x), acos(n.z)) * make_float2(0.5, 1.0) / M_PI + make_float2(0.5, 0.0);
		//float2 coords = make_float2(atan2(n.x, n.z), acos(n.y)) * make_float2(0.5, 1.0) / M_PI + make_float2(0.5, 0.0);
		//float4 reflectance = tex2D(earth_texture, texcoord.x, texcoord.y) * make_float4(0.2, 0.2, 0.2, 1.0);

		//brown
		float4 reflectance =  make_float4(165.0/255.0,42.0/255.0,42.0/255.0, 0.5);

		reflectance = tex2D(earth_texture, coords.x, coords.y) * make_float4(0.2, 0.2, 0.2, 1.0);
		if (r0 > Rg + 0.01) {
			reflectance = make_float4(0.4, 0.4, 0.4, 0.0);
		}


		// direct sun light (radiance) reaching x0
		float muS = dot(n, s);
		float3 sunLight = make_float3(0.01,0.01,0.01);//transmittanceWithShadow(r0, muS, trans_texture, Rg, Rt);//make_float3(0.0,0.2,0.0);
		//if(sunLight.x > 0.0 || sunLight.y > 0.0)
		//	rtPrintf("sunLight.x > 0.0 || sunLight.y > 0.0\n");

		// precomputed sky light (irradiance) (=E[L*]) at x0
		// ERROR: black circle gets introduced somewhere here
		//vec3 groundSkyLight = irradiance(irradianceSampler, r0, muS);

		// light reflected at x0 (=(R[L0]+R[L*])/T(x,x0))
		float3 groundColor = make_float3(reflectance.x,reflectance.y,reflectance.z) * (max(muS, 0.0) * sunLight /*+ groundSkyLight*/) * sunPower / M_PI;


		// water specular color due to sunLight
		if (reflectance.w > 0.0) {
			float3 h = normalize(s - v);
			float fresnel = 0.02 + 0.98 * pow(1.0 - dot(-v, h), 5.0);
			float waterBrdf = fresnel * pow(max(dot(h, n), 0.0), 150.0);
			groundColor += reflectance.w * max(waterBrdf, 0.0) * sunLight * sunPower;
		}
		HitRecord& rec = rtpass_output_buffer[ launch_index ];
		//rec.attenuated_Kd += emitted*hit_prd.attenuation; 
		rec.attenuated_Kd += groundColor;
		//rec.attenuated_Kd += arial_perspective(origin, hit_point, direction);
		rec.t_hit_exitvolume = 0;
		rec.flags = 0u;
		//hit_prd.L0 = rec.attenuated_Kd;
		return;
	} 

	// Check if this is a light source
	if( fmaxf( emitted ) > 0.0f ) {
		HitRecord& rec = rtpass_output_buffer[ launch_index ];
		//rec.attenuated_Kd = emitted*hit_prd.attenuation; 
		rec.attenuated_Kd += normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
		rec.attenuated_Kd += arial_perspective(origin, hit_point, direction);
		rec.t_hit_exitvolume = 0;
		rec.flags = 0u;
		//hit_prd.L0 = rec.attenuated_Kd;
		return;
	}


	if( fmaxf( Kd ) > 0.0f ) { 
		//rtPrintf("rtpass_closest_hit in if\n");
		// We hit a diffuse surface; record hit and return
		HitRecord rec;
		rec.position = hit_point; 
		rec.normal = ffnormal;

		/*
		// cast shadow ray
		PerRayData_shadow shadow_prd;
		shadow_prd.attenuation = make_float3(1.0f);
		float Ldist = 2000;
		optix::Ray shadow_ray( hit_point, -light.direction, rtpass_shadow_ray_type, scene_epsilon, Ldist );
		rtTrace(top_shadower, shadow_ray, shadow_prd);
		float3 light_attenuation = shadow_prd.attenuation;
		*/

		if( !use_grid ) {
			//rec.attenuated_Kd = Kd * hit_prd.attenuation;
			rec.attenuated_Kd += Kd*(normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f);
		} else {
			float grid_size = 50.0f; 
			float line_size = 2.0f; 
			float xx = ( hit_point.x + 1025.0f ) / grid_size;
			xx = ( xx - static_cast<float>( static_cast<int>( xx ) ) ) * grid_size;
			float zz = ( hit_point.z + 1025.0f ) / grid_size;
			zz = ( zz - static_cast<float>( static_cast<int>( zz ) ) ) * grid_size;
			if( xx < line_size  || zz < line_size ){
				rec.attenuated_Kd += grid_color * hit_prd.attenuation;
				//rec.attenuated_Kd = grid_color*(normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f);
			}
			else{
				rec.attenuated_Kd += Kd * hit_prd.attenuation;
				//rec.attenuated_Kd = Kd*(normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f);
			}
		}
		//rec.attenuated_Kd = (normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f);
		//rec.attenuated_Kd += arial_perspective(origin, hit_point, direction);
		//rec.attenuated_Kd *= light_attenuation;
		//rec.attenuated_Kd += arial_perspective(origin, hit_point, direction);

		//hit_prd.L0 = rec.attenuated_Kd;

		hit_prd.prev_ray_length = t_hit;
		rec.flags = PPM_HIT;

		rec.radius2 = rtpass_default_radius2;
		rec.photon_count = 0;
		rec.accum_atten = 0.0f;
		rec.flux = make_float3(0.0f, 0.0f, 0.0f);
		if(hit_prd.ray_depth > 0 && fmaxf( Kd ) > 0.0f ){
			//rtPrintf("Surface: rec.attenuated_Kd = (%f,%f,%f); Kd = (%f,%f,%f)\n",rec.attenuated_Kd.x, rec.attenuated_Kd.y, rec.attenuated_Kd.z, Kd.x, Kd.y, Kd.z);
			//rec.attenuated_Kd = make_float3(0.0,1.0,0.0);
		}
		rec.t_hit_exitvolume = 1;
		rtpass_output_buffer[launch_index] = rec;
	} else {
		// Make reflection ray
		//hit_prd.attenuation = hit_prd.attenuation * Ks + arial_perspective(origin, hit_point, direction);
		//hit_prd.L0 = hit_prd.attenuation;
		//hit_prd.ray_depth++;
		//float3 R = reflect( direction, ffnormal );
		//optix::Ray refl_ray( hit_point, R, rtpass_ray_type, scene_epsilon );
		//rtTrace( top_object, refl_ray, hit_prd );
	}

}


//
// Miss program
//
rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void rtpass_miss()
{

	float theta = atan2f( ray.direction.x, ray.direction.z );
	float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
	float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v     = 0.5f * ( 1.0f + sin(phi) );
	float3 result = make_float3(tex2D(envmap, u, v));

	HitRecord& rec = rtpass_output_buffer[launch_index];
	rec.flags = 0u;
	rec.t_hit_exitvolume = 0;
	if(bruneton_single == 0){
		return;
	}

	rec.attenuated_Kd = hit_prd.attenuation * result;
	//hit_prd.prev_ray_length = t_hit;


	if (t_hit > 0.0) {
		rec.attenuated_Kd = make_float3(0.0);
	} else {
		float3 x = ray.origin;
		float r = length(x);
		float muS = dot(x, light.direction) / r;
		//rtPrintf("r: %f\n",r);
		rec.attenuated_Kd = make_float3(0.0,0.0,0.0);
		if(fabs(muS) < 0.1)
			rec.attenuated_Kd = make_float3(1.0,0.0,0.0);
	}
}

/*
rtDeclareVariable(float3, rtpass_bg_color, , );
RT_PROGRAM void rtpass_miss()
{
HitPRD& prd = hit_prd.reference();
uint2 index = make_uint2( launch_index.get() );
HitRecord& rec = rtpass_output_buffer[index];

rec.flags = 0u;
rec.attenuated_Kd = prd.attenuation * rtpass_bg_color;
rec.t_hit_exitvolume = 0;
}
*/

//       
// Stack overflow program
//
rtDeclareVariable(float3, rtpass_bad_color, , );
RT_PROGRAM void rtpass_exception()
{
	HitRecord& rec = rtpass_output_buffer[launch_index];
	const unsigned int code = rtGetExceptionCode(); 
	//if( code == RT_EXCEPTION_STACK_OVERFLOW ) {
	//	rec.flags = PPM_OVERFLOW;
	//	rec.attenuated_Kd = make_float3(1.0,0.0,1.0);//rtpass_bad_color;
	//}
	//else 
	//	rec.attenuated_Kd = rtpass_bad_color;
	rec.t_hit_exitvolume = 0;

	const float3 buffer_index_out_of_bounds_color = make_float3(1,0,0); // red
	const float3 stack_overflow_color             = make_float3(1,1,0); // yellow
	const float3 invalid_ray_color                = make_float3(1,0,1); // magenta
	const float3 user0_color                      = make_float3(1,1,1); // white
	const float3 user1_color                      = make_float3(0,0.6f,0.85f); // blue

	float3 result;

	switch(code) {
case RT_EXCEPTION_STACK_OVERFLOW:
	rtPrintf("rtpass_exception = RT_EXCEPTION_STACK_OVERFLOW (%d)\n", rec.depth);

	result = stack_overflow_color;
	break;

case RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS:
	rtPrintExceptionDetails();
	rtPrintf("rtpass_exception = RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS\n");
	result = buffer_index_out_of_bounds_color;
	break;

case RT_EXCEPTION_INVALID_RAY:
	rtPrintf("rtpass_exception = RT_EXCEPTION_INVALID_RAY\n");
	result = invalid_ray_color;
	break;


default:
	rtPrintf("rtpass_exception = default\n");
	result = make_float3(1,0,0); // black for unhandled exceptions
	break;
	}
	rec.attenuated_Kd = result;
}


RT_PROGRAM void gather_samples_exception()
{
	HitRecord& rec = rtpass_output_buffer[launch_index];
	rec.flags = PPM_OVERFLOW;
	rec.attenuated_Kd = rtpass_bad_color;
}


// assume that the center of the sphere is at the center of the world
__device__ __inline__ float3 light_atten(float3 pos, float3 light_pos, float3 eye_dir)
{
	//CASES:
	// 1) inside the atmosphere (sun is visible)
	// 2) on the border of the atmosphere (sun is visible)
	// 3) reached the surface of the Earth
	// (NOT POSSIBLE) 2) inside the atmosphere (sun is not visible)
	// (NOT POSSIBLE) 3) outside the atmosphere (sun is visible)
	// (NOT POSSIBLE) 4) outside the atmosphere (sun is not visible)

	//float3 sigmaE = sigma_a + sigma_s;
	//float3 T = trans(pos, light_pos, sigmaE);


	float r = sqrtf(dot(pos,pos));
	float3 light_dir = light_pos-pos;
	float d = sqrtf(dot(light_dir,light_dir));
	float mu = dot(eye_dir,light_dir);

	float3 T = TransmittanceAtmosphereAnalytic(r, mu, d,
		Rg, Rt, RL,
		betaR, betaMEx,
		HR, HM);

	/*
	float3 T = TransmittanceAtmosphereIntegration(r, mu, d,
	Rg, Rt, RL,
	betaR, betaMEx,
	HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
	*/
	return T;
}

// assume that the center of the sphere is at the center of the world
__device__ __inline__ float3 sun_atten(float3 pos, float3 new_pos, float3 sundir, float3 eye_dir)
{
	//CASES:
	// 1) inside the atmosphere (sun is visible)
	// 2) on the border of the atmosphere (sun is visible)
	// 3) reached the surface of the Earth
	// (NOT POSSIBLE) 2) inside the atmosphere (sun is not visible)
	// (NOT POSSIBLE) 3) outside the atmosphere (sun is visible)
	// (NOT POSSIBLE) 4) outside the atmosphere (sun is not visible)

	//float3 sigmaE = sigma_a + sigma_s;
	//float3 T = trans(pos, light_pos, sigmaE);


	float r = sqrtf(dot(pos,pos));
	float mu = dot(eye_dir, sundir)/r;

	/*
	float3 T = TransmittanceAtmosphereAnalytic(r, mu,
	Rg, Rt, RL,
	betaR, betaMEx,
	HR, HM);
	*/

	float3 T = tranAtm(r, mu, new_pos, eye_dir,
		Rg, Rt, RL,
		betaR, betaMEx,
		HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
	//float3 T = tranAtm(r, mu, trans_texture, new_pos, eye_dir,
	//					Rg, Rt, RL,
	//					betaR, betaMEx,
	//					HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
	return T;
}


__device__ __inline__ 
void accumulateVolumePhoton( const PackedPhotonRecord& photon,
							uint& num_new_photons, float3& view_direction, float3& flux_VM )
{
	float3 photon_pos = make_float3( photon.a.x, photon.a.y, photon.a.z );
	float3 photon_energy = make_float3( photon.c.y, photon.c.z, photon.c.w );
	float3 photon_ray_dir = make_float3( photon.b.z, photon.b.w, photon.c.x );
	float3 flux = make_float3(0.0);
	float r = length(photon_pos);
	r = clamp(r, Rg, Rt);
	float muS = dot(photon_pos, light.direction) / r;
	/*

	mu = clamp(mu, -1.0, 1.0);
	muS = clamp(muS, -1.0, 1.0);
	float var = sqrtf(1.0 - mu * mu) * sqrtf(1.0 - muS * muS);
	nu = clamp(nu, muS * mu - var, muS * mu + var);

	float cthetamin = -sqrtf(1.0 - (Rg / r) * (Rg / r));

	vec3 v = vec3(sqrtf(1.0 - mu * mu), 0.0, mu);
	float sx = v.x == 0.0 ? 0.0 : (nu - muS * mu) / v.x;
	vec3 s = vec3(sx, sqrtf(max(0.0, 1.0 - sx * sx - muS * muS)), muS);
	*/

	//check out what the phase function thinks of your new direction
	//float3 ref = PhaseRayleigh(photon_ray_dir, view_direction); //scene->volumeRegion->p(interactPt,rn.d,direction);
	float3 v = view_direction;
	float3 w =  photon_ray_dir;
	float3 s = light.direction;
	float nu1 = dot(s, w);
	float nu2 = dot(v, w);
	float pr2 = phaseFunctionR(nu2);
	float pm2 = phaseFunctionM(nu2, mieG);

	// second term = inscattered light, =deltaS
	//if (frame_number == 0.0) {
	// first iteration is special because Rayleigh and Mie were stored separately,
	// without the phase functions factors; they must be reintroduced here
	float pr1 = phaseFunctionR(nu1);
	float pm1 = phaseFunctionM(nu1, mieG);
	float3 ray1 = make_float3(tex4D_inscatter(raySampler, r, w.z, muS, nu1));
	float3 mie1 = make_float3(tex4D_inscatter(mieSampler, r, w.z, muS, nu1));
	flux += ray1 * pr1 + mie1 * pm1;
	//}

	//else {
	//    flux += += texture4D(deltaSRSampler, r, w.z, muS, nu1).rgb;
	//}

	// light coming from direction w and scattered in direction v
	// = light arriving at x from direction w (raymie1) * SUM(scattering coefficient * phaseFunction) see Eq (7)
	flux += photon_energy * (betaR * exp(-(r - Rg) / HR) * pr2 + betaMSca * exp(-(r - Rg) / HM) * pm2);//* rec_atten_Kd;
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


rtDeclareVariable(float, stepSizeP, ,);

__device__ __inline__ void get_light_info(float3 ffnormal, float& scale, float& n_dot_l, float& light_dist)
{
	float3 point_on_light = light.anchor;
	float dist_scale = 1.0;
	if( light.is_area_light == 2) { //distant light
		uint2  seed2   = image_rnd_seeds[launch_index];
		float2 sample = make_float2( rnd( seed2.x ), rnd( seed2.y ) ); 
		image_rnd_seeds[launch_index] = seed2;
		point_on_light = light.anchor + sample.x*light.v1 + sample.y*light.v2; 
		dist_scale = 100.0f*light.worldRadius / ( M_PIf * 0.5f);//10000.0f;

		//point_on_light = light.position;
		//dist_scale = light.worldRadius / ( M_PIf * 0.5f);
	}
	float3 to_light    = point_on_light;// - rec_position;
	light_dist  = length( to_light );
	to_light = to_light / light_dist;
	scale =1.0 / (dist_scale*light_dist*light_dist);
	n_dot_l     = fmaxf( 0.0f, dot( ffnormal, to_light ) );
} 



__device__ __inline__ float limit(float r, float mu) {
	float dout = -r * mu + sqrtf(r * r * (mu * mu - 1.0) + RL * RL);
	float delta2 = r * r * (mu * mu - 1.0) + Rg * Rg;
	if (delta2 >= 0.0) {
		float din = -r * mu - sqrtf(delta2);
		if (din >= 0.0) {
			dout = min(dout, din);
		}
	}
	return dout;
}

//ground radiance at end of ray x+tv, when sun in direction s
//attenuated bewteen ground and viewer (=R[L0]+R[L*])
__device__ __inline__  float3 groundColor(float3 x, float t, float3 v, float3 s, float sunPower, float r, float mu, float3 attenuation)
{
	float3 result;
	float3 direction    = ray.direction;
	float3 origin       = ray.origin;
	float3 hit_point    = origin + t*direction;



	if (t > 0.0) { // if ray hits ground surface
		// ground reflectance at end of ray, x0
		float3 x0 = x + t * v;
		float r0 = length(x0);
		float3 n = x0 / r0;
		//vec2 coords = vec2(atan(n.y, n.x), acos(n.z)) * vec2(0.5, 1.0) / M_PI + vec2(0.5, 0.0);
		float2 coords = make_float2(atan2(n.y, n.x), acos(n.z)) * make_float2(0.5, 1.0) / M_PI + make_float2(0.5, 0.0);
		//float2 coords = make_float2(atan2(n.x, n.z), acos(n.y)) * make_float2(0.5, 1.0) / M_PI + make_float2(0.5, 0.0);
		//float4 reflectance = tex2D(earth_texture, texcoord.x, texcoord.y) * make_float4(0.2, 0.2, 0.2, 1.0);

		float4 reflectance = make_float4(0.0, 0.0, 0.0, 0.0);
		if(length(origin) < Rt){
			hit_prd.ray_vol_depth++;
			optix::Ray ray( hit_point, direction, rtpass_ray_type, scene_epsilon );
			if(bruneton_single == 1)
				rtTrace( single_top_object, ray, hit_prd );
			else
				rtTrace( top_object, ray, hit_prd );
			hit_prd.ray_vol_depth--;

			HitRecord& rec = rtpass_output_buffer[launch_index];
			reflectance = make_float4(rec.attenuated_Kd.x,rec.attenuated_Kd.y,rec.attenuated_Kd.z,1.0);

		}
		else{
			reflectance = tex2D(earth_texture, coords.x, coords.y) * make_float4(0.2, 0.2, 0.2, 1.0);
		}

		if (r0 > Rg + 0.01) {
			reflectance = make_float4(0.4, 0.4, 0.4, 0.0);
		}


		// direct sun light (radiance) reaching x0
		float muS = dot(n, s);
		float3 sunLight = transmittanceWithShadow(r0, muS, trans_texture, Rg, Rt);//make_float3(0.0,0.2,0.0);
		//if(sunLight.x > 0.0 || sunLight.y > 0.0)
		//	rtPrintf("sunLight.x > 0.0 || sunLight.y > 0.0\n");

		// precomputed sky light (irradiance) (=E[L*]) at x0
		// ERROR: black circle gets introduced somewhere here
		//vec3 groundSkyLight = irradiance(irradianceSampler, r0, muS);

		// light reflected at x0 (=(R[L0]+R[L*])/T(x,x0))
		float3 groundColor = make_float3(reflectance.x,reflectance.y,reflectance.z) * (max(muS, 0.0) * sunLight /*+ groundSkyLight*/) * sunPower / M_PI;

        
		// water specular color due to sunLight
		if (reflectance.w > 0.0) {
			float3 h = normalize(s - v);
			float fresnel = 0.02 + 0.98 * pow(1.0 - dot(-v, h), 5.0);
			float waterBrdf = fresnel * pow(max(dot(h, n), 0.0), 150.0);
			groundColor += reflectance.w * max(waterBrdf, 0.0) * sunLight * sunPower;
		}
		//return groundColor;

		result = attenuation * groundColor; //=R[L0]+R[L*]
	} 
	else { // ray looking at the sky
		result = make_float3(0.0);
	}
	return result;
}



/*
__device__ __inline__ float3  inscatterN(float r, float mu, float muS, float nu, float3 x, float3 v) {
float3 raymie = make_float3(0.0);
float dx = limit(r, mu) / float(INSCATTER_INTEGRAL_SAMPLES);
float3 x_old = x;
float3 x_new = x;
float3 raymiei = integrand(r, mu, muS, nu, 0.0, x, v);
float3 Tr = make_float3(1.0);
for (int i = 1; i <= INSCATTER_INTEGRAL_SAMPLES; ++i) {
float xj = float(i) * dx;
x_new = v*dx + x_old;
Tr *= sun_atten(x_old, x_new, light.direction, v);
float3 raymiej = integrand(r, mu, muS, nu, xj, x_new, v);
//raymie += (raymiei + raymiej) / 2.0 * dx;
raymie += (raymiei + raymiej) / 2.0 * dx + Tr * raymie;
//xi = xj;
raymiei = raymiej;
x_old = x_new;
}

float4 Li4 = tex4D_inscatter(inscatterSampler,r, mu, muS, nu);
float3 Li =  make_float3(Li4.x,Li4.y,Li4.z) * transmittance_RMuD(r, mu, float(INSCATTER_INTEGRAL_SAMPLES) * dx);
if(singleOnly == 1.0)
return Li;
if(singleOnly == 2.0)
return raymie;
return raymie + Li;
}
*/
__device__ __inline__ float4 test_inscatter(float3 x, float3 v, float3 s, 
											float r, float mu, float muS, float nu, bool bSingleOnly)
{
	//if(jm_mult == 0)
	//	return make_float4(0.0);
	//float r = length(x);
	//mu = dot(x, v) / r;
	//nu = dot(v, s);
	//muS = dot(x, s) / r;
	if(bSingleOnly){
		if(length(x) < 6380){
			//rtPrintf("test_inscatter: X:(%f,%f,%f), v:(%f,%f,%f), s:(%f,%f,%f)\n", x.x,x.y,x.z,v.x,v.y,v.z,s.x,s.y,s.z);
			//rtPrintf("test_inscatter: r:(%f), mu:(%f), muS:(%f), nu:(%f)\n", r,mu,muS,nu);
		}
		if(bruneton_single == 1 && jm_mult == 0)
			return tex4D_inscatter(inscatterSampler,r, mu, muS, nu);
		else
			return tex4D_inscatter(inscatterPhotonSampler,r, mu, muS, nu);
	}
	float3 Lii = make_float3(0.0);//inscatterN(r, mu, muS, nu, x, v);
	return make_float4(Lii.x, Lii.y, Lii.z ,0.0);
}

//inscattered light along ray x+tv, when sun in direction s (=S[L]-T(x,x0)S[L]|x0)
__device__ __inline__ float3  inscatter(float3 x, float t, float3 v, float3 s, float sunPower, float& r, float& mu, float3& attenuation,  bool bSingleOnly) 
{
	float3 result;
	r = length(x);
	mu = dot(x, v) / r;
	float d = 0.0;
	if(r * r * (mu * mu - 1.0) + Rt * Rt >= 0.0)
		d = -r * mu - sqrtf(r * r * (mu * mu - 1.0) + Rt * Rt);


	result = make_float3(1.0,0.0,0.0);
	if (d > 0.0) { // if x in space and ray intersects atmosphere
		// move x to nearest intersection of ray with top atmosphere boundary
		//rtPrintf("inscatterf: d = %f, mu = %f\n", d, mu);
		x += d * v;
		t -= d;
		mu = (r * mu + d) / Rt;
		r = Rt;
		//result = make_float3(1.0,0.0,1.0);//pink
	}

	if(r > Rt){	//TODO: this is a temporary fix for cases when the ray is in space 
		//and the ray appears to be looking into space when in fact is intersecting atmosphere.
		r = Rt;
	}

	if (r <= Rt) { // if ray intersects atmosphere
		float nu = dot(v, s);
		float muS = dot(x, s) / r;
		float phaseR = phaseFunctionR(nu);
		float phaseM = phaseFunctionM(nu, mieG);
		float4 inscatterf = max(test_inscatter(x,v,s, r, mu, muS, nu, bSingleOnly), make_float4(0.0));
		//result = make_float3(1.0,0.0,0.0);
		//rtPrintf("inscatterf(1): r=%f, mu=%f, mS=%f, nu=%f, -> (%f,%f,%f)\n",r, mu, muS,nu,inscatterf.x,inscatterf.y,inscatterf.z);
		//rtPrintf("inscatter r: %f, t: %f\n",r, t);
		attenuation = analyticTransmittance(r, mu, t, HR, HM, Rg, betaR, betaMEx);
		if (t > 0.0) {
			//result = make_float3(1.0,0.0,0.0);
			//return result*sunPower;
			//result = make_float3(0.0,1.0,0.0);
			float3 x0 = x + t * v;
			float r0 = length(x0);
			float rMu0 = dot(x0, v);
			float mu0 = rMu0 / r0;
			float muS0 = dot(x0, s) / r0;
#ifdef FIX
			// avoids imprecision problems in transmittance computations based on textures
			//attenuation = analyticTransmittance(r, mu, t);
			attenuation = analyticTransmittance(r, mu, t, HR, HM, Rg, betaR, betaMEx);
#else
			attenuation = tranAtmNew(r, mu, trans_texture, x0, v,  Rg, Rt ,RL, betaR, betaMEx,
				HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
			/*
			attenuation = tranAtm(r, mu,
			x0, v,
			Rg, Rt, RL,
			betaR, betaMEx,
			HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
			*/

			//attenuation = analyticTransmittance(r, mu, t, HR, HM, Rg, betaR, betaMEx);
#endif
			if (r0 > Rg + 0.01) {
				// computes S[L]-T(x,x0)S[L]|x0
				//rtPrintf("inscatterf: %f,%f,%f\n",inscatterf.x,inscatterf.y,inscatterf.z);
				inscatterf = max(inscatterf - make_float4(attenuation, attenuation.x) * test_inscatter(x,v,s, r0, mu0, muS0, nu, bSingleOnly), make_float4(0.0));
#ifdef FIX
				// avoids imprecision problems near horizon by interpolating between two points above and below horizon
				const float EPS = 0.004;
				float muHoriz = -sqrtf(1.0 - (Rg / r) * (Rg / r));
				if (abs(mu - muHoriz) < EPS) {
					float a = ((mu - muHoriz) + EPS) / (2.0 * EPS);

					mu = muHoriz - EPS;
					r0 = sqrtf(r * r + t * t + 2.0 * r * t * mu);
					mu0 = (r * mu + t) / r0;
					float4 inScatter0 = tex4D_inscatter(inscatterSampler,r, mu, muS, nu);
					float4 inScatter1 = tex4D_inscatter(inscatterSampler,r0, mu0, muS0, nu);
					float4 att4 = make_float4(attenuation.x, attenuation.y, attenuation.z, attenuation.x);
					float4 inScatterA = max(inScatter0 - att4 * inScatter1, make_float4(0.0));

					mu = muHoriz + EPS;
					r0 = sqrtf(r * r + t * t + 2.0 * r * t * mu);
					mu0 = (r * mu + t) / r0;
					inScatter0 = tex4D_inscatter(inscatterSampler,r, mu, muS, nu);
					inScatter1 = tex4D_inscatter(inscatterSampler,r0, mu0, muS0, nu);
					att4 = make_float4(attenuation.x, attenuation.y, attenuation.z, attenuation.x);
					float4 inScatterB = max(inScatter0 - att4 * inScatter1, make_float4(0.0));

					inscatterf = lerp(inScatterA, inScatterB, a);
				}
#endif
			}
		}
		else{
			//rtPrintf("t: %f\n",t);
			//result = make_float3(1.0,0.0,0.0);
			//return result*sunPower;
			//inscatterf = make_float4(0.0,1.0,0.0,1.0);
		}
#ifdef FIX
		// avoids imprecision problems in Mie scattering when sun is below horizon
		inscatterf.w *= smoothstep(0.00, 0.02, muS);
#endif
		//result = max(make_float3(inscatterf.x,inscatterf.y,inscatterf.z) * phaseR + getMie(inscatterf, betaR) * phaseM, make_float3(0.0));
		//result = max(getMie(inscatterf, betaR) * phaseM, make_float3(0.0));
		result = max(make_float3(inscatterf.x,inscatterf.y,inscatterf.z) * phaseR,make_float3(0.0,0.0,0.0));
		//result = make_float3(0.0,1.0,0.0);

	} else { // x in space and ray looking in space
		//float muS = dot(x, s) / r;
		//rtPrintf("r: %f\n",r);
		//result = make_float3(1.0,0.0,0.0);
		//if(fabs(muS) < 0.1)
		//	result = make_float3(1.0,0.0,1.0);//pink

	}

	//rtPrintf("result: %f,%f,%f\n",
	return result*sunPower;// * ISun;
}

__device__ __inline__ void earth()
{
	float3 direction    = ray.direction;
	float3 origin       = ray.origin;
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal     = faceforward( world_shading_normal, -direction, world_geometric_normal );
	float3 hit_point    = origin + t_hit*direction;
	float3 Ld = make_float3(0.0);
	float3 Li = make_float3(0.0);
	float3 L_all = make_float3(0.0);
	unsigned int seed = rot_seed( image_rnd_seeds[ launch_index ].x, frame_number );

	size_t2 size     = rtpass_output_buffer.size();

	if(hit_prd.ray_vol_depth == 0){
		// Compute  scale
		float  n_dot_l = 1.0;
		float scale = 1.0;
		float light_dist = 0.0;
		get_light_info(ffnormal, scale, n_dot_l, light_dist);

		hit_prd.ray_vol_depth++;
		optix::Ray ray( hit_point, direction, rtpass_ray_type, scene_epsilon );
		if(bruneton_single == 1)
			rtTrace( single_top_object, ray, hit_prd );
		else
			rtTrace( top_object, ray, hit_prd );
		hit_prd.ray_vol_depth--;

		HitRecord& recp = rtpass_output_buffer[launch_index];

		float dist = stepSizeP*rnd( seed) ;

		float3 Tr = make_float3(1.0);
		float lenToX0 = hit_prd.prev_ray_length;
		float3 dest_point = hit_point + lenToX0*direction;
		//float3 TXtoX0 = light_atten(hit_point, dest_point, direction); 
		float3 TXtoX0 = sun_atten(hit_point, dest_point, -light.direction, direction)*10.0;


		// first point
		float3 x = origin + t_hit*direction;
		if(length(origin) < Rt && length(origin) >= Rg){
			x = origin;
			//if(pm_index >= 260965 && pm_index <= 262986) {
			//	rtPrintf("pm_index %d: r = (%f) \n", pm_index, length(origin));
			//} 
		}
		//if(length(origin) < 6419 ){//(pm_index >= 260965 && pm_index <= 262986) {
		//	rtPrintf("pm_index %d: r1 = (%f) \n", pm_index, length(origin)); 
		//}

		// view direction
		float3 v = normalize(direction);



		HitRecord& rec = rtpass_output_buffer[launch_index];  
		rec.normal = ffnormal;//surf_normal;
		rec.flags = PPM_HIT_VOLUME;

		rec.radius2 = rtpass_default_radius2;
		rec.photon_count = 0;
		rec.accum_atten = 0.0f;
		rec.flux = make_float3(0.0f, 0.0f, 0.0f);


		//rec.attenuated_Kd =  /*(TXtoX0*attn) + */ L_all/(numSteps)/100.0;//*scale;
		//rec.attenuated_Kd =  /*(TXtoX0*attn) + */ L_all/(numSteps)/100000000.0;//*scale;
		//rec.attenuated_Kd =  ((TXtoX0*attn) + L_all/(numSteps)*scale);
		//rec.attenuated_Kd =  (TXtoX0*attn + L_all*100/(numSteps));
		//rec.attenuated_Kd =  (TXtoX0*attn + L_all/(numSteps*5000));
		//rec.attenuated_Kd = make_float3(tex2D(trans_texture, launch_index.x, launch_index.y))/10;


		//float3 x = origin + t_hit*direction;
		//float3 v = normalize(direction);
		float3 temp = make_float3(1.0);

		float r = length(x);
		float mu = dot(x, v) / r;
		float t = 0.0;

		if((r * r * (mu * mu - 1.0) + Rg * Rg) >= 0.0){
			t =  -r * mu - sqrtf(r * r * (mu * mu - 1.0) + Rg * Rg);
			//rec.attenuated_Kd = make_float3(1.0,0.0,0.0);
			//t = -r * mu - 100000000.0;
			//t = -t_hit;
			//rtPrintf("ray length: (t_hit)%f, %f: %f\n",t, t_pac);
		}
		/*
		float3 g = x - make_float3(0.0, 0.0, Rg + 10.0);
		float a = v.x * v.x + v.y * v.y - v.z * v.z;
		float b = 2.0 * (g.x * v.x + g.y * v.y - g.z * v.z);
		float c = g.x * g.x + g.y * g.y - g.z * g.z;
		float d = -(b + sqrtf(b * b - 4.0 * a * c)) / (2.0 * a);
		bool cone = d > 0.0 && fabs(x.z + d * v.z - Rg) <= 10.0;

		if (t > 0.0) {
		if (cone && d < t) {
		t = d;
		}
		} else if (cone) {
		t = d;
		}
		*/

		rec.attenuated_Kd = make_float3(0.0);
		float muS = dot(x, light.direction) / r;
		float nu = dot(v, light.direction);
		if(length(origin) < Rt){
			//t = -t_hit;
			//rec.attenuated_Kd = make_float3(0.0,0.5,0.0);
			//rtPrintf("ray length: (r)%f (mu)%f\n", r, mu);
			//float4 inscatter_res = tex4D_inscatter(inscatterSampler, 6360, mu, 0.07, 0.79);
			//rec.attenuated_Kd  += make_float3(inscatter_res.x,inscatter_res.y,inscatter_res.z);
			//rec.attenuated_Kd  +=  make_float3(muS,nu,1.0);
			//if(mu <= -0.0006 && mu > -0.0010)
			//	rec.attenuated_Kd  += make_float3(0.0,1.0,0.0);
			//else 
			rec.attenuated_Kd  += inscatter(x, t, v, light.direction, 100.0, r, mu, temp, true);
			//rec.attenuated_Kd  +=  groundColor(x, t, v, light.direction, 100.0, r, mu, temp); //R[L0]+R[L*]
			//rec.attenuated_Kd  +=  transmittanceR_MU(r, mu);
		}
		else{
			rec.attenuated_Kd = make_float3(0.0);
			//rtPrintf("--->ray length: (X)%f, (pac t)%f\n", length(origin), t);
			//rec.attenuated_Kd  += TXtoX0*attn; 
			//rec.attenuated_Kd  += tex4D_inscatter(inscatterSampler,r, mu, muS, nu);
			//rec.attenuated_Kd  += inscatter(x, t, v, light.direction, 100.0, r, mu, temp, true);
			rec.attenuated_Kd  +=  groundColor(x, t, v, light.direction, 100.0, r, mu, temp); //R[L0]+R[L*]
			//rec.attenuated_Kd  +=  transmittanceR_MU(r, mu);//make_float3(muS,nu,1.0);transmittanceR_MU(r, mu); 

			//rec.attenuated_Kd  += L_all/(numSteps)/100.0;

			//float nu = dot(v, light.direction);
			//float muS = dot(x, light.direction) / r;
			//float4 single = test_inscatter(x,v,light.direction, r, mu, muS, nu, true);
			//rec.attenuated_Kd = make_float3(single.x,single.y,single.z);
		}
		//if(muS > 0.95 ){
		//	rec.attenuated_Kd = make_float3(1.0,1.0,0.0)*10 +  groundColor(x, t, v, light.direction, 100.0, r, mu, temp);
		//}
		//rtpass_output_buffer[launch_index] = rec;
	}
	else{
		hit_prd.ray_vol_depth++;
		optix::Ray ray( hit_point, direction, rtpass_ray_type, scene_epsilon );
		if(bruneton_single == 1)
			rtTrace( single_top_object, ray, hit_prd );
		else
			rtTrace( top_object, ray, hit_prd );
		hit_prd.ray_vol_depth--;
		hit_prd.prev_ray_length = t_hit;
		return;
	}
} 
RT_PROGRAM void rtpass_vol_closest_hit()
{
	earth();
}

__device__ __inline__ float pow5( float x )
{
	float t = x*x;
	return t*t*x;
}

RT_PROGRAM void shadow_any_hit()
{
	prd_shadow.prev_ray_length = t_hit;

	//what happens if we are inside a participating medium?
	prd_shadow.attenuation = make_float3(0.0f);
	prd_shadow.inShadow = true;
	//rtIgnoreIntersection();
	rtTerminateRay();

	/*
	// no direct shadow from light sources
	if (fmaxf(emissive) == 0.0f)
	{
	prd_shadow.inShadow = true;
	rtTerminateRay();
	}
	*/
}

__device__ __inline__ float3 inscatter_x( float3 x, float mu )
{
	float3 result = make_float3(0.0);
	float3 temp = make_float3(1.0);
	float3 direction    = ray.direction;

	// view direction
	float3 v = normalize(direction);

	float r = length(x);
	//mu = dot(x, v) / r;
	float t = 0.0;

	if((r * r * (mu * mu - 1.0) + Rg * Rg) >= 0.0){
		t =  -r * mu - sqrtf(r * r * (mu * mu - 1.0) + Rg * Rg);
	}

	float3 g = x - make_float3(0.0, 0.0, Rg + 10.0);
	float a = v.x * v.x + v.y * v.y - v.z * v.z;
	float b = 2.0 * (g.x * v.x + g.y * v.y - g.z * v.z);
	float c = g.x * g.x + g.y * g.y - g.z * g.z;
	float d = -(b + sqrtf(b * b - 4.0 * a * c)) / (2.0 * a);
	bool cone = d > 0.0 && fabs(x.z + d * v.z - Rg) <= 10.0;

	if (t > 0.0) {
		if (cone && d < t) {
			t = d;
		}
	} else if (cone) {
		t = d;
	}

	float muS = dot(x, light.direction) / r;
	float nu = dot(v, light.direction);

	result = inscatter(x, t, v, light.direction, 100.0, r, mu, temp, true);
	return result;	

}

__device__ __inline__ float3 arial_perspective(float3 origin, float3 hit_point, float3 direction)
{
	float3 result = make_float3(0.0);
	float r = length(hit_point);
	float rx = length(origin);
	if(rx <=Rt){
		float dHx = sqrtf(Rt*Rt-rx*rx);
		float dH = sqrtf(Rt*Rt-r*r);
		float d = length(origin-origin);

		normalize(direction);
		float mux = dot(origin, direction) / rx;
		float d0x = 0.0;
		if((rx * rx * (mux * mux - 1.0) + Rg * Rg) >= 0.0){
			d0x =  -rx * mux - sqrtf(rx * rx * (mux * mux - 1.0) + Rg * Rg);
		}
		float d0 = d0x - d;

		float muX = d0x/dHx; 
		float muXS = d0/dH;

		result = inscatter_x( origin, muX ) - tranAtm(	length(origin), muXS, hit_point, 
			direction, Rg, Rt, RL, 
			betaR, betaMEx, HR, HM, 
			TRANSMITTANCE_INTEGRAL_SAMPLES) * ( inscatter_x( hit_point, muXS )); 
		//result = inscatter_x( origin, muX ) - tranAtm(	length(origin), muXS, trans_texture, hit_point, 
		//								direction, Rg, Rt, RL, 
		//								betaR, betaMEx, HR, HM, 
		//								TRANSMITTANCE_INTEGRAL_SAMPLES) * ( inscatter_x( hit_point, muXS )); 
	}
	return result;
}

RT_PROGRAM void rtpass_mesh_closest_hit()
{
	float3 direction    = ray.direction;
	float3 origin       = ray.origin;
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal     = faceforward( world_shading_normal, -direction, world_geometric_normal );
	float3 hit_point    = origin + t_hit*direction;
	float2 uv                     = make_float2(texcoord);
	float3 Kd = make_float3(tex2D(diffuse_map, uv.x, uv.y));

	//rtPrintf("rtpass_mesh_closest_hit: t_hit = %f, vol_depth=%d\n",t_hit, hit_prd.ray_vol_depth);
	HitRecord rec;

	rec.attenuated_Kd = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
	rec.attenuated_Kd *= Kd;
	// direct sun light (radiance) reaching x0
	float muS = dot(world_shading_normal, direction);
	float3 sunLight = make_float3(1.0);//transmittanceWithShadow(length(hit_point), muS, trans_texture, Rg, Rt);//make_float3(0.0,0.2,0.0);
	//rec.attenuated_Kd = rec.attenuated_Kd * (max(muS, 0.0) * sunLight) * sunPower;// / M_PI;


	// arial perspective
	rec.attenuated_Kd += arial_perspective(origin, hit_point, direction);

	rec.position = hit_point; 
	rec.normal = ffnormal;
	//hit_prd.L0 = rec.attenuated_Kd;
	hit_prd.prev_ray_length = t_hit;
	rec.flags = PPM_HIT;

	rec.radius2 = rtpass_default_radius2;
	rec.photon_count = 0;
	rec.accum_atten = 0.0f;
	rec.flux = make_float3(0.0f, 0.0f, 0.0f);
	rec.t_hit_exitvolume = 1;
	rtpass_output_buffer[launch_index] = rec;
}



// returns R bin position radius for the lower bound of the bin position
// Note: R bin position is always LOWER bound
__device__ __inline__ float RBin2Radius(int rbinid)
{
	float rt_rg = (Rt-Rg);
	float rad = Rg + rt_rg * (float)rbinid/(float)(RES_R_BIN-1);
	return rad;
}
/*
// return R bin position (lower bound) if bDown = true
// return upper R bin position if bDown = false 
__device__ __inline__ int findRBin(optix::float3 new_hit_point, bool bDown)
{
float len_r = sqrtf(dot(new_hit_point, new_hit_point));
if(len_r < Rg)
len_r = Rg;

float rt_rg = (Rt-Rg);
float r_rg = (len_r-Rg);

int ind = 0;
int ind_floor = floor((float)(r_rg/rt_rg)*(int)(RES_R_BIN-1)); //or /(RES)
int ind_ceil = ceil((float)(r_rg/rt_rg)*(int)(RES_R_BIN-1));
if(ind_floor == ind_ceil)
{
// case if new_hit_point is right on the R bin boundary
if(bDown)
ind = (ind_floor - 1) > 0 ? ind_floor -1  : ind_floor;
else  
ind = (ind_ceil + 1) < (RES_R_BIN-1) ?  ind_ceil + 1 : ind_ceil;
if(ind < 0)
ind  = 0;
if(ind > RES_R_BIN-1)
ind = RES_R_BIN -1;
}
else{
// case if new_hit_point is between R bin boundaries
if(bDown)
ind = ind_floor;
else  
ind = ind_ceil;

}
return ind;
}
*/


// collect radiance between first and last point,
// type : 1 - down, 2 -up, 3 - changing direction
// d    : distance from first to last
// mu   : cos (first point and viewing ray)


__device__ __inline__ float3 photon_gather(float3 first, float3 last, float d, float mu_first, int type ) 
{
	float3 direction    = ray.direction;
	float3 origin       = first;
	float3 hit_point    = last;

	// mid_point determines R position bin id
	float3 mid_point = (hit_point + origin)/2.0;
	float r_origin = sqrtf(dot(origin, origin));
	float r_hit_point = sqrtf(dot(hit_point, hit_point));
	float r_mid_point = sqrtf(dot(mid_point, mid_point));
	float3 result = make_float3(0.0);

	int muS_bound=findPosBin(mid_point, light.direction, RES_MU_S_BIN);
	int r_ind = -1;//findRBin(mid_point, true);
	float3 norm_point = normalize(mid_point);
	float muS = fabsf(dot(light.direction,norm_point)); //ONLY store cos values from 0 to 1
	float mu = dot(direction, norm_point);
	float nu = dot(direction, light.direction);

	int mu_ind = (int)(mu*RES_MU);
	int nu_ind = (int)(nu*RES_NU);
	float3 attenuation = make_float3(1.0);
	float volume = 1.0;
	if(r_mid_point < Rt){
		volume = 0;//posbin_volume(mid_point, light.direction, Rt, Rg, RES_R_BIN, RES_MU_S_BIN);
		uint3 ind;
		ind.x = r_ind;
		ind.y = muS_bound*nu_ind;
		ind.z = mu_ind;
		result = pbin_output_buffer[ind];
		float r1 = sqrtf(dot(first, first));
		float3 norm_first = normalize(first);
		float mu1 = dot(direction, norm_first);
		attenuation = analyticTransmittance(r1, mu, d, HR, HM, Rg, betaR, betaMEx);
	}
	//if(total_in_bin > 0.0)
	//	rtPrintf("(%d,%d) res(%f,%f,%f) atten(%f,%f,%f) %f\n",muS_bound,r_ind, result.x,result.y,result.z,attenuation.x,attenuation.y,attenuation.z,volume);	
	result = result*(1.0-attenuation)/volume;
	//float sum = result.x + result.y + result.z;

	return result;
}

/*
//calculates radiants contribution at each bin as viewing ray passes through it
__device__ __inline__ float3 bincount_mult_scatterExt(float raylen, float3 hit_point, float3 point, float3 direction)
{
size_t2 size     = rtpass_output_buffer.size();

float r_dest = sqrtf(dot(hit_point, hit_point));

float r = sqrtf(dot(point, point));
float3 sector_atten = make_float3(0.0);
float3 test_point = make_float3(0.0);
float3 result_atten = make_float3(0.0);
float3 old_point = make_float3(0.0);

float mu = dot(point, direction)/r;
float3 transm = tranAtm(r, mu,
point-direction*hit_prd.total_dist, direction,
Rg, Rt, RL,
betaR, betaMEx,
HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
result_atten =  make_float3(0.01,0.0001,0.0)*transm;//photon_gather(point, hit_point, raylen, mu, type)*transm;
return result_atten;
}
*/
//calculates radiants contribution at each bin as viewing ray passes through it
__device__ __inline__ float3 bincount_mult_scatter(float raylen, float3 hit_point, float3 point, float3 direction)
{
	float t_hit_total = 0.0;
	bool bDone = false;
	int count = 0;
	size_t2 size     = rtpass_output_buffer.size();

	float r_dest = sqrtf(dot(hit_point, hit_point));

	float r = sqrtf(dot(point, point));
	float o2d_step =  raylen / 4.0;
	float3 sector_atten = make_float3(0.0);
	float test_step = (float)(Rt-Rg)/100.0;
	float3 test_point = make_float3(0.0);
	int type = 0;
	float r_test = 0.0;
	float3 result_atten = make_float3(0.0);
	float sum = 0.0;
	float3 old_point = make_float3(0.0);
	while( !bDone && count < 20){
		// determine if viewing ray is going down the atmosphere or up?
		r = sqrtf(dot(point, point));
		//if(r < Rg)
		//	break;
		test_point = point + direction*test_step;
		r_test = sqrtf(dot(test_point, test_point));
		type = 0;

		int rbinid = -1;//findRBin(test_point, true);
		float R_current = RBin2Radius(rbinid);
		if(r_test < r) {  //viewing ray is going DOWN
			type = 1;
		}
		else if(r_test > r) { // viewing ray is going UP
			type = 2;
		}
		else if( r_test == r) { //viewing ray will change direction
			type = 3;
		}



		float mu = dot(point, direction)/r;
		float d = -1; //by deafult there is no intersection
		if((r * r * (mu * mu - 1.0) + R_current * R_current) >= 0.0){
			d =  -r * mu - sqrtf(r * r * (mu * mu - 1.0) + R_current * R_current);
		}

		// if solution does not exist
		if(d < 0){
			//d = -1; //by deafult there is no intersection
			R_current = RBin2Radius(rbinid+1);
			if((r * r * (mu * mu - 1.0) + R_current * R_current) >= 0.0){
				d =  -r * mu - sqrtf(r * r * (mu * mu - 1.0) + R_current * R_current);
			}
		}

		// PROBLEM!
		// strange case when we keep traversing the boundary of the same sphere
		// might have something to do with numerical impresision or a bug in our code
		if( d <= 0.5){
			d = o2d_step;
		}

		//if(pm_index == 21901){//type == 2){//pm_index == 24180) {
		//	rtPrintf("pm_index %d:  rbinid = %d count =%d \n", pm_index, rbinid, count);
		//rtPrintf("pm_index %d:  R_current = (%f) rbinid = %d count =%d d = %f\n", pm_index, R_current, rbinid, count, d); 
		//rtPrintf("pm_index %d:  r_dest = %f, r = %f, raylen = %f, t_hit_total = %f, type = %d, mu = %f\n", pm_index, r_dest, r, raylen, t_hit_total, type, mu); 
		//}

		// gather photons inside R position bin --> [rbinid..rbinid+1] and 	
		t_hit_total += d;
		old_point = point;
		if(t_hit_total >= raylen){
			point = hit_point;
			bDone = true;
		}
		else{
			point += d*direction;
		}

		result_atten =  photon_gather(old_point, point, d, mu, type);
		sum = result_atten.x +  result_atten.y + result_atten.z;
		if(sum > 0.0001){
			//rtPrintf("pm_index %d: (count %d) raylen =%f, t_hit_total=%f \n", pm_index, count, raylen, t_hit_total);
			//rtPrintf("result_atten = %f, %f, %f, count = %d\n", result_atten.x, result_atten.y, result_atten.z, count);// result_atten.y, result_atten.z);
		}
		sector_atten += result_atten;
		count++;
	}
	return sector_atten;
}

// calculate single scattering component starting at point x
__device__ __inline__ float3 prepare_ss_ext(float r1, float mu1, float d, float3 x, float3 v, float3 s, float sunPower, bool doAttenuation)
{
	float r = length(x);
	float mu = dot(x, v) / r;
	float t = 0.0;

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

	float muS = dot(x, s) / r;
	float nu = dot(v, s);
	float3 temp = analyticTransmittance(r1, mu1, d, HR, HM, Rg, betaR, betaMEx);//make_float3(1.0);
	float3 result = inscatter(x, t, v, s, sunPower, r, mu, temp, true);
	if(doAttenuation)
		return result*temp;
	else
		return result;
}

// calculate single scattering component starting at point x
__device__ __inline__ float3 prepare_ss(float3 x, float3 v, float3 s, float sunPower, bool doAttenuation)
{
	float r = length(x);
	float mu = dot(x, v) / r;
	float t = 0.0;

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

	float muS = dot(x, s) / r;
	float nu = dot(v, s);
	float3 temp = analyticTransmittance(length(x), mu, t, HR, HM, Rg, betaR, betaMEx);//make_float3(1.0);
	float3 result = inscatter(x, t, v, s, sunPower, r, mu, temp, true);
	if(doAttenuation)
		return result*temp;
	else
		return result;
}


RT_PROGRAM void transit_closest_hit_ext()
{
	if(jm_mult == 1){
		earth();
		return;
	}

	HitRecord& rec = rtpass_output_buffer[launch_index];  
	float3 direction    = ray.direction;
	float3 origin       = ray.origin;
	float3 hit_point    = origin + t_hit*direction;
	size_t2 size     = rtpass_output_buffer.size();
	uint    pm_index = (launch_index.y * size.x + launch_index.x);


	// first point
	float3 x = origin + t_hit*direction;
	if(length(origin) <= Rt && length(origin) >= Rg){
		x = origin;
	}
	// view direction
	float3 v = normalize(direction);

	rec.flags = PPM_HIT_VOLUME;

	rec.radius2 = rtpass_default_radius2;
	rec.photon_count = 0;
	rec.accum_atten = 0.0f;
	rec.flux = make_float3(0.0f, 0.0f, 0.0f);

	float r = length(x);
	float mu = dot(x, v) / r;
	float t = 0.0;//-r * mu;

	if((r * r * (mu * mu - 1.0) + Rg * Rg) >= 0.0){
		t =  -r * mu - sqrtf(r * r * (mu * mu - 1.0) + Rg * Rg);
	}
	//float3 temptran = atten_for_ground(x, t, r, mu, v);
	//float3 ground = groundColor(x, t, v, light.direction, 100.0, r, mu, temptran); //R[L0]+R[L*]
	rec.attenuated_Kd = make_float3(0.0);


	float muS = dot(x, light.direction) / r;
	float nu = dot(v, light.direction);
	float3 temp = make_float3(1.0);
	int count = 0;
	float arr[24];
	bool bt_hit = false;
	float total_dist = 0.0;

	if(/*length(origin) < Rt ||*/ muS < 0.0){
		rec.attenuated_Kd = make_float3(0.0,0.0,0.0);
		return;
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
		//	rec.attenuated_Kd = make_float3(10.0,0.0,0.0);
		//	return;
		//}

		int layer;
		float final_dist = 0.0;
		if(length(origin) >= Rt){
			sphere_planet_atm_boundary_dist(pm_index, light.direction, p, v, layer, t, true, Rg, Rt, RES_R_BIN);
			si_dir = sphere_planet_atm_boundary_dist_ext(pm_index, light.direction, p, v, t,  si, next_t_hit_s, 
				true, Rg, Rt, RES_R_BIN); //bSpace = true because we are in space
			final_dist = t;
		}
		else{
			sphere_planet_atm_boundary_dist(pm_index, light.direction, p, v, layer, t, true, Rg, Rt, RES_R_BIN);
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
				rec.attenuated_Kd += next_t_hit_s*photon_gather_contib_ext(pm_index, p, p+dist*v, v, index, num_photons, ci_temp, si);
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
			rec.attenuated_Kd = make_float3(1.0,0.0,0.0);
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

					rec.attenuated_Kd  = make_float3(0.0,0.0,1.0);
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
					//rec.attenuated_Kd  = make_float3(1.0,0.0,1.0);
					//return;	
					}
					else{
					rtPrintf(">cone>>pm_index %d, ci:%d, ci_dir:%d, si:%d, si_dir:%d, <count:%d>, next_t_hit_c:%f, next_t_hit_s:%f, total_dist:%f, final_dist:%f\n",
					pm_index, ci, ci_dir, si, si_dir,
					count, next_t_hit_c, next_t_hit_s, total_dist, final_dist); 
					//rec.attenuated_Kd  = make_float3(0.0,1.0,1.0);
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
							//rec.attenuated_Kd = make_float3(0.0);
							//return;	
						}
						next_t_hit_c = 0.0;
					}
					else{
						rtPrintf(">else>>pm_index %d, ci:%d, ci_dir:%d, si:%d, si_dir:%d, <count:%d>, next_t_hit_c:%f, next_t_hit_s:%f, total_dist:%f, final_dist:%f\n",
							pm_index, ci, ci_dir, si, si_dir,
							count, next_t_hit_c, next_t_hit_s, total_dist, final_dist); 
						//rec.attenuated_Kd  = make_float3(10.0,0.0,0.0);
						//return;
					}

					//ci_temp =  findPosBin((orig_p + (final_dist/2.0)*v), light.direction, RES_MU_S_BIN);
					//ci_temp =  findPosBin((p + (dist/2.0)*v), light.direction, RES_MU_S_BIN);
					ci_temp =  findPosBin(p, light.direction, RES_MU_S_BIN);
					if(dist > 0){
						if(ci == -1){
							index = (int)(ci_temp*RES_R_BIN + (i_si))*RES_DIR_BIN_TOTAL; 
							//rec.attenuated_Kd  += dist*total_attn*photon_gather_contib_ext(pm_index, p, p+dist*v, v, index, num_photons, ci_temp, i_si);
							//rec.attenuated_Kd  = make_float3(1.0,0.0,0.0);
							//return;

						}
						else{
							index = (int)(ci_temp*RES_R_BIN + (i_si))*RES_DIR_BIN_TOTAL;
							//index = ((i_ci)*RES_R_BIN + (i_si))*RES_DIR_BIN_TOTAL;
							if(index <= ((RES_MU_S_BIN-1)*RES_R_BIN + RES_R_BIN-1)*RES_DIR_BIN_TOTAL){
								rec.attenuated_Kd  += dist*total_attn* photon_gather_contib_ext(pm_index, p, p+dist*v, v, index, num_photons, ci_temp, i_si);
								//if((ci == 8 || (ci>=10 && ci < 20)) && (si >= 3)){
								//if(count > 3 && count < 6 && num_ci== 1 && ci_dir == -1 && si_dir == -1 && ci_temp != i_ci){
								//	rtPrintf(">cone>>pm_index %d, ci:%d, ci_dir:%d, i_ci:%d, si:%d, si_dir:%d, i_si:%d, ci_temp:%d, num_si:%d, <count:%d>, next_t_hit_c:%f, next_t_hit_s:%f\n",
								//								pm_index, ci, ci_dir, i_ci, si, si_dir, i_si, ci_temp, num_si, 
								//								count, next_t_hit_c, next_t_hit_s); 
								//rec.attenuated_Kd += dist*total_attn* make_float3(0.0,0.0,0.0);
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
					rec.attenuated_Kd  += total_attn*make_float3(0.0,1.0,0.0);

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
						//rec.attenuated_Kd = make_float3(10.0,0.0,0.0);
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
							rec.attenuated_Kd  += dist*total_attn*photon_gather_contib_ext(pm_index, p, p+dist*v, v, index, num_photons, ci_temp, i_si);
							//rec.attenuated_Kd  += total_attn*make_float3(0.0,0.1,0.1);
							//if(pm_index == 146184)
							//rtPrintf("<<test gathering>pm_index %d, index: %d, si-dir:%d, ci_temp:%d, <num_photons:%d> si_dir:%d, <count:%d>, final_dist:%f dist:%f\n",
							//					pm_index, index, si-si_dir, ci_temp, num_photons, si_dir, 
							//					count, final_dist, dist); 

						}
						else {
							index = (int)(i_ci*RES_R_BIN + i_si)*RES_DIR_BIN_TOTAL;
							rec.attenuated_Kd  += dist*total_attn*photon_gather_contib_ext(pm_index, p, p+dist*v, v, index, num_photons, i_ci, i_si);
							//rec.attenuated_Kd = make_float3(0.0);
							//return;
						}


					}
					else{
						//if(pm_index == 23434)
						//	rtPrintf(">(sp and cone) pm_index:%d, count:%d, si:%d, ci:%d, si_dir:%d, ci_dir:%d, total_dist:%f, final_dist:%f\n",
						//								pm_index, count, si, ci, si_dir, ci_dir, total_dist, final_dist); 
						//rec.attenuated_Kd  = make_float3(1.0,0.0,0.0);
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
			rec.attenuated_Kd = make_float3(0.0,10.0,0.0);
			return;
			}
			*/


			//rec.attenuated_Kd *= SCALE_FACTOR;//1000000.0;//0.01;//100.0;//1000000000.0;//;
			//if((pm_index == 219778)){
			//	rtPrintf("<<>>pm_index %d, count:%d, rec.attenuated_Kd(%f,%f,%f)\n",pm_index, count, rec.attenuated_Kd.x,
			//	rec.attenuated_Kd.y, rec.attenuated_Kd.z);
			//}
			//if(pm_index == 30271){
			//	rec.attenuated_Kd = make_float3(0.4,0.0,1.0);
			//}

			//if(pm_index == 22626){
			//	rec.attenuated_Kd = make_float3(0.0,4.0,1.0);
			//}
			//if(length(p) < (Rt-0.1) && length(p) > (Rg+0.2)){ 

			//}
			


		}
		else{// (si_dir == 0)
			rec.attenuated_Kd = make_float3(165.0/255.0,42.0/255.0,42.0/255.0);
			return;
		}

		if(rec.attenuated_Kd.x <= 0.1 && rec.attenuated_Kd.y <= 0.1 && rec.attenuated_Kd.z <= 0.1){
			//if(count == 12){
			//	rec.attenuated_Kd  += inscatter(p, t, v, light.direction, 100.0, r, mu, temp, true);
			//rec.attenuated_Kd = make_float3(1.0,1.0,0.0);
			//rtPrintf("<RED>pm_index %d, count:%d,t:%f, total_dist:%f, final_dist:%f, len(p):%f\n",pm_index, count, t, total_dist, final_dist, length(p)); 
			//rec.attenuated_Kd = make_float3(10.0,00.0,0.0);
			//return;
		}
		//rec.attenuated_Kd *= 1000000000.0;


	}
	//rec.attenuated_Kd +=ground;
	//if(count > 2)
	//	rtPrintf("pm_index %d, count:%d,t:%f, total_dist:%f\n",pm_index, count, t, total_dist); 
}




RT_PROGRAM void transit_any_hit()
{
	rtIgnoreIntersection();
	/*
	float3 direction    = ray.direction;
	float3 origin       = ray.origin;
	float3 hit_point    = origin + t_hit*direction;

	optix::Ray ray( hit_point, direction, rtpass_ray_type, scene_epsilon );
	rtTrace( top_object, ray, hit_prd );

	/////////////////////////////////////////////
	// old code

	prd_shadow.prev_ray_length = t_hit;

	//what happens if we are inside a participating medium?
	prd_shadow.attenuation = make_float3(0.0f);
	prd_shadow.inShadow = true;
	//rtIgnoreIntersection();
	rtTerminateRay();
	*/
}

/*
rtTextureSampler<float4, 2>   diffuse_map;   
RT_PROGRAM void closest_hit_diffuse_radiance()
{
float3 hit_point = ray.origin + t_hit * ray.direction;

float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
float3 ffnormal               = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
float2 uv                     = make_float2(texcoord);

float3 Kd = make_float3(tex2D(diffuse_map, uv.x, uv.y));
float3 result = make_float3(0);

// Compute indirect bounce
if(hit_prd.ray_depth < 1) {
optix::Onb onb(ffnormal);
unsigned int seed = rot_seed( image_rnd_seeds[ launch_index ], frame );
const float inv_sqrt_samples = 1.0f / float(sqrt_diffuse_samples);

int nx = sqrt_diffuse_samples;
int ny = sqrt_diffuse_samples;
while(ny--) {
while(nx--) {
// Stratify samples via simple jitterring
float u1 = (float(nx) + rnd( seed ) )*inv_sqrt_samples;
float u2 = (float(ny) + rnd( seed ) )*inv_sqrt_samples;

float3 dir;
optix::cosine_sample_hemisphere(u1, u2, dir);
onb.inverse_transform(dir);

HitPRD radiance_prd;
radiance_prd.importance = hit_prd.importance * optix::luminance(Kd);
radiance_prd.ray_depth = hit_prd.ray_depth + 1;

if(radiance_prd.importance > 0.001) {
optix::Ray radiance_ray = optix::make_Ray(hit_point, dir, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
rtTrace(top_object, radiance_ray, radiance_prd);

result += radiance_prd.result;
}
}
nx = sqrt_diffuse_samples;
} 
result *= (Kd)/((float)(M_PI*sqrt_diffuse_samples*sqrt_diffuse_samples));
}

// Compute direct lighting
int num_lights = lights.size();
while(num_lights--) {
const BasicLight& light = lights[num_lights];
float3 L = light.pos - hit_point;
float Ldist = length(light.pos - hit_point);
L /= Ldist;
float nDl = dot( ffnormal, L);

if(nDl > 0.f) {
if(light.casts_shadow) {
PerRayData_shadow shadow_prd;
shadow_prd.attenuation = make_float3(1.f);
optix::Ray shadow_ray = optix::make_Ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
rtTrace(top_shadower, shadow_ray, shadow_prd);

if(fmaxf(shadow_prd.attenuation) > 0.f) {
result += Kd * nDl * light.color * shadow_prd.attenuation;
}
}
}
}

prd.result = result;
}
*/

/////////////////////////////////////////////////////////////////////////////////////
// intersection code (bypassing intersection program to preserve stack space)

__device__ __inline__ int shape_boundary_intersection(	int pm_index, float3 x, float3 v, float dist_boundary, int & sphere_layer, int & cone_layer, 
													  /* out*/ float &dist, /*out*/ bool & bSphere)
{
	float cone_dist = 0.0;
	float sphere_dist = 0.0;
	//int cone_layer = -1;
	//int sphere_layer = -1;
	bSphere = false;
	int cone_ret = -1;
	//int sphere_ret = sphere_next_boundary_dist(pm_index, x, v, dist_boundary, sphere_layer, sphere_dist, Rt, Rg, RES_R_BIN);
	int sphere_ret = sphere_next_boundary_dist_OLD(light.direction, pm_index, x, v, dist_boundary, sphere_layer, sphere_dist, Rt, Rg, RES_R_BIN);
	//layer = sphere_layer;
	//dist = sphere_dist; 
	//return sphere_ret;

	//layer = cone_layer;
	//dist = cone_dist;
	//return cone_ret;

	if(cone_ret == -1 && sphere_ret <= -1){
		return -1;
	}
	else{

		if(cone_ret == -1){
			bSphere = true;
			//layer = sphere_layer;
			dist = sphere_dist;
			return sphere_ret;
		}

		if(sphere_ret <= -1){
			//layer = cone_layer;
			dist = cone_dist;
			return cone_ret;
		}

		if(cone_ret != -1 && sphere_ret > -1){
			if(dist_boundary > 0){
				if(cone_dist < sphere_dist){
					//layer = cone_layer;
					dist = cone_dist;
					return cone_ret;
				}
				else{
					bSphere = true;
					//layer = sphere_layer;
					dist = sphere_dist;
					return sphere_ret;
				}
			}
			else{ //dist_boundary <= 0)
				if(cone_dist < 0.0 && sphere_dist < 0.0){
					if(fabs(cone_dist) < fabs(sphere_dist)){
						//layer = cone_layer;
						dist = cone_dist;
						return cone_ret;
					}
					else{
						bSphere = true;
						//layer = sphere_layer;
						dist = sphere_dist;
						return sphere_ret;
					}
				}
			}
		}

	}
	return -1;
}

// walk the ray and detect intersections with spheres and cones and accumulate radiance
__device__ __inline__ float3 inscatterN(float3 x_start, float3 x_end, float3 v_in)
{
}

// this should only be computed once for each BIN (similar to inscatterS.glsl)
__device__ __inline__ float3 photon_gather_contib(float3 x_start, float3 x_end, float3 v_in, int index)
{
	// Input: x_start (ray start poistion) x_end (ray primitive intersection) v_in (viewing direction)

	// Calculate mid coridnates x which is in the middle of the path through a bin
	// Calculate bin R id and mu_s id: use them to calculate Bin id

	float3 x = (x_end-x_start)/2.0;
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
	uint total_in_bin = 0;
	uint w_weight = 0;

	int muS_bound=findPosBin(x, light.direction, RES_MU_S_BIN);
	//int r_ind = 0; //TODO
	//int index = (muS_bound*RES_R_BIN + r_ind)*RES_DIR_BIN_TOTAL;

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

	for(int i=0; i < RES_DIR_BIN_TOTAL; i++){
		w_weight = 0;
		//if(index+i < 0 || index+i > RES_MU_S_BIN*RES_R_BIN*RES_DIR_BIN_TOTAL)
		//	rtPrintf("muS_bound: %d, r_ind: %d\n",muS_bound,r_ind);	

		w_weight = 0;//rtpass_global_photon_counts[index+i];	
		w = directionFromBin(i,RES_DIR_BIN);
		//rtPrintf("direction: (%f,%f,%f), w: (%f,%f,%f) i: %d\n",direction.x,direction.y,direction.z, w.x,w.y,w.z, i);

		nu2 = dot(v, w);
		pr2 = phaseFunctionR(nu2);
		pm2 = phaseFunctionM(nu2, mieG);

		if(w_weight > 0){
			result += w_weight*(betaR * exp(-(r - Rg) / HR) * pr2 + betaMSca * exp(-(r - Rg) / HM) * pm2);
			//result = w_weight*rayDen * pr2 + mieDen * pm2;
		}
		total_in_bin += w_weight;
	}

	//if(total_in_bin > 0.0)
	//	rtPrintf("(%d,%d) %d\n",muS_bound,r_ind, total_in_bin);		

	float volume = 0;//posbin_volume(x, s, Rt, Rg, RES_R_BIN, RES_MU_S_BIN);
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
	//	rtPrintf("(%d,%d) res(%f,%f,%f) atten(%f,%f,%f) %f\n",muS_bound,r_ind, result.x,result.y,result.z,attenuation.x,attenuation.y,attenuation.z,volume);	
	result = result*(1.0-attenuation);///volume;
	return result;
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
			return make_float3(0.09,0.0,0.0)*SCALE_FACTOR/volume;	
		}else if(ci == 4 || (ci > 99 && ci < 150)){
			return make_float3(0.0,0.09,0.0)*SCALE_FACTOR/volume;
		}else if(ci == 9 || (ci > 34 && ci < 60)){
			return make_float3(0.0,0.0,0.09)*SCALE_FACTOR/volume;
		}else if(ci > 19 && ci < 35){
			//	return make_float3(0.5,0.0,0.0);
		}else if(ci == 8 || (ci>=10 && ci < 20)){
			if(si < 20)
				return make_float3(0.03,0.004,0.0)*SCALE_FACTOR/volume;
			else
				return make_float3(0.0,0.0,0.0)*SCALE_FACTOR/volume;
		}
		return make_float3(0.0,0.1,0.1)*SCALE_FACTOR/volume;
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

	return SCALE_FACTOR*result/volume;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Terrain renderer
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////