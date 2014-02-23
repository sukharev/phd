
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

#ifndef _PPM_H
#define _PPM_H

#include <optixu/optixu_math_namespace.h>

#define  PPM_X         ( 1 << 0 )
#define  PPM_Y         ( 1 << 1 )
#define  PPM_Z         ( 1 << 2 )
#define  PPM_LEAF      ( 1 << 3 )
#define  PPM_NULL      ( 1 << 4 )

#define  PPM_IN_SHADOW ( 1 << 5 )
#define  PPM_OVERFLOW  ( 1 << 6 )
#define  PPM_HIT       ( 1 << 7 )
#define  PPM_HIT_VOLUME       ( 1 << 8 )

const float MARCHING_STEP= 5;//10;//22;//12.0;
const float VIEWING_MARCHING_STEP= 22;//12.0;
const int MAX_NUM_EVENTS=1000;

typedef enum { 
  FILTER_BOX,
  FILTER_TRIANGLE,
  FILTER_GAUSSIAN,
  FILTER_MITCHELL,
  FILTER_SINC,
  NUM_FILTERS
} zpFilterType;


struct PPMLight
{
  optix::uint   is_area_light;
  optix::float3 power;

  // For spotlight
  optix::float3 position;
  optix::float3 direction;
  float         radius;

  // Parallelogram
  optix::float3 anchor;
  optix::float3 v1;
  optix::float3 v2;
  optix::float3 v3;

  // For distant
  optix::float3 worldCenter;
  float  worldRadius;
  

  //plane connecting light anchor and world center (Earth center)
  optix::float3 plane_v1;
  optix::float3 plane_v2;
};

struct HitRecord
{
 // float3 ray_dir;          // rgp

  optix::float3 position;         //
  optix::float3 normal;           // Material shader
  optix::float3 attenuated_Kd;
  optix::uint   flags;

  float         radius2;          //
  float         photon_count;     // Client TODO: should be moved clientside? (make sure to add vol_photon_count
  optix::float3 flux;             //
  float         accum_atten;
  optix::float3	direction;		  // ray cast from the camera (eye)
  float			t_hit_exitvolume; // position in the ray near the furthest side of volume along a ray from the camera
  float			step;
  optix::uint   depth;
  optix::float2 temp;
};


struct PackedHitRecord
{
  optix::float4 a;   // position.x, position.y, position.z, normal.x
  optix::float4 b;   // normal.y,   normal.z,   atten_Kd.x, atten_Kd.y
  optix::float4 c;   // atten_Kd.z, flags,      radius2,    photon_count
  optix::float4 d;   // flux.x,     flux.y,     flux.z,     accum_atten 
  optix::float4 e;   // direction.x direction.y direction.z t_hit_exitvolume
  optix::float4 f;   // step
};


struct HitPRD
{
  optix::float3 attenuation;
  optix::uint   ray_depth;
  //optix::float3 Lv;
  //optix::float3 L0;
  float total_dist;
  optix::uint bSurfaceReached;
  optix::uint   ray_vol_depth;
  float	 prev_ray_length;
  float  step;
  int numSteps;
};


struct PhotonRecord
{
  optix::float3 position;
  optix::float3 normal;      // Pack this into 4 bytes
  optix::float3 ray_dir;
  optix::float3 energy;
  optix::uint   axis;
  //optix::float3 pad;
  optix::float3 position_rotate;
  optix::float3 ray_dir_rotate;
  float pad;

};


struct PhotonCountRecord
{
  optix::uint pos_dir_index;
  optix::uint dir_count;
  //optix::uint3 cuda_index;
};

struct PackedPhotonRecord
{
  optix::float4 a;   // position.x, position.y, position.z, normal.x
  optix::float4 b;   // normal.y,   normal.z,   ray_dir.x,  ray_dir.y
  optix::float4 c;   // ray_dir.z,  energy.x,   energy.y,   energy.z
  //optix::float4 d;   // axis,       
  optix::float4 d;   // axis,       position_rotate.x, position_rotate.y, position_rotate.z
  optix::float4 e;   // ray_dir_rotate.x ray_dir_rotate.y ray_dir_rotate.z pad
};


struct PhotonPRD
{
  optix::uint   type;  // 0 - undefined, 1 - indirect, 2 - volume
  optix::float3 energy;
  optix::uint2  sample;
  optix::uint   pm_index;
  optix::uint	pm_vol_index;
  optix::uint   num_deposits;
  optix::uint   num_voldeposits;
  optix::uint   ray_depth;
  float prev_ray_length;
};


struct ShadowPRD
{
  float attenuation;
};


// support for volume photon mapping
enum EMaterialType
{
  MT_Matte,
  MT_Metal,
  MT_Glass,
  MT_GlossyMetal,
  MT_Light
};

struct zpSample {
  float x;
  float y;
  HitRecord value;
};


struct TrackIntersection{
  int si; // RES_R_BIN;
  int ci;//RES_MU_S_BIN;
  int si_dir; // -1: down, +1: up
  int ci_dir; // -1: down, +1: up
  bool b_si_dir; 
  bool b_ci_dir;
  bool b_ci_change;

  float next_t_hit_s;
  float next_t_hit_c;
};

#endif // _PPM_H