#pragma once
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

template<unsigned int N>
__host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for( unsigned int n = 0; n < N; n++ )
  {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
__host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

__host__ __device__ __inline__ unsigned int lcg2(unsigned int &prev)
{
  prev = (prev*8121 + 28411)  % 134456;
  return prev;
}

// Generate random float in [0, 1)
__host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

// Multiply with carry
__host__ __inline__ unsigned int mwc()
{
  static unsigned long long r[4];
  static unsigned long long carry;
  static bool init = false;
  if( !init ) {
    init = true;
    unsigned int seed = 7654321u, seed0, seed1, seed2, seed3;
    r[0] = seed0 = lcg2(seed);
    r[1] = seed1 = lcg2(seed0);
    r[2] = seed2 = lcg2(seed1);
    r[3] = seed3 = lcg2(seed2);
    carry = lcg2(seed3);
  }

  unsigned long long sum = 2111111111ull * r[3] +
                           1492ull       * r[2] +
                           1776ull       * r[1] +
                           5115ull       * r[0] +
                           1ull          * carry;
  r[3]   = r[2];
  r[2]   = r[1];
  r[1]   = r[0];
  r[0]   = static_cast<unsigned int>(sum);        // lower half
  carry  = static_cast<unsigned int>(sum >> 32);  // upper half
  return static_cast<unsigned int>(r[0]);
}

__host__ __inline__ unsigned int random1u()
{
#if 0
  return rand();
#else
  return mwc();
#endif
}

__host__ __inline__ optix::uint2 random2u()
{
  return optix::make_uint2(random1u(), random1u());
}

__host__ __inline__ void fillRandBuffer( unsigned int *seeds, unsigned int N )
{
  for( unsigned int i=0; i<N; ++i ) 
    seeds[i] = mwc();
}

__host__ __device__ __inline__ unsigned int rot_seed( unsigned int seed, unsigned int frame )
{
    return seed ^ frame;
}

/*
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   // constant vector a 
#define UPPER_MASK 0x80000000UL // most significant w-r bits 
#define LOWER_MASK 0x7fffffffUL // least significant r bits 

static unsigned long mt[N]; // the array for the state vector  
static int mti=N+1; // mti==N+1 means mt[N] is not initialized 
// Random Number Functions
__host__ __device__ __inline__  void init_genrand(unsigned long seed) {
	mt[0]= seed & 0xffffffffUL;
	for (mti=1; mti<N; mti++) {
		mt[mti] =
		(1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
		// See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. 
		// In the previous versions, MSBs of the seed affect   
		// only MSBs of the array mt[].                        
		// 2002/01/09 modified by Makoto Matsumoto             
		mt[mti] &= 0xffffffffUL;
		// for >32 bit machines 
	}
}


__host__ __device__ __inline__ unsigned long genrand_int32(void)
{
	unsigned long y;
	unsigned long mag01[2]={0x0UL, MATRIX_A};
	// mag01[x] = x * MATRIX_A  for x=0,1 

	if (mti >= N) { // generate N words at one time 
		int kk;

		if (mti == N+1)   // if init_genrand() has not been called, 
			init_genrand(5489UL); // default initial seed 

		for (kk=0;kk<N-M;kk++) {
			y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
			mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		for (;kk<N-1;kk++) {
			y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
			mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
		mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

		mti = 0;
	}

	y = mt[mti++];

	// Tempering 
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

	return y;
}

__host__ __device__ __inline__ unsigned long RandomUInt() {
	return genrand_int32();
}

// generates a random number on [0,1)-real-interval 
__host__ __device__ __inline__ float genrand_real2(void)
{
	return (RandomUInt() & 0xffffff) / float(1 << 24);
}


__host__ __device__ __inline__ float RandomFloat() {
	return genrand_real2();
}
*/