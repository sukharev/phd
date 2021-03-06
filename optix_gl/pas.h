/**
 * Precomputed Atmospheric Scattering
 * Copyright (c) 2008 INRIA
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holders nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Author: Eric Bruneton
 */
#pragma once
//#define DEMO_MODE
const float Rg = 6360.0; //150;
const float Rt = 6420.0; //180;
const float RL = 6421.0; //181;
const float RS = RL*2;//1400; //sun distance
const float ptStep = 5;

const int TRANSMITTANCE_W = 256;//2024;
const int TRANSMITTANCE_H = 64;//2024;

const int SKY_W = 64;
const int SKY_H = 128;

/*
//original
const int RES_R = 32;
const int RES_MU = 128;
const int RES_MU_S = 32;
const int RES_NU = 8;
*/

//NVidia 330M 256MB
const int RES_R = 24;//24;//24
const int RES_MU = 128;//128;//128;   //v.x/r
const int RES_MU_S = 32;//32;		 //s.x/r
const int RES_NU = 8;//8;			 //v.s

const int RES_MU_S_BIN = 800;//RES_MU_S;//RES_MU_S*8;//RES_MU_S;
const int RES_R_BIN = 200;//12;//12;//RES_R;
const int RES_DIR_BIN = 5; // for each axis there RES_DIR_BIN directions total_num_bins = RES_DIR_BIN*RES_DIR_BIN*RES_DIR_BIN
const int RES_DIR_BIN_TOTAL = RES_DIR_BIN*RES_DIR_BIN*RES_DIR_BIN;

const int RES_PHOTONS_X = RES_MU_S*2;
const int RES_PHOTONS_Y = RES_MU_S*2;
const int BufferSize = RES_MU_S*2;

//resolution of each phase function
const int PHASE_DISTR_RES = 1024;