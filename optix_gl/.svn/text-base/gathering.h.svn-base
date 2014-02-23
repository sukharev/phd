
#ifndef _GATHERING_H
#define _GATHERING_H

#include <optix.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include "ppm.h"
#include "random.h"
#include "helpers.h"

__device__ __inline__ int findRBin(optix::float3 new_hit_point,  float Rt, float Rg, int RES_R_BIN);
__device__ __inline__ void rotate_with_sundir(	float3 x, float3 & new_x, float3 light_dir);
__device__ __inline__ int cone_next_boundary_dist(int pm_index, float3 x, float3 v, float curr_dist, 
												  float muS, float & out_dist, int & out_ind,
												  float3 light_dir, const optix::Ray & ray, float Rg, float Rt, int RES_MU_S_BIN);
__device__ __inline__ int intersect_spheres_dir(uint pm_index, int dir_center, float3 light_dir, 
						  float3 x, float3 v, float t, 
						  int & out_ind /*in/out*/, float & out_dist,
						  float Rt, float Rg, int RES_R_BIN, bool & bChangeDir);
__device__ __inline__ bool intersect_spheres_nodir(uint pm_index, float3 light_dir, 
						  float3 x, float3 v, float t, 
						  int & out_ind /*in/out*/, float & out_dist,
						  float Rt, float Rg, int RES_R_BIN);
__device__ __inline__ int findRBin(float len,  float Rt, float Rg, int RES_R_BIN);

using namespace optix;
__device__ __inline__ int sphere_intersection(	float3 light_dir, float3 xx, float3 vv, int updownsp, /*-1 or -2 or actual index */
												/* out*/ float &root1, /* out*/ float & root2, /*out*/int & radius_ind,
												float Rt, float Rg, int RES_R_BIN)
{
    // find points of intersection with sphere X 
	// need to determine radius of sphere X before we proceed
	
	//UP or DOWN???
	
	float3 x, v;
	//rotate_with_sundir(xx,x, light_dir);
	//rotate_with_sundir(vv,v, light_dir);
	x= xx;
	v = vv;

	float3 direction = v;
	float r = length(x);



	
	//if(updownsp == -1)
	//	radius_ind = (int)floor(((float)(r-Rg)/(float)(Rt-Rg))*(RES_R_BIN-1));
	//else if(updownsp == -2)
	//	radius_ind = (int)ceil(((float)(r-Rg)/(float)(Rt-Rg))*(RES_R_BIN-1));
	//else 
	radius_ind = updownsp;
	
	if(radius_ind>(RES_R_BIN-1) || radius_ind < 0)
		return -1;

	float radius = ((float)radius_ind/(float)(RES_R_BIN-1))*(float)(Rt-Rg)+Rg;
	

	float A = dot(direction,direction);
	float B = 2*dot(direction,x);
	float C = dot(x,x) - radius*radius;
	float Desc = B*B-4*A*C;
	int result = -1;

	float dist = 0.0;
	if(Desc > 0){
		// 2 solutions
		root1 = (-B+sqrtf(Desc))/(2*A);
		root2 = (-B-sqrtf(Desc))/(2*A);
		if(root1 < root2){
			dist = root1;
		}
		else{
			dist = root2;
		}
		result = 1;
	}
	else if(Desc == 0){
		// 1 solution
		root1 = -B/(2*A);
		dist = root1;
		root2 = 0.0;
		result = 0;
	}
	else{
		root1 = 0.0;
		root2 = 0.0;
		result = -1;
		// no intersections
	}

	if(result >= 0){
		float3 new_x = x + dist*v;
		float new_r = length(new_x);
		int new_radius_ind = (int)(((float)(new_r-Rg)/(float)(Rt-Rg))*(RES_R_BIN-1));
		//if(r > new_r && radius_ind > new_radius_ind+1){
		//	return - 1;
		//}
		//else if(r < new_r && radius_ind > new_radius_ind+1){
		//}
	}
	return result;
}


__device__ __inline__ int searchall_spheres_dir(uint pm_index, int dir_center, float3 light_dir, 
						  float3 x, float3 v, float t, 
						  int & out_ind /*in/out*/, float & out_dist,
						  float Rt, float Rg, int RES_R_BIN, bool & bChangeDir)
{
	int maxind=RES_R_BIN-1;
	float next_t_hit1 = 0.0;
	float next_t_hit2 = 0.0;
	float next_t_hit3 = 0.0;
	

	int count =0;
	int new_dir_center = 0;//dir_center;
	bool test_bChangeDir = false;
	bChangeDir = false;
	out_dist = 0.0;
	int out_ind1 = findRBin(length(x), Rt, Rg, RES_R_BIN);
	int out_ind2 = (out_ind1 + 1)<=(RES_R_BIN-1) ? out_ind1 + 1 : RES_R_BIN -1 ;
	int out_ind3 = (out_ind1 - 1)>=0 ? out_ind1 - 1 : 0 ;

	float next_t_hit = 0.0;
	test_bChangeDir = false;
	bool bInt1 = intersect_spheres_nodir(pm_index, light_dir, 
					  x, v, t, 
					  out_ind1, next_t_hit1,
					  Rt, Rg, RES_R_BIN);
	bool bInt2 = intersect_spheres_nodir(pm_index, light_dir, 
					  x, v, t, 
					  out_ind2, next_t_hit2,
					  Rt, Rg, RES_R_BIN);
	bool bInt3 = intersect_spheres_nodir(pm_index, light_dir, 
					  x, v, t, 
					  out_ind3, next_t_hit3,
					  Rt, Rg, RES_R_BIN);
		

	//cases:
	if(bInt1){
		if(bInt2 && bInt3){
			if(next_t_hit2 > next_t_hit1 && next_t_hit3< next_t_hit1){
				new_dir_center = 1;
				out_dist = next_t_hit1;
				out_ind = out_ind3;
			}
			else if(next_t_hit2 < next_t_hit1 && next_t_hit1< next_t_hit3){
				new_dir_center = -1;
				out_dist = next_t_hit1;
				out_ind = out_ind1;
			}
		}
		else{
			if(bInt2){
				if(next_t_hit2 > next_t_hit1){
					new_dir_center = 1;
					out_dist = next_t_hit1;
					out_ind = out_ind3;
				}
				else{
					new_dir_center = -1;
					out_dist = next_t_hit1;
					out_ind = out_ind1;
				}
			}
			else if(bInt3){
				if(next_t_hit3 > next_t_hit1){
					new_dir_center = -1;
					out_dist = next_t_hit1;
					out_ind = out_ind1;
				}
				else{
					new_dir_center = 1;
					out_dist = next_t_hit1;
					out_ind = out_ind3;
				}
			}
			else{
				new_dir_center = 0;
				out_dist = next_t_hit1;
				out_ind = out_ind1;
			}
		}
	}
	else{
		if(bInt2 && bInt3){
			//problem 
			new_dir_center = 0;
			out_dist =0.0;
		}
		else{
			if(bInt2){
				new_dir_center = 1;//0;
				out_dist = next_t_hit2;
				out_ind = out_ind1;
			}
			else if(bInt3){
				new_dir_center = -1;//0;
				out_dist = next_t_hit3;
				out_ind = out_ind3;
			}
			else{
				new_dir_center = 0;
			}
		}
	}
	
	return new_dir_center;
}

// return bool (intersection found or not)
// @param dir_center:	-1, +1
// @param in_ind	:	0...RES_R_BIN-1 
__device__ __inline__ bool intersect_spheres_nodir(uint pm_index, float3 light_dir, 
						  float3 x, float3 v, float t, 
						  int & out_ind /*in/out*/, float & out_dist,
						  float Rt, float Rg, int RES_R_BIN)
{
	float root1, root2;
	int res = -1;

	if(out_ind >= RES_R_BIN || out_ind < 0)
		return 0;
	int in_ind = out_ind;
	res = sphere_intersection(light_dir, x, v, in_ind, root1, root2, out_ind, Rt, Rg, RES_R_BIN);
	
	//no intersection
	if(res < 0)
	{
		return false;
	}
	else{

		float min_len = 0.05;
		// ignoring t for now
		if(root2 > root1 && root2 > 0.0){
			if(root1 > min_len){
				out_dist =  root1;
			}
			else{
				out_dist =  root2;
			}
		}
		else if(root1 > root2 && root1 > 0.0){
			if(root2 > min_len){
				out_dist =  root2;
			}
			else{
				out_dist =  root1;
			}
		}
		else{
			if(root1 < 0.0 && root2 < 0.0){
				return false;
				//out_dist = 0.0;
			}
			else{
				if(root1 < 0.0){
					out_dist = root2;
				}
				else if(root2 < 0.0)
					out_dist = root1;
			}
		}
	}
	return true;
}


// return new direction
// @param dir_center:	-1, +1
// @param in_ind	:	0...RES_R_BIN-1 
__device__ __inline__ int intersect_spheres_dir(uint pm_index, int dir_center, float3 light_dir, 
						  float3 x, float3 v, float t, 
						  int & out_ind /*in/out*/, float & out_dist,
						  float Rt, float Rg, int RES_R_BIN, bool & bChangeDir)
{
	float root1, root2;
	int res = -1;
	int new_dir_center = dir_center;
	bChangeDir = false;
	out_dist = 0.0;
	if(out_ind >= RES_R_BIN || out_ind < 0)
		return 0;
	int save_ind = out_ind;
	int in_ind = out_ind;
	res = sphere_intersection(light_dir, x, v, in_ind, root1, root2, out_ind, Rt, Rg, RES_R_BIN);
	
	//no intersection
	if(res < 0)
	{
		if(dir_center == -1){
			if(in_ind+1 >= RES_R_BIN || in_ind+1 < 0)
				return 0;
			in_ind = in_ind +1;
			res = sphere_intersection(light_dir, x, v, in_ind, root1, root2, out_ind, Rt, Rg, RES_R_BIN);
			if(res < 0)
					return 0;
			bChangeDir = true;
			new_dir_center = 1;
		}
		else if(dir_center == 1){
			if(in_ind+1 >= RES_R_BIN || in_ind+1 < 0)
					return 0;
			in_ind = in_ind + 1;
			res = sphere_intersection(light_dir, x, v, in_ind, root1, root2, out_ind, Rt, Rg, RES_R_BIN);
			if(res < 0)
					return 0;
			new_dir_center = 1;
		}
	}
	if(root1 < 0.0 && root2 < 0.0){
		if(dir_center == -1){
			if(in_ind+1 >= RES_R_BIN || in_ind+1 < 0)
					return 0;
			res = sphere_intersection(light_dir, x, v, in_ind+1, root1, root2, out_ind, Rt, Rg, RES_R_BIN);
			if(res < 0)
					return 0;
			bChangeDir = true;
			new_dir_center = 1;
		}
		else if(dir_center == 1){
			in_ind++;
			if(in_ind >= RES_R_BIN || in_ind < 0)
					return 0;
			
			while(in_ind < RES_R_BIN){
				res = sphere_intersection(light_dir, x, v, in_ind, root1, root2, out_ind, Rt, Rg, RES_R_BIN);
				if(res >= 0 && (root1 > 0.0 || root2 > 0.0))
					break;
				in_ind++;
			}
			if(res < 0)
					return 0;

			new_dir_center = 1;
		}
	}

	float min_len = 0.05;
	// ignoring t for now
	if(root1 <= 0.0 && root2 <= 0.0){
			out_dist = 0.0;//(fabs(root2) > fabs(root1)) ? root1 : root2;
			new_dir_center = 0;
	}
	else if(root2 >= root1){
		if(root1 > min_len && new_dir_center == -1){
			out_dist =  root1;
		}
		else if(root2 > 0.0 && root1 <= 0.0){
			new_dir_center = 1;
			out_dist =  root2;
		}
	}
	else if(root1 > root2){
		if(root2 > min_len && new_dir_center == -1){
			out_dist =  root2;
		}
		else if(root1 > 0.0 && root2 <= 0.0){
			new_dir_center = 1;
			out_dist =  root1;
		}
	}
	else{
		//??????
		if(root1 < 0.0 && root2 < 0.0){
			out_dist = 0.0;//(fabs(root2) > fabs(root1)) ? root1 : root2;
			new_dir_center = 0;
		}
		else{
			if(root1 < 0.0){
				out_dist = root2;
			}
			else if(root2 < 0.0)
				out_dist = root1;


		}
	}

	if(new_dir_center == 0 && dir_center == 1 && save_ind>0 && save_ind < (RES_R_BIN-1)){
		
		in_ind = save_ind;
		while(in_ind < RES_R_BIN){
			res = sphere_intersection(light_dir, x, v, in_ind, root1, root2, out_ind, Rt, Rg, RES_R_BIN);
			if(res >= 0 && (root1 > 0.0 || root2 > 0.0))
				break;
			in_ind++;
		}
		if(res >= 0){
			if(root1 > 0.0 || root2 > 0.0){
				//rtPrintf("intersection: out_ind:%d, save_ind:%d \n", out_ind, save_ind);	
				if(root2 >= root1){
					if(root1 > min_len){
						new_dir_center = 1;
						out_dist =  root1;
					}
					else if(root2 > 0.0 && root1 <= 0.0){
						new_dir_center = 1;
						out_dist =  root2;
					}
				}
				else if(root1 > root2){
					if(root2 > min_len){
						new_dir_center = 1;
						out_dist =  root2;
					}
					else if(root1 > 0.0 && root2 <= 0.0){
						new_dir_center = 1;
						out_dist =  root1;
					}
				}
			}
		}
	}
	return new_dir_center;
}

// return new direction
// @param dir_center:	-1, +1
// @param in_ind	:	0...RES_MU_S_BIN-1
__device__ __inline__ int intersect_cones_dir(uint pm_index, int dir_center, float3 light_dir, 
						  float3 x, float3 v, float t, 
						  int & out_ind, float & out_dist,
						  float Rt, float Rg, int RES_MU_S_BIN, const optix::Ray & ray, bool & bChangeDir)
{
	float root1, root2;
	int res = -1;
	int new_dir_center = dir_center;
	float next_t_hit = 0.0;
	float cos_cone;
	out_dist = 0.0;
	int maxind=RES_MU_S_BIN-1;
	float min_len = 0.001;  // minimal distance to the nearest cone
	int temp_out_ind = -1;
	bChangeDir = false;
	int in_ind = out_ind;
	if(in_ind == -1){
		for(int i=0; i <= maxind; i++){
			next_t_hit = 0.0;
			//cos_cone = (float)(2*maxind-i)/(float)(2*maxind);
			cos_cone = (float)(i)/(float)(maxind);
			if(cone_next_boundary_dist(pm_index, x, v, t, 
										cos_cone, next_t_hit, temp_out_ind,
										light_dir, ray, Rg, Rt, RES_MU_S_BIN) != -1){
				if(next_t_hit < t && next_t_hit > 0.0){
					if(out_dist == 0.0 || (out_dist > 0.0 && next_t_hit < out_dist)){
						//out_ind = temp_out_ind;//i;//(int)(cos_cone * (RES_MU_S_BIN-1));
						out_ind = i;
						out_dist = next_t_hit;
					}
				}
			}
		}


		if(out_dist <= 0.0){
			out_dist = 0.0;
			new_dir_center = 0;
		}
		else{
			//TODO: error need to determine the direction
			new_dir_center = -1;
		}
	}
	else{
		if(in_ind >= RES_MU_S_BIN || in_ind < 0)
			return 0;
		cos_cone = (float)in_ind/(float)(RES_MU_S_BIN-1);
		res = cone_next_boundary_dist(pm_index, x, v, t, 
									cos_cone, out_dist, temp_out_ind,
									light_dir, ray, Rg, Rt,RES_MU_S_BIN);
		
		//no intersection
		if(res < 0)
		{
			if(in_ind+1 >= RES_MU_S_BIN || in_ind+1 < 0)
				return 0;

			if(dir_center == -1){
				out_dist = min_len;
				for(int i=0; i <= maxind; i++){
					next_t_hit = 0.0;
					//cos_cone = (float)(2*maxind-i)/(float)(2*maxind);
					cos_cone = (float)(i)/(float)(maxind);
					if(cone_next_boundary_dist(pm_index, x, v, t, 
												cos_cone, next_t_hit, temp_out_ind,
												light_dir, ray, Rg, Rt, RES_MU_S_BIN) != -1){
						if(next_t_hit < t && next_t_hit > 0.0){
							if(out_dist == min_len || (out_dist > min_len && next_t_hit < out_dist)){
								//out_ind = temp_out_ind;//out_ind = i;//(int)(cos_cone * (RES_MU_S_BIN-1));
								out_ind = i;
								out_dist = next_t_hit;
							}
						}
					}
				}

				if(out_dist <= min_len){
					out_dist = 0.0;
					new_dir_center = 0;
				}
				else{
					//cos_cone = (float)(in_ind+1)/(float)(RES_MU_S_BIN-1);
					//res = cone_next_boundary_dist(pm_index, x, v, t, 
					//					cos_cone, out_dist, out_ind,
					//					light_dir, ray, Rg, Rt);

					//TODO: error need to determine the direction
					new_dir_center = 1;
					bChangeDir = true;
				}
			}
			else
				new_dir_center = 0; //PROBLEM
		}
		else{
			//out_ind = temp_out_ind;
		}
	}
	
	if(new_dir_center != 0){
		float small_dist = 0.1;
		float new_out_dist1 = 0.0;
		float new_out_dist2 = 0.0;
		new_dir_center = 0;
		//find the true direction
		//1. try large cone (dist:1)
		cos_cone = (out_ind)/(float)(maxind);
		int new_maxind = maxind*100-1;
		int new_out_ind = cos_cone * (new_maxind);
		cos_cone = (new_out_ind+1)/(float)(new_maxind);
		if(cone_next_boundary_dist(pm_index, x, v, t, 
									cos_cone, next_t_hit, temp_out_ind,
									light_dir, ray, Rg, Rt, (new_maxind+1)) != -1){
			if(next_t_hit < t && next_t_hit > 0.0 && next_t_hit > out_dist){
				new_dir_center = 1;
				new_out_dist1 = next_t_hit;
			}
		}

		if(new_dir_center == 0){
			cos_cone = (new_out_ind-1)/(float)(new_maxind);
			if(cone_next_boundary_dist(pm_index, x, v, t, 
										cos_cone, next_t_hit, temp_out_ind,
										light_dir, ray, Rg, Rt, (new_maxind+1)) != -1){
				if(next_t_hit < t && next_t_hit > 0.0 && next_t_hit > out_dist){
					new_dir_center = -1;
					new_out_dist2 = next_t_hit;
				}
			}
		}

	}
	return new_dir_center;
}

__device__ __inline__ bool sphere_select_roots(float t, int layer, float root1, float root2, int & out_ind, float & out_dist)
{
	bool bRes = false;
	float old_out_dist = out_dist;
	int old_out_ind = out_ind;
	if(t >= 0.0){
		if(root1 <= 0.0 && root2 <= 0.0){
			out_dist = 0.0;
			out_ind = -1;
		}
		else{ //(root1 >= 0.0 && root2 >= 0.0)
			if(root1 < root2 && root1 > 0.0){
				out_dist = root1;
				out_ind = layer;
				bRes = true;
			}
			else if(root2 < root1 && root2 > 0.0){
				out_dist = root2;
				out_ind = layer;
				bRes = true;
			}
			else{
				if(root1 > 0.0 && root2 == 0.0){
					out_dist = root1;
				}
				else if(root2 > 0.0 && root2 == 0.0){
					out_dist = root2;
				}
				out_ind = layer;
				bRes = true;
			}
		}
		if((old_out_dist < out_dist && old_out_dist > 0.0) || out_dist == 0.0){
			out_ind = old_out_ind;
			out_dist = old_out_dist;
			bRes = true;
		}

	}
	else{ // t < 0
		if(root1 >= 0.0 && root2 >= 0.0){
			out_dist = 0.0;
			out_ind = -1;
		}
		else{// if( root1 < 0.0 && root2 < 0.0){
			out_dist = (fabs(root1) <= fabs(root2) && root1 < 0.0) ? root1 : root2;
			out_ind = layer;
			bRes = true;
		}
	}
	
	if(out_dist == 0.0)
		bRes = false;
	return bRes;
}

__device__ __inline__ int sphere_next_boundary_dist(float3 light_dir, int pm_index, float3 x, float3 v, float t, int & out_ind, float & out_dist,
													float Rt, float Rg, int RES_R_BIN)
{
	//int R_bound = findRBin(x, Rt, Rg, RES_R_BIN);
	float root1, root2;
	float temp_d = 0.0;
	int temp_i = -1;
	int ind = -1;
	int result = -1;
	out_dist = 0.0;
	out_ind = -1;
	int res = -1;
	float temp_d1 = out_dist;

	for(int i = 0; i < RES_R_BIN; i++){ 
		res = sphere_intersection(light_dir, x, v, i, root1, root2, ind, Rt, Rg, RES_R_BIN);
		if(res >= 0){
			//bDone = sphere_select_roots(t, ind, last, root1, root1, temp_i, temp_d);
			if(sphere_select_roots(t, ind, root1, root2, out_ind, out_dist))
				result = 1;
		}
		
		//if(bDone){
		//	temp_d1 = out_dist;
		//	sphere_select_roots(t, ind, temp_d, temp_d1, out_ind, out_dist);
		//	result = 1;
		//}
	}
	
	if(out_dist == 0.0 || out_ind < 0 )
		return -1;
	return result;
}
__device__ __inline__ int sphere_next_boundary_dist_OLD(float3 light_dir, int pm_index, float3 x, float3 v, float t, int & out_ind, float & out_dist,
													float Rt, float Rg, int RES_R_BIN)
{
	float root1d, root2d;
	float root1u, root2u;
	float root1l, root2l;
	int ind_down = 0;
	int ind_up = 0;
	int res_down = -1;
	int res_up = -1;
	int ind_lim = -1;
	res_down = sphere_intersection(light_dir, x, v, -1, root1d, root2d, ind_down, Rt, Rg, RES_R_BIN);
	res_up = sphere_intersection(light_dir, x, v, -2, root1u, root2u, ind_up, Rt, Rg, RES_R_BIN);
	int res_limit = sphere_intersection(light_dir, x, v, 0, root1l, root2l, ind_lim, Rt, Rg, RES_R_BIN);
	//float total_roots = root1u+root2u+root1d+root2d;
	
	int prev_ind = 0;
	if(ind_up != ind_down && ind_up > 0)
		prev_ind = ind_up -1;
	else
		prev_ind = ind_up;	
	/*
	if(ind_up == ind_down){// || total_roots <= 0.0){
		
		if((root1u == 0.0 && root2u == 0.0) || res_down == -1){
			if(prev_ind > 0)
				res_down = sphere_intersection(light_dir, x, v, prev_ind-1, root1d, root2d, ind_down, Rt, Rg, RES_R_BIN);
			else if(prev_ind < RES_R_BIN)
				res_up = sphere_intersection(light_dir, x, v, prev_ind+1, root1u, root2u, ind_up, Rt, Rg, RES_R_BIN);
		}
	}
	*/
	
	//if(pm_index == 431660)
	if(res_down == -1 && res_up == -1)
	{
		
		float r_len = length(x);
		int ind_old = (int)ceil(((float)(r_len-Rg)/(float)(Rt-Rg))*(RES_R_BIN-1));
		//rtPrintf("t=%f , prev_ind:%d, r_len: %f, ind_old: %d, DOWN: %d, UP: %d, x(%f,%f,%f),v(%f,%f,%f)\n",t, prev_ind, r_len, ind_old, ind_down, ind_up, x.x,x.y,x.z,v.x,v.y,v.z);
		//rtPrintf("pm_index %d, t=%f ,DOWN: %d, (1):%f, (2):%f, UP: %d, (1):%f, (2):%f\n",pm_index, t, ind_down, root1d, root2d, ind_up, root1u, root2u);
		return -1;
	}
	
	float arr[4] = {root1u,root2u, root1d, root2d};	
/*
	float min_root_pos = 0.0;
	float min_root_neg = 0.0;
	int min_root_pos_ind = -1;
	int min_root_neg_ind = -1;
	for(int i=0; i < 4; i++){
		if(arr[i] < 0.0){
			if(min_root_neg == 0 || min_root_neg > fabs(arr[i])){
				min_root_neg = arr[i];
				min_root_neg_ind = i;
			}
		}
		else{ // arr[i] >= 0.0
			if(min_root_pos == 0 || min_root_pos > fabs(arr[i])){
				min_root_pos = arr[i];
				min_root_pos_ind = i;
			}
		}
	}

	if(min_root_pos != 0.0){
		out_dist = min_root_pos;
		if(min_root_pos_ind < 2) 
			out_ind = ind_up;
		else
			out_ind = ind_down;
	}

	if(min_root_neg != 0.0 && min_root_pos == 0.0){
		out_dist = min_root_neg;
		if(min_root_neg_ind < 2) 
			out_ind = ind_up;
		else
			out_ind = ind_down;
	}

	if(out_dist == 0.0 || (fabs(out_dist) > fabs(t)) ){// && ind_down <= RES_R_BIN){// && minpos < 0.05){
		return -1;
	}
	
	if(out_ind < 0)
		return -1;
*/

	out_dist = 0.0;
	float minpos = 0.0; // minimum positive
	int minpos_ind = 0;
	for(int i=0; i < 4; i++){
		if(arr[i] > 0.0 && ((minpos == 0.0) || (minpos < arr[i])) ){
			minpos = arr[i];
			if(i < 2) 
				minpos_ind = ind_up;
			else
				minpos_ind = ind_down;
		}
		
		if(arr[i] >= 0.05 && out_dist == 0.0){
			out_dist = arr[i];
			if(i < 2) 
				out_ind = ind_up;
			else
				out_ind = ind_down;
		}
		else if(arr[i] >= 0.05 && arr[i] < out_dist){
			out_dist = arr[i];
			if(i < 2) 
				out_ind = ind_up;
			else
				out_ind = ind_down;
		}
	}


	//if(out_dist == 0.0 && ind_down < RES_R_BIN && minpos > 0.0){
	//	out_dist = minpos;
	//	out_ind = minpos_ind;
	//}

	if(out_dist == 0.0 && ind_down <= RES_R_BIN && minpos < 0.05){
	    //rtPrintf("pm_index %d, t=%f ,DOWN: %d, (1):%f, (2):%f, UP: %d, (1):%f, (2):%f\n",pm_index, t, ind_down, root1d, root2d, ind_up, root1u, root2u);
		return -1;
	}	
  
	if(res_limit != -1){
		if(root1l < root2l && root1l < out_dist && root1l > 0.0){
			out_dist = root1l;
			out_ind = 0;
		}
		else if(root2l < root1l && root2l < out_dist && root2l > 0.0){
			out_dist = root2l;
			out_ind = 0;
		}
	}

	if(fabs(t) < fabs(out_dist))
		return -1;

	//rtPrintf("pm_index %d,DOWN: %d, (1):%f, (2):%f, UP: %d, (1):%f, (2):%f\n",pm_index, ind_down, root1d, root2d, ind_up, root1u, root2u);
	return 0;
}

__device__ __inline__ void findPosBinEnvelope(float3 point, float& upper, float & lower, float3 lightdir, int RES_MU_S_BIN)
{

	float3 norm_point = normalize(point);
	float muS = fabsf(dot(lightdir,norm_point)); //ONLY store cos values from 0 to 1
	float muS_sin = sqrtf(1 - muS*muS);
	int lower_ind = floor((float)((muS_sin)*(int)(RES_MU_S_BIN-1)));
    int upper_ind = ceil((float)((muS_sin)*(int)(RES_MU_S_BIN-1)));
	//lower_ind = lower_ind > 0 ? lower_ind - 1 : 0;
	/*
	if(lower_ind == upper_ind){
		if(upper_ind < (RES_MU_S_BIN-1)){
			upper_ind = upper_ind +1;
		}
		else{
			upper_ind = (RES_MU_S_BIN-1);
			if(lower_ind > 0){
				lower_ind = lower_ind -1;
			}
		}
	}
	*/
	lower = (float)lower_ind/(float)(RES_MU_S_BIN-1);
	upper = (float)upper_ind/(float)(RES_MU_S_BIN-1);
	//if(lower == upper){
	//	rtPrintf("lower_ind = %d, upper_ind = %d, lower = %f, upper = %f\n", lower_ind, upper_ind, lower, upper);
	//}
	
}

__device__ __inline__ int findPosBin(float muS,  int RES_MU_S_BIN)
{
	//if(muS > 0.0)
	//	return -1;
	//float muS_sin = sqrtf(1 - muS*muS);
	int lower_ind = floor((float)((muS)*(int)(RES_MU_S_BIN-1)));
	//lower_ind = lower_ind > 0 ? lower_ind - 1 : 0;
	return lower_ind;
}

__device__ __inline__ int findPosBin(optix::float3 new_hit_point,  float3 light_dir, int RES_MU_S_BIN)
{

	float upper,lower;
	float muS = dot(normalize(new_hit_point),light_dir);
	//if(muS > 0.0)
	//	return -1;
    findPosBinEnvelope(new_hit_point, upper, lower, light_dir, RES_MU_S_BIN);


	float muS_sin = sqrtf(1 - muS*muS);
	int lower_ind = floor((float)((muS_sin)*(int)(RES_MU_S_BIN-1)));
	//int lower_ind = (int)(lower * (RES_MU_S_BIN-1));

    return lower_ind;
}

__device__ __inline__ int findRBin(float len,  float Rt, float Rg, int RES_R_BIN)
{
	if(len < 0.0)
		return -1;
	float rt_rg = (Rt-Rg);
	float r_rg = (len-Rg);
	int ind = floor((float)(r_rg/rt_rg)*(int)(RES_R_BIN-1)); //or /(RES)
	return ind;
}

__device__ __inline__ int findRBin(optix::float3 new_hit_point,  float Rt, float Rg, int RES_R_BIN)
{
	float len_r = sqrtf(dot(new_hit_point, new_hit_point));
	float rt_rg = (Rt-Rg);
	float r_rg = (len_r-Rg);

	if(r_rg < 0.0)
		return -1;
	// it appears that we count all angles from 0 to M_PI when we should count only
	// from M_PI/2...0...M_PI/2 or only from 0...M_PI/2
	int ind = floor((float)(r_rg/rt_rg)*(int)(RES_R_BIN-1)); //or /(RES)
	return ind;
}


__device__ __inline__ void rotateVectorToLightPlane( const PPMLight& light, float3& position , float3& direction)
{
	float3 planeNormal = normalize(cross(light.plane_v1, light.plane_v2));
	float3 nv1 = normalize(light.plane_v1);
	float3 nv2 = normalize(light.plane_v2);
	
	//transfer position into the space of the light plane and plane normal
	float3 lp_position = make_float3(dot(position,nv1), dot(position,nv2), dot(position,planeNormal));
	float lp_position_len = sqrtf(dot(lp_position,lp_position));
	float3 lp_planeNormal = make_float3(0.0,0.0,1.0);
	
	//float3 lp_proj_position = lp_position - dot(lp_position, lp_planeNormal) * lp_planeNormal;
	
	float3 new_rad = make_float3(0.0, lp_position.y, lp_position.z);
	float new_rad_len = sqrtf(lp_position.y*lp_position.y + lp_position.z*lp_position.z);
	
	
	float3 conv_x = make_float3(nv1.x, nv2.x, planeNormal.x);
	float3 conv_y = make_float3(nv1.y, nv2.y, planeNormal.y);
	float3 conv_z = make_float3(nv1.z, nv2.z, planeNormal.z);
	//position =  make_float3(dot(new_pos,conv_x), 
	//						dot(new_pos,conv_y), 
	//						dot(new_pos,conv_z));
	
	
	//rotate direction
	//find angle between new_rad and lp Y axis

	float sinRotAng, cosRotAng;
	if(new_rad.y <0){
		sinRotAng = dot(normalize(new_rad), lp_planeNormal); 
		cosRotAng = sqrtf(1 - sinRotAng*sinRotAng);
	}
	else{
		sinRotAng = dot(normalize(new_rad), lp_planeNormal); 
		cosRotAng = -sqrtf(1 - sinRotAng*sinRotAng);
		//sinRotAng = (-1.0)*dot(normalize(new_rad), lp_planeNormal); 
		//cosRotAng = sqrtf(1 - sinRotAng*sinRotAng);
	}
	
    /*
	float3 lp_dir = make_float3(dot(direction,nv1), dot(direction,nv2), dot(position,planeNormal));
	lp_dir = make_float3(lp_dir.x, lp_dir.y*cosRotAng - lp_dir.z*sinRotAng, lp_dir.y*sinRotAng + lp_dir.z*cosRotAng);
	direction =  make_float3(dot(lp_dir,conv_x), 
							dot(lp_dir,conv_y), 
							dot(lp_dir,conv_z));
	*/
	float3 old_position = position;
	lp_position = make_float3(lp_position.x, lp_position.y*cosRotAng - lp_position.z*sinRotAng, lp_position.y*sinRotAng + lp_position.z*cosRotAng);
	position =  make_float3(dot(lp_position,conv_x), 
							dot(lp_position,conv_y), 
							dot(lp_position,conv_z));
							
	//cos of angle between old position vector and new position vector
	float cosTheta = dot(normalize(position),normalize(old_position)); 
	float sinTheta = sqrtf(1.0 - cosTheta*cosTheta);
	float3 new_dir;
	new_dir.x = dot(direction,make_float3(1.0,0.0,0.0));
	new_dir.y = dot(direction,make_float3(0.0,cosRotAng,-sinRotAng));
	new_dir.z = dot(direction,make_float3(0.0,sinRotAng,cosRotAng));

	//float3 lp_dir = make_float3(direction.x, direction.y*cosRotAng - direction.z*sinRotAng, direction.y*sinRotAng + direction.z*cosRotAng);
	//new_dir =  make_float3(dot(lp_dir,conv_x), 
	//						dot(lp_dir,conv_y), 
	//						dot(lp_dir,conv_z));
	direction = new_dir;
}

//VERIFIED
__device__ __inline__ float3 directionFromBin(int bin, int RES_DIR_BIN)
{
	float3 dir = make_float3(0,0,0);
	float tempdir = 0.0;
	int r = 0;
	for(int j=2; j>=0; j--){
		r = bin % RES_DIR_BIN;
		tempdir = ((float)r / (float)(RES_DIR_BIN)) * 2.0 - 1.0;
		tempdir = ((((float)(r+1) / (float)(RES_DIR_BIN)) * 2.0 - 1.0) + tempdir)/2.0;
		if(j == 2){
			dir.z = tempdir;
		}
		else if(j == 1){
			dir.x = tempdir;
		}
		else if(j == 0){
			dir.y = tempdir;
		}
		bin = bin / RES_DIR_BIN;
	}
	
	//convert to -1..+1
	//dir = dir*2.0-make_float3(1.0);
	//if(dir.x + dir.y+ dir.z > 0)
	//   rtPrintf("bin id = %d, (%f,%f,%f)\n",bin,dir.y,dir.x,dir.z);
	return normalize(dir);
}

__device__ __inline__ void findRenvelope(float3 point, float& upper, float & lower, float Rt, float Rg, int RES_R_BIN)
{
	float len_r = sqrtf(dot(point, point));
	float rt_rg = (Rt-Rg);
	float r_rg = (len_r-Rg);
	
    int lower_ind = floor((float)(r_rg/rt_rg)*(int)(RES_R_BIN-1));
    int upper_ind = ceil((float)(r_rg/rt_rg)*(int)(RES_R_BIN-1));
	lower = Rg + rt_rg* (float)lower_ind/(float)(RES_R_BIN-1);
	upper = Rg + rt_rg* (float)upper_ind/(float)(RES_R_BIN-1);
}
/*
__device__ __inline__ float posbin_volume(float3 point, float3 light_dir,  float Rt, float Rg, int RES_R_BIN, int RES_MU_S_BIN)
{
	float upperPosBin, lowerPosBin, upperR, lowerR;
	findPosBinEnvelope(point, upperPosBin, lowerPosBin, light_dir, RES_MU_S_BIN);
	findRenvelope(point, upperR, lowerR, Rt, Rg, RES_R_BIN);
	return 2*M_PI*(fabs(upperPosBin-lowerPosBin))*(upperR*upperR*upperR-lowerR*lowerR*lowerR)/3.0;	
}

__device__ __inline__ float posbin_volume(int ci_low, int si_low, float Rt, float Rg, int RES_R_BIN, int RES_MU_S_BIN)
{
	float upperPosBin, lowerPosBin, upperR, lowerR;
	
	lowerPosBin = (float)ci_low/(float)(RES_MU_S_BIN-1);
	upperPosBin = (float)(ci_low+1)/(float)(RES_MU_S_BIN-1);

	//lowerPosBin = sqrtf(1.0 - lowerPosBin_sin*lowerPosBin_sin);
	//upperPosBin = sqrtf(1.0 - upperPosBin_sin*upperPosBin_sin);

	lowerR = Rg + (Rt-Rg)* (float)si_low/(float)(RES_R_BIN-1);
	upperR = Rg + (Rt-Rg)* (float)(si_low+1)/(float)(RES_R_BIN-1);

	// take only a fraction of lowerR and upperR and then mult by 10^9
	lowerR /= 1000.0;
	upperR /= 1000.0; 
	//findPosBinEnvelope(point, upperPosBin, lowerPosBin, light_dir, RES_MU_S_BIN);
	//findRenvelope(point, upperR, lowerR, Rt, Rg, RES_R_BIN);
	return 2*M_PI*(fabs(upperPosBin-lowerPosBin))*(upperR*upperR*upperR-lowerR*lowerR*lowerR)*1000000000.0/3.0;	
}
*/

__device__ __inline__ float posbin_volume(int ci_low, int si_low, float Rt, float Rg, int RES_R_BIN, int RES_MU_S_BIN)
{

	float upperPosBin_sin, lowerPosBin_sin, upperR, lowerR;
	float lowerPosBin, upperPosBin;
	lowerPosBin = (float)ci_low/(float)(RES_MU_S_BIN-1);
	upperPosBin = (float)(ci_low+1)/(float)(RES_MU_S_BIN-1);

	lowerPosBin_sin = sqrtf(1.0 - lowerPosBin*lowerPosBin);
	upperPosBin_sin = sqrtf(1.0 - upperPosBin*upperPosBin);

	lowerR = Rg + (Rt-Rg)* (float)si_low/(float)(RES_R_BIN-1);
	upperR = Rg + (Rt-Rg)* (float)(si_low+1)/(float)(RES_R_BIN-1);

	//float res = 2.0*M_PI*(upperPosBin_sin-lowerPosBin_sin)*(upperR*upperR*upperR - lowerR*lowerR*lowerR)/(3.0*2.0);
	//return res;

	float theta_lower = acosf(lowerPosBin_sin);
	float theta_upper = acosf(upperPosBin_sin);
	float R_mid = (lowerR+upperR)/2.0 * cosf((theta_upper+theta_lower)/2.0);
	float res = 2.0*M_PI*R_mid*((theta_upper-theta_lower)/2.0)*(upperR*upperR - lowerR*lowerR);
	return res;


/*
	float muS_sin = sqrtf(1 - muS*muS);
	int lower_ind = floor((float)((muS_sin)*(int)(RES_MU_S_BIN-1)));
    int upper_ind = ceil((float)((muS_sin)*(int)(RES_MU_S_BIN-1)));
*/
}

//__device__ __inline__ void rotate_with_sundir(	float3 x, float3 v, float3 & new_x, float3 &new_v) 
__device__ __inline__ void rotate_with_sundir(	float3 x, float3 & new_x, float3 light_dir) 
{
	// rotate x for changing sun direction
	
	float3 ldir = normalize(-light_dir);
	float3 init_sun=make_float3(0.0, 0.0, 1.0);
	float3 raxis = cross(init_sun, ldir);//rotation axis
	float cphi = dot(ldir,init_sun);				//cos phi where phi is the rotation angle
	float sphi = sqrtf(1 - cphi*cphi);	//sin phi
	//row 1
	float n00 = cphi+(1-cphi)*raxis.x*raxis.x;
	float n01 = (1-cphi)*raxis.x*raxis.y-raxis.z*sphi;
	float n02 = (1-cphi)*raxis.x*raxis.z+raxis.y*sphi;
	//row 2
	float n10 = (1-cphi)*raxis.x*raxis.y+raxis.z*sphi;
	float n11 = cphi+(1-cphi)*raxis.y*raxis.y;
	float n12 = (1-cphi)*raxis.y*raxis.z-raxis.x*sphi;
	//row 3
	float n20 = (1-cphi)*raxis.x*raxis.z-raxis.y*sphi;
	float n21 = (1-cphi)*raxis.y*raxis.z+raxis.x*sphi;
	float n22 = cphi+(1-cphi)*raxis.z*raxis.z;
	
	float3 rotmat_row_0 = make_float3(n00,n01,n02);
	float3 rotmat_row_1 = make_float3(n10,n11,n12);
	float3 rotmat_row_2 = make_float3(n20,n21,n22);
	new_x = make_float3(dot(rotmat_row_0,x),dot(rotmat_row_1,x),dot(rotmat_row_2,x));
	//new_v = make_float3(dot(rotmat_row_0,v),dot(rotmat_row_1,v),dot(rotmat_row_2,v));
	
}

__device__ __inline__ void swap(float &t0, float &t1)
{
	float temp = t0;
	t0 = t1;
	t1 = temp;
}


__device__ __inline__ bool Quadratic(float A, float B, float C, float & t0,
		float & t1) {
	// Find quadratic discriminant
	float discrim = B * B - 4.f * A * C;
	if (discrim < 0.) return false;
	float rootDiscrim = sqrtf(discrim);
	// Compute quadratic _t_ values
	float q;
	if (B < 0) q = -0.5f * (B - rootDiscrim);
	else       q = -0.5f * (B + rootDiscrim);
	t0 = q / A;
	t1 = C / q;
	if (t0 > t1) swap(t0, t1);
	return true;
}
__device__ __inline__ int cone_intersection(	float3 xx, float3 vv, float3 cone_data, 
												/* out*/ float & t0, /* out*/ float & t1, /* out */ float & thit, 
												float3 light_dir, const optix::Ray & ray, float Rg)
{
	float phi;
	float3 phit;
	float3 matrix_row_0 = make_float3(1.0,0.0,0.0);
	float3 matrix_row_1 = make_float3(0.0,1.0,0.0);
	float3 matrix_row_2 = make_float3(0.0,0.0,1.0);
	float3 x, v;
	rotate_with_sundir(xx,x, light_dir);
	rotate_with_sundir(vv,v, light_dir);
	
	// Transform _Ray_ to object space
	// Compute quadratic cone coefficients
	float radius = cone_data.x;
	float height = cone_data.y;
	float phiMax = cone_data.z;
	float k = radius / height;
	k = k*k;
	
	int coef = 1;
	
	float A = v.x * v.x + v.y * v.y -
		k * v.z * v.z;
	float B = 2 * (v.x * x.x + v.y * x.y -
		k * v.z * (x.z*coef) );
	float C = x.x * x.x + x.y * x.y -
		k * (x.z) * (x.z*coef);
	
	
	
	// Solve quadratic equation for _t_ values
	//float t0, t1;
	if (Quadratic(A, B, C, t0, t1)){
		// Compute intersection distance along ray
		if (t0 > ray.tmax || t1 < ray.tmin)
			return -1;
		
		thit = t0;
		if (t0 < ray.tmin) {
			thit = t1;
			if (thit > ray.tmax) 
				return -1;
		}
		// Compute cone inverse mapping
		phit = x + v * thit;
		if(phit.z*phit.z + phit.x*phit.x + phit.y*phit.y< (Rg)*(Rg))
			return -1;
		phi = atan2f(phit.y, phit.x);
		if (phi < 0.) phi += 2.f*M_PI;
		// Test cone intersection against clipping parameters
		if (phit.z < 0 || phit.z > height || phi > phiMax) {
			if (thit == t1) 
				return -1;
			thit = t1;
			if (t1 > ray.tmax) 
				return -1;
			// Compute cone inverse mapping
			phit = x + v*thit;
			phi = atan2f(phit.y, phit.x);
			if (phi < 0.) 
				phi += 2.f*M_PI;
			if (phit.z < 0 || phit.z > height || phi > phiMax)
				return -1;
		}
		// Find parametric representation of cone hit
		float lu = phi / phiMax;
		float lv = phit.z / height;
		// Compute cone \dpdu and \dpdv
		float3 dpdu = make_float3(-phiMax * phit.y, phiMax * phit.x, 0);
		float3 dpdv = make_float3(-phit.x / (1.f - lv),
					-phit.y / (1.f - lv), height);
		// Compute cone \dndu and \dndv
		float3 d2Pduu = -phiMax * phiMax *
						make_float3(phit.x, phit.y, 0.);
		float3 d2Pduv = phiMax / (1.f - lv) *
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
		
		return 1;
	}
}


__device__ __inline__ int cone_next_boundary_dist(int pm_index, float3 x, float3 v, float curr_dist, 
												  float muS, float & out_dist, int & out_ind,
												  float3 light_dir, const optix::Ray & ray, float Rg, float Rt, int RES_MU_S_BIN)
{
   float root0 = -1.0;
   float root1 = -1.0;
   float thit = -1.0;
   float3 cone_data;
   //cone_data.x = cosf(M_PI*muS)*(Rt); 
   //cone_data.y = sinf(M_PI*muS)*(Rt);
   float cosC = muS;
   float sinC = sqrtf(1 - muS*muS);
   cone_data.x = cosC*(Rt);//+400); 
   cone_data.y = sinC*(Rt);//+400);
   cone_data.z = 2*M_PI;
   int result = cone_intersection(x, v, cone_data, root0, root1, thit, light_dir, ray, Rg);
   if(thit > 0.0 && result == 1){
		out_dist = thit;
		out_ind = floor((float)((sinC)*(int)(RES_MU_S_BIN-1)));
		return 0;
   }
   else{
	   out_dist = 0.0;
	   out_ind = -1;	
	   return -1;
   }
}




// high level cones and spheres inersection and gathering functions
/*
__device__ __inline__ void foo1(int si_dir, int ci_dir,
								  float next_t_hit_s, float next_t_hit_c, 
								  float total_dist, float final_dist,
								  float3& color, float3 p,
								  bool b_ci_dir, bool b_si_dir,
								  int & si, int & ci,
								  float3 orig_p, float3 v,
								  float Rg, float Rt, int RES_MU_S_BIN, 
								  int RES_R_BIN, int RES_DIR_BIN_TOTAL, 
								  float3 light_dir, float3 & total_attn)
{
	float dist = 0.0;
	float ci_temp;
	int num_photons= 0;
	int index;
	float3 attn = make_float3(1.0);

	if(si_dir != 0 && ci_dir != 0){
		dist = 0.0;
		if((next_t_hit_s <= next_t_hit_c && next_t_hit_s > 0.0) || (next_t_hit_c == 0.0)){
			total_dist += next_t_hit_s;
			if(next_t_hit_c >= next_t_hit_s){
				next_t_hit_c = next_t_hit_c - next_t_hit_s;
			}
			
			si = si + si_dir;
			dist = next_t_hit_s;
			next_t_hit_s = 0.0;
		}
		else if((next_t_hit_c < next_t_hit_s && next_t_hit_c > 0.0) || (next_t_hit_s == 0.0)){
			total_dist += next_t_hit_c;
			if(next_t_hit_s >= next_t_hit_c){
				next_t_hit_s = next_t_hit_s - next_t_hit_c;
			}
			
			ci = ci + ci_dir;
			dist = next_t_hit_c;
			next_t_hit_c = 0.0;
		}
		else{
			//color  = make_float3(0.1,0.0,0.0);;
			//return;
		}
		
		//attn = analyticTransmittance(length(p), mu, dist, HR, HM, Rg, betaR, betaMEx);
		//total_attn = total_attn * attn;//(1-attn);
		
		ci_temp =  findPosBin((orig_p + (final_dist/2.0)*v), light_dir, RES_MU_S_BIN);
		
		int i_si = 0;
		int i_ci = 0;
		if(b_si_dir){
			i_si = (si == 1) ? 0 : si - 2;
		}
		else{
			i_si = si - si_dir;
		}
		
		if(b_ci_dir){
			i_ci = (ci == 1) ? 0 : ci - 2;
		}
		else{
			i_ci = ci - ci_dir;
		}
		
		if(ci - ci_dir == 0)
		{
			index = ((0)*RES_R_BIN + (i_si))*RES_DIR_BIN_TOTAL;
			if(index <= ((RES_MU_S_BIN-1)*RES_R_BIN + RES_R_BIN-1)*RES_DIR_BIN_TOTAL){
				color  += total_attn* photon_gather_contib_ext(p, p+dist*v, v, index, num_photons, 0, i_si);
				
			}
			else{
				color  += total_attn*make_float3(1.0,1.0,1.0);
			}
		}
		else if(ci == -1 && (ci - ci_dir) != 0){
		    index = (int)(ci_temp*RES_R_BIN + (i_si))*RES_DIR_BIN_TOTAL; 
			color  += total_attn*photon_gather_contib_ext(p, p+dist*v, v, index, num_photons, ci_temp, i_si);
			//rec.attenuated_Kd  += total_attn*make_float3(0.1,0.1,0.0);
			
		}
		else{
			//index = (int)(ci_temp*RES_R_BIN + (i_si))*RES_DIR_BIN_TOTAL;
			index = ((i_ci)*RES_R_BIN + (i_si))*RES_DIR_BIN_TOTAL;
			if(index <= ((RES_MU_S_BIN-1)*RES_R_BIN + RES_R_BIN-1)*RES_DIR_BIN_TOTAL){
				color  += total_attn* photon_gather_contib_ext(p, p+dist*v, v, index, num_photons, i_ci, i_si);
				
			}
			else{
				color  += total_attn*make_float3(1.0,1.0,1.0);
			}
		
			//rec.attenuated_Kd  += total_attn*make_float3(0.0,0.1,0.1);
		}
		
		
		attn = tranAtmNew(r, mu, trans_texture, p+dist*v, v,  Rg, Rt ,RL, betaR, betaMEx,
				HR, HM, TRANSMITTANCE_INTEGRAL_SAMPLES);
		total_attn = total_attn * attn;//(1-attn);
		
		p += dist*v;
		r = length(p);
		mu = dot(p, v)/r;
	
	}
}
*/

__device__ __inline__ int sphere_planet_atm_boundary_dist_ext(int pm_index, float3 light_dir, 
														  float3 x, float3 v, float t, int & out_ind, 
														  float & out_dist, bool bSpace, 
														  float Rg, float Rt, int RES_R_BIN)
{
	float root1 =0.0;
	float root2 =0.0;
	int res = 0;
	res = sphere_intersection(light_dir, x, v, RES_R_BIN-1, root1, root2, out_ind, Rt, Rg, RES_R_BIN);
	if(res == -1){
		res = sphere_intersection(light_dir, x, v, RES_R_BIN-2, root1, root2, out_ind, Rt, Rg, RES_R_BIN);
		if(res == -1){
			return 0;
		}
	}
		
	if(t >= 0){
		out_dist = (root1 > root2) ? root2 : root1;
		res = -1;
	}
	else{ // t<0
		if(root1 < 0.0 && root2 < 0.0){
			out_dist = (fabs(root1) > fabs(root2)) ? root2 : root1;
			res = -1;
		}	
		else{
			out_dist = 0.0;
			res = 0;
		}
	}
	return res;
}

__device__ __inline__ int sphere_planet_atm_boundary_dist(int pm_index, float3 light_dir, 
														  float3 x, float3 v, int & out_ind, 
														  float & out_dist, bool bSpace, 
														  float Rg, float Rt, int RES_R_BIN)
{
	float root1d, root2d;
	float root1u, root2u;
	int ind_down = 0;
	int ind_up = 0;
	int res_down = -1;
	int res_up = -1;
	res_down = sphere_intersection(light_dir, x, v, 0, root1d, root2d, ind_down, Rt, Rg, RES_R_BIN);
	res_up = sphere_intersection(light_dir, x, v, RES_R_BIN-1, root1u, root2u, ind_up, Rt, Rg, RES_R_BIN);
	
	
	if(res_down == -1 && res_up == -1)
	{
		return -2;
	}
	
	if(root1u < root2u)
		root1u = root2u;
	
	if(root1d > root2d)
		root1d = root2d;
	float arr[2] = {root1u, root1d};	
	
	out_dist = 0.0;
	float minpos = 0.0; // minimum positive
	int minpos_ind = 0;
	for(int i=0; i < 2; i++){
		if(arr[i] > 0.0 && ((minpos == 0.0) || (minpos < arr[i])) ){
			minpos = arr[i];
			if(i < 1) 
				minpos_ind = ind_up;
			else
				minpos_ind = ind_down;
		}
		if(arr[i] >= 0.5 && out_dist == 0.0){
			out_dist = arr[i];
			if(i < 1) 
				out_ind = ind_up;
			else
				out_ind = ind_down;
		}
		else if(arr[i] >= 0.5 && arr[i] < out_dist){
			out_dist = arr[i];
			if(i < 1) 
				out_ind = ind_up;
			else
				out_ind = ind_down;
		}
	}
	
	if(out_dist == 0.0 && ind_down < RES_R_BIN && minpos == 0.0){
	//	rtPrintf("pm_index %d, t=%f ,DOWN: %d, (1):%f, (2):%f, UP: %d, (1):%f, (2):%f\n",pm_index, t, ind_down, root1d, root2d, ind_up, root1u, root2u);
		return -1;
	}	
	if(out_dist == 0.0 && ind_down < RES_R_BIN && minpos > 0.0){
		out_dist = minpos;
		out_ind = minpos_ind;
		
	}
	//rtPrintf("pm_index %d,DOWN: %d, (1):%f, (2):%f, UP: %d, (1):%f, (2):%f\n",pm_index, ind_down, root1d, root2d, ind_up, root1u, root2u);
	return 0;
}

#endif