
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

//-------------------------------------------------------------------------------
//
//  ppm.cpp -- Progressive photon mapping scene
//
//-------------------------------------------------------------------------------


#define RT_PASS_ON

#if defined(__APPLE__)
#  include <GLUT/glut.h>
#  define GL_FRAMEBUFFER_SRGB_EXT           0x8DB9
#  define GL_FRAMEBUFFER_SRGB_CAPABLE_EXT   0x8DBA
#else
#  include <GL/glew.h>
#  if defined(_WIN32)
#    include <GL/wglew.h>
#  endif
#  include <GL/glut.h>
#endif

#include "photonmap.h"
#include "pas_api.h"
#include <cutil.h>
#include <ObjLoader.h>
#include "ObjLoaderEx.h"

#include <sstream>
#include <string>
using namespace std;
using namespace optix;

unsigned int   ProgressivePhotonScene::_texId                = 0;

//OpenGL init
void initializeGL(int argc, char **argv);


// Finds the smallest power of 2 greater or equal to x.
inline unsigned int pow2roundup(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x+1;
}

inline float max(float a, float b)
{
	return a > b ? a : b;
}

inline RT_HOSTDEVICE int max_component(float3 a)
{
	if(a.x > a.y) {
		if(a.x > a.z) {
			return 0;
		} else {
			return 2;
		}
	} else {
		if(a.y > a.z) {
			return 1;
		} else {
			return 2;
		}
	}
}

float3 sphericalToCartesian( float theta, float phi )
{
	float cos_theta = cosf( theta );
	float sin_theta = sinf( theta );
	float cos_phi = cosf( phi );
	float sin_phi = sinf( phi );
	float3 v;
	v.x = cos_phi * sin_theta;
	v.z = sin_phi * sin_theta;
	v.y = cos_theta;
	return v;
}


const unsigned int ProgressivePhotonScene::sqrt_samples_per_pixel = 1;
const unsigned int ProgressivePhotonScene::WIDTH  = 768u;
const unsigned int ProgressivePhotonScene::HEIGHT = 768u;
const unsigned int ProgressivePhotonScene::MAX_PHOTON_COUNT = 2u;
const unsigned int ProgressivePhotonScene::MAX_VOL_PHOTON_COUNT = 256u;
const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH = 64u;//64u;//64u;
const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT = 64u;//64u;
//const unsigned int ProgressivePhotonScene::VOL_PHOTON_LAUNCH_WIDTH = 256u;
//const unsigned int ProgressivePhotonScene::VOL_PHOTON_LAUNCH_HEIGHT = 256u;
const unsigned int ProgressivePhotonScene::NUM_PHOTONS = (ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH *
														  ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT *
														  ProgressivePhotonScene::MAX_PHOTON_COUNT);

const unsigned int ProgressivePhotonScene::NUM_VOL_PHOTONS = (ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH *
															  ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT *
															  ProgressivePhotonScene::MAX_VOL_PHOTON_COUNT);
float3 Cross(const float3 &v1, const float3 &v2) {
	return make_float3((v1.y * v2.z) - (v1.z * v2.y),
		(v1.z * v2.x) - (v1.x * v2.z),
		(v1.x * v2.y) - (v1.y * v2.x));
}

void CoordinateSystem(const float3 &v1, float3 *v2, float3 *v3) {
	if (fabsf(v1.x) > fabsf(v1.y)) {
		float invLen = 1.f / sqrtf(v1.x*v1.x + v1.z*v1.z);
		*v2 = make_float3(-v1.z * invLen, 0.f, v1.x * invLen);
	}
	else {
		float invLen = 1.f / sqrtf(v1.y*v1.y + v1.z*v1.z);
		*v2 = make_float3(0.f, v1.z * invLen, -v1.y * invLen);
	}
	*v3 = Cross(v1, *v2);
}

// Create ONB from normal.  Resulting W is Parallel to normal
void createONB( const optix::float3& n,
			   optix::float3& U,
			   optix::float3& V,
			   optix::float3& W )
{
	using namespace optix;

	W = normalize( n );
	U = cross( W, make_float3( 0.0f, 1.0f, 0.0f ) );
	if ( fabsf( U.x) < 0.001f && fabsf( U.y ) < 0.001f && fabsf( U.z ) < 0.001f  )
		U = cross( W, make_float3( 1.0f, 0.0f, 0.0f ) );
	U = normalize( U );
	V = cross( W, U );
}

void ProgressivePhotonScene::initExperimentViewProj()
{
	float data[16] = {1.0, 0, 0, 0,
		0.0, 1.0, 0.0, 0,
		0.0, 0.0, 1.0, 0,
		0, 0, 0, 1};
	float data1[16] = {1.0, 0, 0, 0,
		0.0, 1.0, 0.0, 0,
		0.0, 0.0, 1.0, 0,
		0, 0, 0, 1};

	_iviewf = Matrix4x4(data);
	_iproj = Matrix4x4(data1); 
	_cam_pos = make_float4(0.0,0.0,0.0,0.0);
}

bool ProgressivePhotonScene::keyPressed(unsigned char key, int x, int y)
{
	float step_size = 0.01f;
	bool light_changed = false;;
	switch (key)
	{
	case 'd':
		_light_phi += step_size;
		if( _light_phi >  M_PIf * 2.0f ) _light_phi -= M_PIf * 2.0f;
		light_changed = true;
		break;
	case 'a':
		_light_phi -= step_size;
		if( _light_phi <  0.0f ) _light_phi += M_PIf * 2.0f;
		light_changed = true;
		break;
	case 's':
		std::cerr << "new theta: " << _light_theta + step_size << " max: " << M_PIf / 2.0f  << std::endl;
		_light_theta = fminf( _light_theta + step_size, M_PIf / 2.0f );
		light_changed = true;
		break;
	case 'w':
		std::cerr << "new theta: " << _light_theta - step_size << " min: 0.0f " << std::endl;
		_light_theta = fmaxf( _light_theta - step_size, 0.0f );
		light_changed = true;
		break;
	case 'v': {
		_camera_index = (_camera_index + 1) % 3;
		GLUTDisplay::setCamera( _camera[_camera_index] );
		m_camera_changed = true;
		return true;
			  } break;
	case 'p': {
		_update_pmap = !_update_pmap;
		m_camera_changed = true;
		std::cerr << "init photon tracing: "  << ((_update_pmap)? "on" : "off") <<  std::endl;
		light_changed = true;
			  }break;
	case 'g': // activate gather
		_full_gather = !_full_gather;
		m_camera_changed = true;
		light_changed = true;
		std::cerr << "full gather: "  << ((_full_gather)? "on" : "off") <<  std::endl;
		break;
	case 'e': //exposure
		_exposure *= 1.1;
		std::cerr << "exposure: " << _exposure << std::endl;
		light_changed = true;
		break;
	case 'E' :
		_exposure /= 1.1;
		std::cerr << "exposure: " << _exposure << std::endl;
		light_changed = true;
		break;
	case 'z' :
		_singleOnly = (_singleOnly + 1) % 3;
		std::cerr << "rendering mode: ";
		if(_singleOnly == 0)
			std::cerr << "single + multiple";
		else if(_singleOnly == 1)
			std::cerr << "single only";
		else if(_singleOnly == 2)
			std::cerr << "multiple only";
		std::cerr << std::endl;
		light_changed = true;
		break;
	case 'h' :
		_hdrOn = !_hdrOn;
		m_camera_changed = true;
		light_changed = true;
		std::cerr << "high dynamic range: "  << ((_hdrOn)? "on" : "off") <<  std::endl;
	case 'o' :
		//_sphere_accel->markDirty();
		break;
	}

	if( light_changed /*&& !_cornell_box*/ ) {
		//std::cerr << " theta: " << _light_theta << "  phi: " << _light_phi << std::endl;
		if(_distant_test){
			_light.is_area_light = 2;
			_light.anchor = _light.worldCenter + make_float3(0.0,0.0,-RS);
			createONB(_light.anchor, _light.v1, _light.v2,  _light.v3);
			_light.direction = normalize(make_float3(0.0,0.0,400.0));//normalize(_light.worldCenter + _light.anchor );
			_light.power  = make_float3( 0.9e6f, 0.9e6f, 0.9e6f );

			_light.plane_v1 = normalize(_light.anchor - _light.worldCenter);
			_light.plane_v2 = normalize(cross(_light.plane_v1,make_float3(1.0,0.0,0.0)));
		}
		else{
			_light.position  = 1000.0f * sphericalToCartesian( _light_theta, _light_phi );
			_light.direction = normalize( make_float3( 0.0f, 0.0f, 0.0f )  - _light.position );
		}
		m_context["light"]->setUserData( sizeof(PPMLight), &_light );
		signalCameraChanged(); 
		return true;
	}

	return false;
}


optix::TextureSampler ProgressivePhotonScene::loadTexture3f( float* data, 
															const unsigned int width, 
															const unsigned int height,
															const unsigned int depth,
															const unsigned int dim,
															int test,
															bool debug)
{
	// Create tex sampler and populate with default values
	optix::TextureSampler sampler = m_context->createTextureSampler();
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	sampler->setWrapMode( 0, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setWrapMode( 1, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setWrapMode( 2, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
	sampler->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
	sampler->setMaxAnisotropy( 1.0f );
	sampler->setMipLevelCount( 1u );
	sampler->setArraySize( 1u );

	// Read in HDR, set texture buffer to empty buffer if fails

	const unsigned int nx = width;
	const unsigned int ny = height;
	const unsigned int nz = depth;

	// Create buffer and populate with HDR data
	optix::Buffer buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny, nz );
	float* buffer_data = static_cast<float*>( buffer->map() );

	
	for ( unsigned int i = 0; i < nx; ++i ) {
		for ( unsigned int j = 0; j < ny; ++j ) {
			for ( unsigned int k = 0; k < nz; ++k) {
				//unsigned int hdr_index = ( (ny-j-1)*nx + i )*4;
				//unsigned int buf_index = ( (j     )*nx + i )*4;
				//unsigned int hdr_index = (k + i*nz + (ny-j-1)*nx*nz)*dim;//(k + i*nz + j*nx*nz)*dim;
				//unsigned int buf_index = (k + i*nz + (ny-j-1)*nx*nz)*dim;//(k + i*nz + j*nx*nz)*dim;
				
			
				//unsigned int hdr_index = ( (k)*nx*ny + ny-j-1 + (nx-i-1)*ny )*dim;
				//unsigned int buf_index = ( (k)*nx*ny + j + i*ny )*dim;
				unsigned int hdr_index =  0;
				unsigned int buf_index =  0;
				if(test == -1){
					hdr_index =  (k + (j)*nz + (nx-i-1)*ny*nz)*dim;
					buf_index =  (k + (j)*nz + (nx-i-1)*ny*nz)*4;
				}
				else{ // testing indeces
					hdr_index =  (k*nx*ny + (j) + (nx-i-1)*ny)*dim;
					buf_index =  (k*nx*ny + (j) + (nx-i-1)*ny)*4;
				}
				buffer_data[ buf_index + 0 ] = data[ hdr_index + 0 ];
				buffer_data[ buf_index + 1 ] = data[ hdr_index + 1 ];
				buffer_data[ buf_index + 2 ] = data[ hdr_index + 2 ];
				if(dim == 4){
					buffer_data[ buf_index + 3 ] = data[ hdr_index + 3 ];
				}
				else if(dim == 3){
					buffer_data[ buf_index + 3 ] = 0.0;
				}
			}
		}
	}

	if(debug){
		FILE* fp = fopen("c:\\SVN\\texttest.txt","w");
		for ( unsigned int i = 0; i < nx; ++i ) {
			for ( unsigned int j = 0; j < ny; ++j ) {
				for ( unsigned int k = 0; k < nz; ++k) {
					unsigned int hdr_index =  (k + (j)*nz + (nx-i-1)*ny*nz)*dim;
					fprintf(fp,"%f,%f,%f" ,data[ hdr_index + 0 ],data[ hdr_index + 1 ],data[ hdr_index + 2 ]);
					if(dim == 4){
						fprintf(fp,",%f",data[ hdr_index + 3 ]);
					}
					fprintf(fp,"\n");
				}
			}
		}
		fclose(fp);
	}

	buffer->unmap();

	sampler->setBuffer( 0u, 0u, buffer );
	sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );

	return sampler;
}

optix::TextureSampler ProgressivePhotonScene::loadTexture2uc( unsigned char* data, 
															 const unsigned int width, 
															 const unsigned int height)
{
	// Create tex sampler and populate with default values
	optix::TextureSampler sampler = m_context->createTextureSampler();
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	sampler->setWrapMode( 0, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setWrapMode( 1, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setWrapMode( 2, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
	sampler->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
	sampler->setMaxAnisotropy( 1.0f );
	sampler->setMipLevelCount( 1u );
	sampler->setArraySize( 1u );

	// Read in HDR, set texture buffer to empty buffer if fails

	const unsigned int nx = width;
	const unsigned int ny = height;

	// Create buffer and populate with HDR data
	optix::Buffer buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, nx, ny );
	unsigned char* buffer_data = static_cast<unsigned char*>( buffer->map() );

	for ( unsigned int i = 0; i < nx; ++i ) {
		for ( unsigned int j = 0; j < ny; ++j ) {

			//unsigned int hdr_index = ( (ny-j-1)*nx + i )*4;
			//unsigned int buf_index = ( (j     )*nx + i )*4;
			unsigned int hdr_index = (i+j*nx)*4;	
			unsigned int buf_index = (i+j*nx)*4;	

			buffer_data[ buf_index + 0 ] = data[ hdr_index + 0 ];
			buffer_data[ buf_index + 1 ] = data[ hdr_index + 1 ];
			buffer_data[ buf_index + 2 ] = data[ hdr_index + 2 ];
			buffer_data[ buf_index + 3 ] = data[ hdr_index + 3 ];
		}
	}

	buffer->unmap();

	sampler->setBuffer( 0u, 0u, buffer );
	sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );

	return sampler;
}

optix::TextureSampler ProgressivePhotonScene::loadTexture2ucext( unsigned char* data, 
															 const unsigned int width, 
															 const unsigned int height,
														  const unsigned int dim)
{
	// Create tex sampler and populate with default values
	optix::TextureSampler sampler = m_context->createTextureSampler();
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	sampler->setWrapMode( 0, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setWrapMode( 1, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setWrapMode( 2, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
	sampler->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
	sampler->setMaxAnisotropy( 1.0f );
	sampler->setMipLevelCount( 1u );
	sampler->setArraySize( 1u );

	// Read in HDR, set texture buffer to empty buffer if fails

	const unsigned int nx = width;
	const unsigned int ny = height;

	// Create buffer and populate with HDR data
	optix::Buffer buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, nx, ny );
	unsigned char* buffer_data = static_cast<unsigned char*>( buffer->map() );

	for ( unsigned int i = 0; i < nx; ++i ) {
		for ( unsigned int j = 0; j < ny; ++j ) {

			//unsigned int hdr_index = ( (ny-j-1)*nx + i )*4;
			//unsigned int buf_index = ( (j     )*nx + i )*4;
			unsigned int hdr_index = (i+j*nx)*dim;	
			unsigned int buf_index = (i+j*nx)*4;	
			//unsigned int hdr_index = ( (ny -j-1)*nx + i )*dim;
			//unsigned int buf_index = ((ny-1-j)*nx + i )*4;

			buffer_data[ buf_index + 0 ] = data[ hdr_index + 0 ];
			buffer_data[ buf_index + 1 ] = data[ hdr_index + 1 ];
			buffer_data[ buf_index + 2 ] = data[ hdr_index + 2 ];
			buffer_data[ buf_index + 3 ] = data[ hdr_index + 3 ];
		}
	}

	buffer->unmap();

	sampler->setBuffer( 0u, 0u, buffer );
	sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );

	return sampler;
}
optix::TextureSampler ProgressivePhotonScene::loadTexture2fext( float* data, 
							    						  const unsigned int width, 
														  const unsigned int height,
														  const unsigned int dim,
														  bool debug)
{
	// Create tex sampler and populate with default values
	optix::TextureSampler sampler = m_context->createTextureSampler();
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	sampler->setWrapMode( 0, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setWrapMode( 1, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setWrapMode( 2, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
	sampler->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
	sampler->setMaxAnisotropy( 1.0f );
	sampler->setMipLevelCount( 1u );
	sampler->setArraySize( 1u );

	// Read in HDR, set texture buffer to empty buffer if fails

	const unsigned int nx = width;
	const unsigned int ny = height;

	// Create buffer and populate with HDR data
	optix::Buffer buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny );
	float* buffer_data = static_cast<float*>( buffer->map() );

	float3 maxdata = make_float3(0.0);
	for ( unsigned int i = 0; i < nx; ++i ) {
		for ( unsigned int j = 0; j < ny; ++j ) {

			unsigned int hdr_index = ( (ny -j-1)*nx + i )*dim;
			//unsigned int hdr_index = ( (nx -i-1)*ny + j )*dim;
			//unsigned int buf_index = ((ny-1-j)*nx + i )*4;
			//unsigned int hdr_index = (i*ny+j)*dim;	
			//unsigned int buf_index = (i*(ny-j-1)+j)*4;	
			//unsigned int hdr_index = (j*nx+i)*dim;	
			//unsigned int buf_index = (j*nx+i)*4;	
			unsigned int buf_index = ((ny-1-j)*nx + i )*4;

			if(maxdata.x < data[ hdr_index])
				maxdata.x = data[ hdr_index];
			if(maxdata.y < data[ hdr_index+1])
				maxdata.y = data[ hdr_index+1];
			if(maxdata.z < data[ hdr_index+2])
				maxdata.z = data[ hdr_index+2];
			buffer_data[ buf_index + 0 ] = data[ hdr_index + 0 ];
			buffer_data[ buf_index + 1 ] = data[ hdr_index + 1 ];
			buffer_data[ buf_index + 2 ] = data[ hdr_index + 2 ];
			buffer_data[ buf_index + 3 ] = dim > 3 ? data[ hdr_index + 3 ] : 1.0;//data[ hdr_index + 3 ];
		}
	}

	if(debug){
		unsigned char *a_f =  (unsigned char*)malloc(sizeof(unsigned char)*3*width*height);
		for(int i = 0; i < nx; i++){
			for(int j = 0; j < ny; j++){
				//a_f[x*TRANSMITTANCE_H*3+y*3] = (unsigned char)((255)*(fx[x*TRANSMITTANCE_H+y]/max_x));
				//a_f[x*TRANSMITTANCE_H*3+y*3+1] = (unsigned char)((255)*(fy[x*TRANSMITTANCE_H+y]/max_y));
				//a_f[x*TRANSMITTANCE_H*3+y*3+2] = (unsigned char)((255)*(fz[x*TRANSMITTANCE_H+y]/max_z));

				//unsigned int hdr_index = ( (ny-j-1)*nx + i )*dim;
				//unsigned int buf_index = ( (j     )*nx + i )*3;
				//unsigned int hdr_index = ( (ny -j-1)*nx + i )*dim;
				unsigned int hdr_index = ( (ny -j-1)*nx + i )*dim;
				//unsigned int buf_index = (j*nx+i)*3;	
				unsigned int buf_index = ((ny-1-j)*nx + i )*3;
				//unsigned int hdr_index = (j*nx+i)*dim;	

				a_f[buf_index] = (unsigned char)((255)*(data[ hdr_index]/maxdata.x));
				a_f[buf_index+1] = (unsigned char)((255)*(data[ hdr_index+1]/maxdata.y));
				a_f[buf_index+2] = (unsigned char)((255)*(data[ hdr_index+2]/maxdata.z));

				//_transmittance[x*3+y*TRANSMITTANCE_W*3] = a_f[x*TRANSMITTANCE_H*3+y*3]; 
				//_transmittance[x*3+y*TRANSMITTANCE_W*3+1] = a_f[x*TRANSMITTANCE_H*3+y*3+1]; 
				//_transmittance[x*3+y*TRANSMITTANCE_W*3+2] = a_f[x*TRANSMITTANCE_H*3+y*3+2]; 
			}
		}
		//cutSavePPMub( "c:\\SVN\\transm_gpu.ppm", a_f, width, height);
		free(a_f);
	}
	buffer->unmap();

	sampler->setBuffer( 0u, 0u, buffer );
	sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );

	return sampler;
}

optix::TextureSampler ProgressivePhotonScene::loadTexture2f( float* data, 
															const unsigned int width, 
															const unsigned int height)
{
	// Create tex sampler and populate with default values
	optix::TextureSampler sampler = m_context->createTextureSampler();
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	sampler->setWrapMode( 0, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setWrapMode( 1, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setWrapMode( 2, RT_WRAP_CLAMP_TO_EDGE );
	sampler->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
	sampler->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
	sampler->setMaxAnisotropy( 1.0f );
	sampler->setMipLevelCount( 1u );
	sampler->setArraySize( 1u );

	// Read in HDR, set texture buffer to empty buffer if fails

	const unsigned int nx = width;
	const unsigned int ny = height;

	// Create buffer and populate with HDR data
	optix::Buffer buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny );
	float* buffer_data = static_cast<float*>( buffer->map() );

	for ( unsigned int i = 0; i < nx; ++i ) {
		for ( unsigned int j = 0; j < ny; ++j ) {

			//unsigned int hdr_index = ( (ny-j-1)*nx + i )*4;
			//unsigned int buf_index = ( (j     )*nx + i )*4;
			unsigned int hdr_index = (i+j*nx)*3;	
			unsigned int buf_index = (i+j*nx)*4;	

			buffer_data[ buf_index + 0 ] = data[ hdr_index + 0 ];
			buffer_data[ buf_index + 1 ] = data[ hdr_index + 1 ];
			buffer_data[ buf_index + 2 ] = data[ hdr_index + 2 ];
			buffer_data[ buf_index + 3 ] = 1.0;//data[ hdr_index + 3 ];
		}
	}

	buffer->unmap();

	sampler->setBuffer( 0u, 0u, buffer );
	sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );

	return sampler;
}

#ifdef OLD_CODE
void ProgressivePhotonScene::signalMouseCoordChanged(Matrix4x4& iviewf, Matrix4x4& iproj, float4 cam_pos)
{
	//update view matrix and send it to CUDA kernel
	cout << "In signalMouseCoordChanged()\n";
	_iviewf = iviewf;
	_iproj = iproj;
	_cam_pos = cam_pos;
	float vfov = 2.0 * atan(float(HEIGHT) / float(WIDTH) * tan(80.0 / 180 * M_PI / 2.0)) / M_PI * 180;

	m_context["iview_matrix_row_0"]->setFloat(iviewf[0*4+0], iviewf[0*4+1], iviewf[0*4+2], iviewf[0*4+3]);
	m_context["iview_matrix_row_1"]->setFloat(iviewf[1*4+0], iviewf[1*4+1], iviewf[1*4+2], iviewf[1*4+3]);
	m_context["iview_matrix_row_2"]->setFloat(iviewf[2*4+0], iviewf[2*4+1], iviewf[2*4+2], iviewf[2*4+3]);
	m_context["iview_matrix_row_3"]->setFloat(iviewf[3*4+0], iviewf[3*4+1], iviewf[3*4+2], iviewf[3*4+3]);

	m_context["iproj_matrix_row_0"]->setFloat(iproj[0*4+0], iproj[0*4+1], iproj[0*4+2], iproj[0*4+3]);
	m_context["iproj_matrix_row_1"]->setFloat(iproj[1*4+0], iproj[1*4+1], iproj[1*4+2], iproj[1*4+3]);
	m_context["iproj_matrix_row_2"]->setFloat(iproj[2*4+0], iproj[2*4+1], iproj[2*4+2], iproj[2*4+3]);
	m_context["iproj_matrix_row_3"]->setFloat(iproj[3*4+0], iproj[3*4+1], iproj[3*4+2], iproj[3*4+3]);

	m_context["new_eye"]->setFloat(cam_pos);

	bool light_changed = true;

	InitialCameraData camera_data = InitialCameraData( make_float3( cam_pos), // eye
		make_float3( 0.0f, 0.0f, 0.0f ),    // lookat
		make_float3( 0.0f, 1.0f,  0.0f ),       // up
		vfov );                                // vfov
	GLUTDisplay::setCamera(camera_data );


	if( light_changed /*&& !_cornell_box*/ ) {
		//std::cerr << " theta: " << _light_theta << "  phi: " << _light_phi << std::endl;
		if(_distant_test){
			_light.is_area_light = 2;
			//_light.anchor = make_float3( 278.0f, 248.6f, 279.5f);
			//_light.anchor = 1000.0f * sphericalToCartesian( _light_theta, _light_phi );

			//_light.anchor = _light.worldCenter + (_light.worldRadius*1.3) * sphericalToCartesian( _light_theta, _light_phi );
			_light.anchor = _light.worldCenter + make_float3(0.0,0.0,-400.0);
			//_light.anchor = make_float3( 278.0f, 274.0f, 558.0f);
			//CoordinateSystem(_light.anchor, &_light.v1, &_light.v2 );
			createONB(_light.anchor, _light.v1, _light.v2,  _light.v3);
			//_light.direction = make_float3( -1.0f, -1.0f, 1.0f);
			_light.direction = normalize(_light.worldCenter + _light.anchor );
			//_light.power  = make_float3( 0.7e6f, 0.7e6f, 0.7e6f );
			_light.power  = make_float3( 0.9e6f, 0.9e6f, 0.9e6f );
			//_light.power  = make_float3( 0.9f, 0.9f, 0.9f );
		}
		else{
			_light.position  = 1000.0f * sphericalToCartesian( _light_theta, _light_phi );
			_light.direction = normalize( make_float3( 0.0f, 0.0f, 0.0f )  - _light.position );
		}
		m_context["light"]->setUserData( sizeof(PPMLight), &_light );
		signalCameraChanged(); 
	}
}
#endif

void ProgressivePhotonScene::getHVAngles(float & vangle, float & hangle)
{
	vangle = M_PI/4;
	hangle = 0.0;//M_PI/4;
	//_angleIndex = _angleIndex % 4;

}

void ProgressivePhotonScene::signalScaleFactorChange(float sfactor)
{
	m_context["SCALE_FACTOR"]->setFloat(sfactor);
	signalCameraChanged();
}

void ProgressivePhotonScene::signalTestRtPassChange()
{
	if(_testRtPassLevel >0)
		_testRtPassLevel=0;
	else
		_testRtPassLevel=1;
	m_context["TEST_RT_PASS"]->setInt(_testRtPassLevel);
	signalCameraChanged();
}

void ProgressivePhotonScene::signalLightChanged(float3 new_light_dir)
{
	_light.is_area_light = 2;
	_light.anchor = _light.worldCenter - RS*new_light_dir;
	createONB(_light.anchor, _light.v1, _light.v2,  _light.v3);
	_light.direction = normalize(new_light_dir);//normalize(_light.worldCenter + _light.anchor );
	_light.power  = make_float3( 0.9e6f, 0.9e6f, 0.9e6f );
	_light.plane_v1 = normalize(_light.anchor - _light.worldCenter);
	_light.plane_v2 = normalize(cross(_light.plane_v1,make_float3(1.0,0.0,0.0)));
	m_context["light"]->setUserData( sizeof(PPMLight), &_light );
	updateGeometry();
	signalCameraChanged(); 
}


void ProgressivePhotonScene::updateGeometry()
{
	//optix::Matrix<4,4> TranslateToOrigin = optix::Matrix<4,4>::translate( make_float3(0.0,0.0,sinf(M_PI/4)*Rg));
	//optix::Matrix<4,4> TranslateBack = optix::Matrix<4,4>::translate( make_float3(0.0,0.0,-sinf(M_PI/4)*Rg));
	//optix::Matrix<4,4> Translate = optix::Matrix<4,4>::translate( make_float3(0.0,0.0,-sinf(M_PI/4)*Rg));
	float3 Axis1 = normalize( make_float3(1.0,0.0,0.0) );
    optix::Matrix4x4 Rotate1 = optix::Matrix4x4::rotate( -M_PI, Axis1 );

	float angleLightPlane_XYPlane = acos(dot(_light.plane_v1,make_float3(_light.plane_v1.x,_light.plane_v1.y,0.0)));
	float angleInitConePlane_XYPlane = acos(dot(make_float3(0.0,0.0,1.0),make_float3(_light.plane_v1.x,_light.plane_v1.y,0.0)));

	float cosZL = dot(make_float3(0.0,0.0,1.0), _light.direction);
	float3 axisZL = make_float3(0.0,1.0,0.0); //Y-axis
	float invCoefZL = -1.f;

	float cosXL = dot(make_float3(0.0,1.0,0.0), _light.direction);
	float3 axisXL = make_float3(0.0,0.0,1.0); //Z-axiz
	float invCoefXL = 1.f;

	float3 axisRot = axisZL;
	float invCoef = invCoefZL;
	float cosRot = cosZL;

	optix::Matrix4x4 Rotate2 = optix::Matrix4x4::rotate( acos(cosRot), axisRot );
	optix::Matrix<4,4> Comp = Rotate2;
		
	optix::Matrix4x4 Rotate2_inv = optix::Matrix4x4::rotate((invCoef)*acos(cosRot) , axisRot );
	optix::Matrix<4,4> Comp_inv = Rotate1*Rotate2_inv;
		

	axisRot = axisXL;
	invCoef = invCoefXL;
	cosRot = cosf(atan2f(_light.direction.x, _light.direction.y));
	Rotate2 = optix::Matrix4x4::rotate( acos(cosRot), axisRot );
	//Comp = Rotate2*Comp;
	
		
	Rotate2_inv = optix::Matrix4x4::rotate((invCoef)*acos(cosRot) , axisRot );
	//Comp_inv = Rotate2_inv*Comp_inv;
	
	
	_transform_inner_obj->setMatrix( false, Comp.getData( ), 0 );
	_transform_inner_obj_inv->setMatrix( false, Comp_inv.getData( ), 0 );
}
/*

void ProgressivePhotonScene::updateGeometry()
{
	//optix::Matrix<4,4> TranslateToOrigin = optix::Matrix<4,4>::translate( make_float3(0.0,0.0,sinf(M_PI/4)*Rg));
	//optix::Matrix<4,4> TranslateBack = optix::Matrix<4,4>::translate( make_float3(0.0,0.0,-sinf(M_PI/4)*Rg));
	//optix::Matrix<4,4> Translate = optix::Matrix<4,4>::translate( make_float3(0.0,0.0,-sinf(M_PI/4)*Rg));
	float3 Axis1 = normalize( make_float3(1.0,0.0,0.0) );
    optix::Matrix4x4 Rotate1 = optix::Matrix4x4::rotate( -M_PI, Axis1 );

	float angleLightPlane_XYPlane = acos(dot(_light.plane_v1,make_float3(_light.plane_v1.x,_light.plane_v1.y,0.0)));
	float angleInitConePlane_XYPlane = acos(dot(make_float3(0.0,0.0,1.0),make_float3(_light.plane_v1.x,_light.plane_v1.y,0.0)));

	if(angleInitConePlane_XYPlane < angleLightPlane_XYPlane){
		float3 Axis2 = _light.plane_v2;//cross( make_float3(0.0,0.0,1.0), _light.direction );//_light.plane_v2
		float cosAlfa = dot(make_float3(0.0,0.0,1.0), _light.direction);
		optix::Matrix4x4 Rotate2 = optix::Matrix4x4::rotate( acos(cosAlfa), Axis2 );
		//optix::Matrix<4,4> Comp = TranslateToOrigin;//*Rotate2;//Rotate1*Translate*Rotate2;
		optix::Matrix<4,4> Comp = Rotate2;
		_transform_inner_obj->setMatrix( false, Comp.getData( ), 0 );
		
		float cosAlfa_inv = dot(make_float3(0.0,0.0,-1.0), _light.direction);
		optix::Matrix4x4 Rotate2_inv = optix::Matrix4x4::rotate(-acos(cosAlfa) , Axis2 );
		optix::Matrix<4,4> Comp_inv = Rotate1*Rotate2_inv;
		_transform_inner_obj_inv->setMatrix( false, Comp_inv.getData( ), 0 );
	
	}
	else{
		float3 Axis2 = _light.plane_v2;//cross( make_float3(0.0,0.0,1.0), _light.direction );
		float cosAlfa = dot(make_float3(0.0,0.0,1.0), _light.direction);
		optix::Matrix4x4 Rotate2 = optix::Matrix4x4::rotate( -acos(cosAlfa), Axis2 );
		//optix::Matrix<4,4> Comp = TranslateToOrigin;//*Rotate2;//Rotate1*Translate*Rotate2;
		optix::Matrix<4,4> Comp = Rotate2;
		_transform_inner_obj->setMatrix( false, Comp.getData( ), 0 );
		
		float cosAlfa_inv = dot(make_float3(0.0,0.0,-1.0), -_light.direction);
		optix::Matrix4x4 Rotate2_inv = optix::Matrix4x4::rotate(acos(cosAlfa) , Axis2 );
		optix::Matrix<4,4> Comp_inv = Rotate1*Rotate2_inv;
		_transform_inner_obj_inv->setMatrix( false, Comp_inv.getData( ), 0 );
	}

}
*/

const char* const ProgressivePhotonScene::ptxpath( const std::string& target, const std::string& base )
{
	static std::string path;
	path = std::string("C:/SVN/Dev/optix_gl/ptx") + "/" + target + "_generated_" + base + ".ptx";
	return path.c_str();
}

void ProgressivePhotonScene::initScene( InitialCameraData& camera_data )
{
	// First do some precomputations
	// which include transmittance
	//pr.precompute();
	initExperimentViewProj();

	// There's a performance advantage to using a device that isn't being used as a display.
	// We'll take a guess and pick the second GPU if the second one has the same compute
	// capability as the first.
	int deviceId = 0;
	int computeCaps[2];
	if (RTresult code = rtDeviceGetAttribute(0, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCaps))
		throw Exception::makeException(code, 0);
	for(unsigned int index = 1; index < Context::getDeviceCount(); ++index) {
		int computeCapsB[2];
		if (RTresult code = rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCapsB))
			throw Exception::makeException(code, 0);
		if (computeCaps[0] == computeCapsB[0] && computeCaps[1] == computeCapsB[1]) {
			deviceId = index;
			break;
		}
	}

	//experiment
	m_context["iview_matrix_row_0"]->setFloat(_iviewf[0*4+0], _iviewf[0*4+1], _iviewf[0*4+2], _iviewf[0*4+3]);
	m_context["iview_matrix_row_1"]->setFloat(_iviewf[1*4+0], _iviewf[1*4+1], _iviewf[1*4+2], _iviewf[1*4+3]);
	m_context["iview_matrix_row_2"]->setFloat(_iviewf[2*4+0], _iviewf[2*4+1], _iviewf[2*4+2], _iviewf[2*4+3]);
	m_context["iview_matrix_row_3"]->setFloat(_iviewf[3*4+0], _iviewf[3*4+1], _iviewf[3*4+2], _iviewf[3*4+3]);

	m_context["iproj_matrix_row_0"]->setFloat(_iproj[0*4+0], _iproj[0*4+1], _iproj[0*4+2], _iproj[0*4+3]);
	m_context["iproj_matrix_row_1"]->setFloat(_iproj[1*4+0], _iproj[1*4+1], _iproj[1*4+2], _iproj[1*4+3]);
	m_context["iproj_matrix_row_2"]->setFloat(_iproj[2*4+0], _iproj[2*4+1], _iproj[2*4+2], _iproj[2*4+3]);
	m_context["iproj_matrix_row_3"]->setFloat(_iproj[3*4+0], _iproj[3*4+1], _iproj[3*4+2], _iproj[3*4+3]);

	m_context["new_eye"]->setFloat(_cam_pos);

	m_context->setPrintEnabled( 1 ); 
	//m_context->setExceptionEnabled(RT_EXCEPTION_ALL,1);
	m_context->setPrintBufferSize( 4096 );
	m_context->setDevices(&deviceId, &deviceId+1);
	m_context->setRayTypeCount( 5 );
	m_context->setEntryPointCount(4);//numPrograms );
	//m_context->setStackSize( 20000 );

	m_context["max_depth"]->setUint(20);
	
	m_context["max_photon_count"]->setUint(MAX_PHOTON_COUNT);
	m_context["max_vol_photon_count"]->setUint(MAX_VOL_PHOTON_COUNT);
	
	m_context["rtpass_ray_type"]->setUint(rtpass);
	m_context["pbinpass_ray_type"]->setUint(pbinpass);
	m_context["rtpass_shadow_ray_type"]->setUint(shadow);
	//m_context["gather_samples"]->setUint(gather_samples);
	m_context["ppass_ray_type"]->setUint(ppass);
	m_context["gather_ray_type"]->setUint(ppass);
        

	m_context["scene_epsilon"]->setFloat( 1.e-1f );
	m_context["alpha"]->setFloat( 0.7f );
	m_context["total_emitted"]->setFloat( 0.0f );
	m_context["frame_number"]->setFloat( 0.0f );
	m_context["full_gather"]->setInt((int)_full_gather);
	m_context["hdrOn"]->setInt((int)_hdrOn);
	m_context["exposure"]->setFloat(_exposure);
	m_context["singleOnly"]->setInt(_singleOnly);
	m_context["use_debug_buffer"]->setUint( _display_debug_buffer ? 1 : 0 );


	// Display buffer
	_display_buffer = createOutputBuffer(RT_FORMAT_FLOAT4, WIDTH, HEIGHT);
	m_context["output_buffer"]->set( _display_buffer );

	

	// Debug output buffer
	_debug_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT );
	_debug_buffer->setFormat( RT_FORMAT_FLOAT4 );
	_debug_buffer->setSize( WIDTH, HEIGHT );
	m_context["debug_buffer"]->set( _debug_buffer );

	// RTPass output buffer
	Buffer output_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT );
	output_buffer->setFormat( RT_FORMAT_USER );
	output_buffer->setElementSize( sizeof( HitRecord ) );
	output_buffer->setSize( WIDTH, HEIGHT );
	m_context["rtpass_output_buffer"]->set( output_buffer );

	Buffer output_samples = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	output_samples->setFormat( RT_FORMAT_USER );
	output_samples->setElementSize( sizeof( zpSample ) );
	output_samples->setSize( WIDTH*sqrt_samples_per_pixel, HEIGHT*sqrt_samples_per_pixel );
	m_context["rtpass_output_samples"]->set( output_samples);//getBuffer()->setSize( sqrt_samples_per_pixel * window_width, sqrt_samples_per_pixel * window_height );

	m_context["sqrt_samples_per_pixel"]->setUint( sqrt_samples_per_pixel );

	/*
	_rt_points = m_context->createBuffer( RT_BUFFER_OUTPUT );
	_rt_points->setFormat( RT_FORMAT_USER );
	_rt_points->setElementSize( sizeof( PhotonRecord ) );
	_rt_points->setSize( NUM_VOL_PHOTONS );
	m_context["rt_points_output_buffer"]->set( _rt_points );
	*/  
	//m_context["stepSize"]->setFloat(60.0);
	m_context["stepSize"]->setFloat(0.50);
	m_context["stepSizeP"]->setFloat(MARCHING_STEP);
	m_context["viewStepSize"]->setFloat(VIEWING_MARCHING_STEP);
	//m_context["numSteps"]->setFloat(10u);



	// RTPass pixel sample buffers
	Buffer image_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT2, WIDTH, HEIGHT );
	m_context["image_rnd_seeds"]->set( image_rnd_seeds );
	uint2* seeds = reinterpret_cast<uint2*>( image_rnd_seeds->map() );
	for ( unsigned int i = 0; i < WIDTH*HEIGHT; ++i )  
		seeds[i] = random2u();
	image_rnd_seeds->unmap();

#ifdef RT_PASS_ON
	// RTPass ray gen program
	{
		std::string ptx_path = ptxpath( "isgShadows", "ppm_rtpass.cu" );
		Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "rtpass_camera" );
		m_context->setRayGenerationProgram( rtpass, ray_gen_program );

		// RTPass exception/miss programs
		Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "rtpass_exception" );
		m_context->setExceptionProgram( rtpass, exception_program );
		m_context["rtpass_bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
		m_context->setMissProgram( rtpass, m_context->createProgramFromPTXFile( ptx_path, "rtpass_miss" ) );
		m_context["rtpass_bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );
	}
#endif
	// Set up camera
	// Declare these so validation will pass
	m_context["rtpass_eye"]->setFloat( make_float3( 0.0f, 160.0f, -160.0f ) );
	m_context["rtpass_U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	m_context["rtpass_V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	m_context["rtpass_W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

	// Gather_sample pass
	//std::string ptx_path = ptxpath( "isgShadows", "ppm_rtpass.cu" );
	//Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "zp_gather_samples" );
	//m_context->setRayGenerationProgram( gather_samples, ray_gen_program ); 
	//Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "gather_samples_exception" );
	//m_context->setExceptionProgram( gather_samples, exception_program );

	// Photon pass
	m_context["color_band"]->setInt( 0 );

	_photons = m_context->createBuffer( RT_BUFFER_OUTPUT );
	_photons->setFormat( RT_FORMAT_USER );
	_photons->setElementSize( sizeof( PhotonRecord ) );
	_photons->setSize( NUM_PHOTONS );
	m_context["ppass_output_buffer"]->set( _photons );

	_vol_photons = m_context->createBuffer( RT_BUFFER_OUTPUT );
	_vol_photons->setFormat( RT_FORMAT_USER );
	_vol_photons->setElementSize( sizeof( PhotonRecord ) );
	_vol_photons->setSize( NUM_VOL_PHOTONS );
	m_context["ppass_vol_output_buffer"]->set( _vol_photons );

	// global photon counts table
	_global_photon_counts = m_context->createBuffer( RT_BUFFER_INPUT);//RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL);
	_global_photon_counts->setFormat( RT_FORMAT_UNSIGNED_INT3 );
	_global_photon_counts->setSize( RES_MU_S_BIN*RES_R_BIN*RES_DIR_BIN_TOTAL );
	memset( _global_photon_counts->map(), 0, RES_MU_S_BIN*RES_R_BIN*RES_DIR_BIN_TOTAL*sizeof(optix::uint3) );
	_global_photon_counts->unmap();
	m_context["rtpass_global_photon_counts"]->set( _global_photon_counts );

	// table used for updating global photon counts table
	_vol_counts_photons = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );// RT_BUFFER_OUTPUT);//RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL);
	//_vol_counts_photons->setFormat( RT_FORMAT_UNSIGNED_INT );
	_vol_counts_photons->setFormat( RT_FORMAT_USER );
	//_vol_counts_photons->setElementSize( sizeof( unsigned int  );
	_vol_counts_photons->setElementSize( sizeof( PhotonCountRecord ) );
	_vol_counts_photons->setSize( NUM_VOL_PHOTONS );
	memset( _vol_counts_photons->map(), 0, NUM_VOL_PHOTONS*sizeof( PhotonCountRecord ) );
	_vol_counts_photons->unmap();
	m_context["ppass_vol_counts_output_buffer"]->set( _vol_counts_photons );
	//unsigned int* data = reinterpret_cast<unsigned int*>( _vol_counts_photons->map() );
	//for ( unsigned int i = 0; i < RES_MU_S * RES_R * RES_DIR_BIN_TOTAL; ++i )  
	//	data[i] = 0;
	//_vol_counts_photons->unmap();


	_light_sample = m_context->createBuffer( RT_BUFFER_OUTPUT );
	_light_sample->setFormat( RT_FORMAT_USER );
	_light_sample->setElementSize( sizeof( PhotonRecord ) );
	_light_sample->setSize( PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT );
	m_context["ppass_light_buffer"]->set( _light_sample );

	_lightplane_sample = m_context->createBuffer( RT_BUFFER_OUTPUT );
	_lightplane_sample->setFormat( RT_FORMAT_USER );
	_lightplane_sample->setElementSize( sizeof( PhotonRecord ) );
	_lightplane_sample->setSize( PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT );
	m_context["ppass_lightplane_buffer"]->set( _lightplane_sample );

	_vol_rotate = m_context->createBuffer( RT_BUFFER_OUTPUT );
	_vol_rotate->setFormat( RT_FORMAT_USER );
	_vol_rotate->setElementSize( sizeof( PhotonRecord ) );
	_vol_rotate->setSize( NUM_VOL_PHOTONS );
	m_context["ppass_vol_rotate_output_buffer"]->set( _vol_rotate );

	_photon_table = m_context->createBuffer( RT_BUFFER_OUTPUT );
	_photon_table->setFormat( RT_FORMAT_USER );
	_photon_table->setElementSize( sizeof( PhotonRecord ) );
	_photon_table->setSize( NUM_VOL_PHOTONS );
	m_context["ppass_photon_table"]->set( _photon_table );

	//m_context["stepSize"]->setFloat(60.0);
	//m_context["stepSize"]->setFloat(20.0);
	{
		std::string ptx_path = ptxpath( "isgShadows", "ppm_ppass.cu");

		Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "ppass_camera" );
		m_context->setRayGenerationProgram( ppass, ray_gen_program );


		Buffer photon_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT /*RT_BUFFER_INPUT*/,
			RT_FORMAT_UNSIGNED_INT2,
			/*PHOTON_LAUNCH_WIDTH, PHOTON_LAUNCH_HEIGHT );*/
			PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT, MAX_VOL_PHOTON_COUNT );
		uint2* seeds = reinterpret_cast<uint2*>( photon_rnd_seeds->map() );
		for ( unsigned int i = 0; i < PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT*MAX_VOL_PHOTON_COUNT; ++i )
			seeds[i] = random2u();
		photon_rnd_seeds->unmap();
		m_context["PHOTON_LAUNCH_WIDTH"]->setUint(PHOTON_LAUNCH_WIDTH);
		m_context["photon_rnd_seeds"]->set( photon_rnd_seeds );

	}

	// PBinPass output buffer
	_pbin_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	_pbin_buffer->setFormat( RT_FORMAT_FLOAT3 );
	_pbin_buffer->setSize( RES_MU_S * RES_NU, RES_MU );
	memset( _pbin_buffer->map(), 0, RES_MU_S * RES_NU * RES_MU*sizeof(optix::float3) );
	_pbin_buffer->unmap();
	m_context["pbin_output_buffer"]->set( _pbin_buffer );

	// PBinPass ray gen program
	{
		std::string ptx_path = ptxpath( "isgShadows", "normal_shader.cu" );
		Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pbin_camera" );
		m_context->setRayGenerationProgram( pbinpass, ray_gen_program );

		// RTPass exception/miss programs
		//Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "rtpass_exception" );
		//m_context->setExceptionProgram( rtpass, exception_program );
		//m_context["rtpass_bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
		//m_context->setMissProgram( rtpass, m_context->createProgramFromPTXFile( ptx_path, "rtpass_miss" ) );
		//m_context["rtpass_bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );
	}

	// Gather phase
	{
		std::string ptx_path = ptxpath( "isgShadows", "ppm_gather.cu" );
		Program gather_program = m_context->createProgramFromPTXFile( ptx_path, "gather" );
		m_context->setRayGenerationProgram( gather, gather_program );
		Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "gather_exception" );
		m_context->setExceptionProgram( gather, exception_program );

		_photon_map_size = pow2roundup( NUM_PHOTONS ) - 1;
		_photon_map = m_context->createBuffer( RT_BUFFER_INPUT );
		_photon_map->setFormat( RT_FORMAT_USER );
		_photon_map->setElementSize( sizeof( PhotonRecord ) );
		_photon_map->setSize( _photon_map_size );
		m_context["photon_map"]->set( _photon_map );

		_vol_photon_map_size = pow2roundup( NUM_VOL_PHOTONS ) - 1;
		_vol_photon_map = m_context->createBuffer( RT_BUFFER_INPUT );
		_vol_photon_map->setFormat( RT_FORMAT_USER );
		_vol_photon_map->setElementSize( sizeof( PhotonRecord ) );
		_vol_photon_map->setSize( _vol_photon_map_size );
		m_context["vol_photon_map"]->set( _vol_photon_map );

		_rt_point_map_size = pow2roundup( NUM_VOL_PHOTONS ) - 1;
		_rt_point_map = m_context->createBuffer( RT_BUFFER_INPUT );
		_rt_point_map->setFormat( RT_FORMAT_USER );
		_rt_point_map->setElementSize( sizeof( PhotonRecord ) );
		_rt_point_map->setSize( _rt_point_map_size );
		m_context["rt_point_map"]->set( _rt_point_map );
	}


	optix::Aabb aabb;
	//createTestGeometry(aabb);
	createTestGeometryEarthAtm(aabb);

	// Set up camera
	//camera_data = InitialCameraData( make_float3( 278.0f, 273.0f, -850.0f ), // eye
	//                                 make_float3( 278.0f, 112.0f, 227.0f ),    // lookat
	//                                 make_float3( 0.0f, 1.0f,  0.0f ),       // up
	//                                 35.0f );                                // vfov
	/*
	camera_data = InitialCameraData( make_float3( 0.0f, 0.0f, -800.0f ), // eye
	make_float3( 0.0f, 0.0f, 0.0f ),    // lookat
	make_float3( 0.0f, 1.0f,  0.0f ),       // up
	100.0f );                                // vfov
	*/


	float vfov = 2.0 * atan(float(HEIGHT) / float(WIDTH) * tan(80.0 / 180 * M_PI / 2.0)) / M_PI * 180;
	//float h = position.length() - Rg;
	//mat4f proj = mat4f::perspectiveProjection(vfov, float(width) / float(height), 0.1 * h, 1e5 * h);

	_camera[0] = InitialCameraData( make_float3( 0.0f, 193.0f, 0.0f ), // eye
		make_float3( 0.0f, 150.0f, 0.0f ),    // lookat
		make_float3( 0.0f, 1.0f,  0.0f ),       // up
		vfov );                                // vfov

	_camera[1] = InitialCameraData( make_float3( 0.0f, 0.0f, -800.0f ), // eye
		make_float3( 0.0f, 0.0f, 0.0f ),    // lookat
		make_float3( 0.0f, 1.0f,  0.0f ),       // up
		13.0f );                                // vfov
	_camera[2] = InitialCameraData( make_float3( 0.0f, 155.0f, -155.0f ), // eye
		make_float3( 0.0f, 120.0f, 250.0f ),    // lookat
		make_float3( 0.0f, 1.0f,  0.0f ),       // up
		13.0f );                                // vfov


	camera_data = _camera[_camera_index];
	_light.is_area_light = 2;
	float3 dmtr = aabb.m_max - aabb.m_min;
	float dmtr_mag = sqrtf(dot(dmtr, dmtr))/1.7;
	//_light.radius = _light.worldRadius;
	_light.worldRadius = (dmtr_mag)/2.0;//10
	_light.worldCenter = (aabb.m_max + aabb.m_min)/2.0f;


	//_light.anchor = _light.worldCenter + make_float3(0.0,0.0,-RS);
	_light.anchor = _light.worldCenter + make_float3(0.0, 0.0, -RS);


	createONB(_light.anchor, _light.v1, _light.v2,  _light.v3);
	_light.direction = make_float3(0.0, 0.0, RS);//normalize(_light.worldCenter +  _light.anchor );
	_light.power  = make_float3( 0.9e6f, 0.9e6f, 0.9e6f );

	_light.plane_v1 = normalize(_light.anchor - _light.worldCenter);
	_light.plane_v2 = normalize(cross(_light.plane_v1,make_float3(1.0,0.0,0.0)));


	//scene

	//float p = 3.14 * _light.worldRadius * _light.worldRadius;
	//_light.power  = make_float3( p, p, p );

	m_context["light"]->setUserData( sizeof(PPMLight), &_light );
	m_context["rtpass_default_radius2"]->setFloat( 50.0f);
	m_context["ambient_light"]->setFloat( 0.0f, 0.0f, 0.0f);
	const float3 default_color = make_float3(0.0f, 0.0f, 0.0f);
	m_context["envmap"]->setTextureSampler( loadTexture( m_context, "", default_color) );
	m_context["SCALE_FACTOR"]->setFloat(100.0);
	m_context["TEST_RT_PASS"]->setInt(_testRtPassLevel);

	loadData();
	precompute();
	//loadTexToOptix();
	// Prepare to run
	//m_context->setStackSize( 2000 );
	int n = m_context->getStackSize();
	m_context->validate();
	m_context->compile();
}

Buffer ProgressivePhotonScene::getOutputBuffer()
{
	return _display_buffer;
}

inline uchar4 makeColor( const float3& c )
{
	uchar4 pixel;
	pixel.x = static_cast<unsigned char>( fmaxf( fminf( c.z, 1.0f ), 0.0f ) * 255.99f );
	pixel.y = static_cast<unsigned char>( fmaxf( fminf( c.y, 1.0f ), 0.0f ) * 255.99f );
	pixel.z = static_cast<unsigned char>( fmaxf( fminf( c.x, 1.0f ), 0.0f ) * 255.99f );
	pixel.w = 0; 
	return pixel;
}


bool photonCmpX( PhotonRecord* r1, PhotonRecord* r2 ) { return r1->position.x < r2->position.x; }
bool photonCmpY( PhotonRecord* r1, PhotonRecord* r2 ) { return r1->position.y < r2->position.y; }
bool photonCmpZ( PhotonRecord* r1, PhotonRecord* r2 ) { return r1->position.z < r2->position.z; }


void buildKDTree( PhotonRecord** photons, int start, int end, int depth, PhotonRecord* kd_tree, int current_root,
				 SplitChoice split_choice, float3 bbmin, float3 bbmax)
{
	// If we have zero photons, this is a NULL node
	if( end - start == 0 ) {
		kd_tree[current_root].axis = PPM_NULL;
		kd_tree[current_root].energy = make_float3( 0.0f );
		return;
	}

	// If we have a single photon
	if( end - start == 1 ) {
		photons[start]->axis = PPM_LEAF;
		kd_tree[current_root] = *(photons[start]);
		return;
	}

	// Choose axis to split on
	int axis;
	switch(split_choice) {
  case RoundRobin:
	  {
		  axis = depth%3;
	  }
	  break;
  case HighestVariance:
	  {
		  float3 mean  = make_float3( 0.0f ); 
		  float3 diff2 = make_float3( 0.0f );
		  for(int i = start; i < end; ++i) {
			  float3 x     = photons[i]->position;
			  float3 delta = x - mean;
			  float3 n_inv = make_float3( 1.0f / ( static_cast<float>( i - start ) + 1.0f ) );
			  mean = mean + delta * n_inv;
			  diff2 += delta*( x - mean );
		  }
		  float3 n_inv = make_float3( 1.0f / ( static_cast<float>(end-start) - 1.0f ) );
		  float3 variance = diff2 * n_inv;
		  axis = max_component(variance);
	  }
	  break;
  case LongestDim:
	  {
		  float3 diag = bbmax-bbmin;
		  axis = max_component(diag);
	  }
	  break;
  default:
	  axis = -1;
	  std::cerr << "Unknown SplitChoice " << split_choice << " at "<<__FILE__<<":"<<__LINE__<<"\n";
	  exit(2);
	  break;
	}

	int median = (start+end) / 2;
	PhotonRecord** start_addr = &(photons[start]);
#if 0
	switch( axis ) {
  case 0:
	  std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpX );
	  photons[median]->axis = PPM_X;
	  break;
  case 1:
	  std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpY );
	  photons[median]->axis = PPM_Y;
	  break;
  case 2:
	  std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpZ );
	  photons[median]->axis = PPM_Z;
	  break;
	}
#else
	switch( axis ) {
  case 0:
	  select<PhotonRecord*, 0>( start_addr, 0, end-start-1, median-start );
	  photons[median]->axis = PPM_X;
	  break;
  case 1:
	  select<PhotonRecord*, 1>( start_addr, 0, end-start-1, median-start );
	  photons[median]->axis = PPM_Y;
	  break;
  case 2:
	  select<PhotonRecord*, 2>( start_addr, 0, end-start-1, median-start );
	  photons[median]->axis = PPM_Z;
	  break;
	}
#endif
	float3 rightMin = bbmin;
	float3 leftMax  = bbmax;
	if(split_choice == LongestDim) {
		float3 midPoint = (*photons[median]).position;
		switch( axis ) {
	  case 0:
		  rightMin.x = midPoint.x;
		  leftMax.x  = midPoint.x;
		  break;
	  case 1:
		  rightMin.y = midPoint.y;
		  leftMax.y  = midPoint.y;
		  break;
	  case 2:
		  rightMin.z = midPoint.z;
		  leftMax.z  = midPoint.z;
		  break;
		}
	}

	kd_tree[current_root] = *(photons[median]);
	buildKDTree( photons, start, median, depth+1, kd_tree, 2*current_root+1, split_choice, bbmin,  leftMax );
	buildKDTree( photons, median+1, end, depth+1, kd_tree, 2*current_root+2, split_choice, rightMin, bbmax );
}


void ProgressivePhotonScene::createPhotonMap()
{
	PhotonRecord* photons_data    = reinterpret_cast<PhotonRecord*>( _photons->map() );
	PhotonRecord* photon_map_data = reinterpret_cast<PhotonRecord*>( _photon_map->map() );

	for( unsigned int i = 0; i < _photon_map_size; ++i ) {
		photon_map_data[i].energy = make_float3( 0.0f );
	}

	// Push all valid photons to front of list
	unsigned int valid_photons = 0;
	PhotonRecord** temp_photons = new PhotonRecord*[NUM_PHOTONS];
	for( unsigned int i = 0; i < NUM_PHOTONS; ++i ) {
		if( fmaxf( photons_data[i].energy ) > 0.0f ) {
			temp_photons[valid_photons++] = &photons_data[i];
		}
	}
	if ( _display_debug_buffer ) {
		std::cerr << " ** valid_photon/NUM_PHOTONS =  " 
			<< valid_photons<<"/"<<NUM_PHOTONS
			<<" ("<<valid_photons/static_cast<float>(NUM_PHOTONS)<<")\n";
	}

	// Make sure we arent at most 1 less than power of 2
	valid_photons = valid_photons >= _photon_map_size ? _photon_map_size : valid_photons;

	float3 bbmin = make_float3(0.0f);
	float3 bbmax = make_float3(0.0f);
	if( _split_choice == LongestDim ) {
		bbmin = make_float3(  std::numeric_limits<float>::max() );
		bbmax = make_float3( -std::numeric_limits<float>::max() );
		// Compute the bounds of the photons
		for(unsigned int i = 0; i < valid_photons; ++i) {
			float3 position = (*temp_photons[i]).position;
			bbmin = fminf(bbmin, position);
			bbmax = fmaxf(bbmax, position);
		}
	}
	/*
	for(unsigned int i = 0; i < valid_photons; ++i) {
	float3 position = (*temp_photons[i]).position;
	std::cerr << "position[" << i << "]= " 
	<< position.x <<", " <<  position.y << ", " << position.z << "\n";
	}
	std::cerr << "_____________" << "\n";
	*/

	SavePhotonMap(temp_photons, valid_photons, "c:\\SVN\\volumemap.bin");
	// Now build KD tree
	buildKDTree( temp_photons, 0, valid_photons, 0, photon_map_data, 0, _split_choice, bbmin, bbmax );

	delete[] temp_photons;
	_photon_map->unmap();
	_photons->unmap();
}

/*
float3 tex4D_inscatter(Buffer sampler, float r, float mu, float muS, float nu)
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
*/

//RES_R, RES_MU_S * RES_NU, RES_MU, "c:\\SVN\\vol_bin_counts.bin"
void ProgressivePhotonScene::savePhotonBins(Buffer pdata, int sizex, int sizey, int sizez, string fname)
{
	float3* photons_data = reinterpret_cast<float3*>( pdata->map() );
	FILE* fp = fopen(fname.c_str(),"w");
	
	int muS_val = 3;
	int r_val = 16;
	float3 val = make_float3(0.0);
	int total = 0;
	for(int x=0; x < sizex ; x++){
		for(int y = 0; y < sizey; y++){
			float muS = (y % sizey) / (float(sizey) - 1.0);
			int muS_ind = (int)(muS*RES_MU_S);
			//float uMuS = 0.5 / float(RES_MU_S) + max(muS + 0.2, 0.0) / 1.2 * (1.0 - 1.0 / float(RES_MU_S));
			for(int z = 0; z < sizez; z++){
				//muS = ((int)x % (int)RES_MU_S) / (float(RES_MU_S) - 1.0);
				//muS = -0.2 + muS * 1.2;
				int ind = x*sizey*sizez+y*sizez +z;
				val = photons_data[ind];
				if(val.x +val.y + val.z > 0.0)
					fprintf(fp,"[%d, %d, %d] (%f,%f,%f) \n", x, y, z, val.x,val.y,val.z);
				//if(muS_ind == muS_val && x == r_val)
				//	fprintf(fp,">>>>>>>[%d, %d, %d] (%f,%f,%f) \n", x, y, z, val.x,val.y,val.z);
			}	
			fprintf(fp,"BIN [muS:%d r:%d, muS X nu:%d]\n", muS_ind, x, y);
		}
	}
	fclose(fp);
	pdata->unmap();
}

void ProgressivePhotonScene::testPhotonBinCounts()
{
	optix::uint3* photoncounts_data    = reinterpret_cast<optix::uint3*>( _global_photon_counts->map() );
	for(int i=0; i < RES_MU_S_BIN ; i++){
		for(int k = 0; k < RES_R_BIN; k++){
			for(int j = 0; j < RES_DIR_BIN_TOTAL; j++){
				if((unsigned int)photoncounts_data[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].x >0)
					cout << "x: " << (unsigned int)photoncounts_data[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].x << ", " << endl;
				if((unsigned int)photoncounts_data[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].y >0)
					cout << "y: " << (unsigned int)photoncounts_data[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].y << ", " << endl;
				if((unsigned int)photoncounts_data[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].z >0)
					cout << "z: " << (unsigned int)photoncounts_data[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].z << ", " << endl;
			}
		}
	}
	_global_photon_counts->unmap();
}

void ProgressivePhotonScene::loadBinCountsTexture(float* pTexBuffer, int k,  int width, int height, int depth, int dim)
{
	
	optix::float3* data = reinterpret_cast<optix::float3*>( _pbin_buffer->map() );
	bool debug = false;
	
	
	// Read in HDR, set texture buffer to empty buffer if fails
	
	const unsigned int nx = width;
	const unsigned int ny = height;
	const unsigned int nz = depth;

	// Create buffer and populate with HDR data
	
	for ( unsigned int i = 0; i < nx; ++i ) {
		for ( unsigned int j = 0; j < ny; ++j ) {
			//for ( unsigned int k = 0; k < nz; ++k) {
				//unsigned int hdr_index = ( (ny-j-1)*nx + i )*4;
				//unsigned int buf_index = ( (j     )*nx + i )*4;
				//unsigned int hdr_index = (k + i*nz + (ny-j-1)*nx*nz)*dim;//(k + i*nz + j*nx*nz)*dim;
				//unsigned int buf_index = (k + i*nz + (ny-j-1)*nx*nz)*dim;//(k + i*nz + j*nx*nz)*dim;
				
			
				//unsigned int hdr_index = ( (k)*nx*ny + ny-j-1 + (nx-i-1)*ny )*dim;
				//unsigned int buf_index = ( (k)*nx*ny + j + i*ny )*dim;
				//unsigned int hdr_index =  (k + (j)*nz + (nx-i-1)*ny*nz)*dim;
				//unsigned int buf_index =  (k + (j)*nz + (i)*ny*nz)*4;
				// int hdr_index = (k*nx*ny + (j)*nx + i)*4;
				//<!!!>unsigned int hdr_index =  ((j) + i*ny);
				//<!!!>unsigned int buf_index =  (k*nx*ny + (j) + (i)*ny)*4;
				//unsigned int buf_index =  (k*nx*ny + (j) + (nx-i-1)*ny*nz)*4;
				//unsigned int buf_index =  (k*nx*ny + (j) + i*ny)*4;
			int ind_jd = j;
			int ind_jt = j;
			if(ind_jd < ny/2){
				ind_jt = ind_jd + ny/2;
			}
			else{
				ind_jt = ind_jd - ny/2;
			}

			int ind_id = i;
			int ind_it = i;
			if(ind_id < nx/2){
				ind_it = ind_id + nx/2;
			}
			else{
				ind_it = ind_id - nx/2;
			}

			unsigned int hdr_index =  ((j)*nx + i);//ind_id);
			unsigned int buf_index =  (k*nx*ny + (j)*nx + i)*4;//(ind_it))*4;
			pTexBuffer[ buf_index + 0 ] = data[ hdr_index].x;
			pTexBuffer[ buf_index + 1 ] = data[ hdr_index].y;
			pTexBuffer[ buf_index + 2 ] = data[ hdr_index].z;
			if(dim == 4){
				pTexBuffer[ buf_index + 3 ] = 0.0;
			}
			else if(dim == 3){
				pTexBuffer[ buf_index + 3 ] = 0.0;
			}
			//}
		}
	}

	if(debug){
		FILE* fp = fopen("c:\\SVN\\texttest.txt","w");
		for ( unsigned int i = 0; i < nx; ++i ) {
			for ( unsigned int j = 0; j < ny; ++j ) {
				for ( unsigned int k = 0; k < nz; ++k) {
					unsigned int hdr_index =  (k + (j)*nz + (nx-i-1)*ny*nz)*dim;
					fprintf(fp,"%f,%f,%f" ,data[ hdr_index + 0 ],data[ hdr_index + 1 ],data[ hdr_index + 2 ]);
					if(dim == 4){
						fprintf(fp,",%f",data[ hdr_index + 3 ]);
					}
					fprintf(fp,"\n");
				}
			}
		}
		fclose(fp);
	}

	_pbin_buffer->unmap();
	memset( _pbin_buffer->map(), 0, RES_MU_S * RES_NU * RES_MU*sizeof(optix::float3) );
	_pbin_buffer->unmap();
	//m_context["inscatterPhotonSampler"]->setTextureSampler(sampler);
}

bool ProgressivePhotonScene::loadPhotonBinCounts(string fname)
{
	ifstream inf(fname.c_str(), ios::binary);
	//#FORMAT:
	//num_cb
	//RES_MU_S_BIN
	//RES_R_BIN
	//RES_DIR_BIN
	//RES_DIR_BIN_TOTAL
	//#Reds:
	//val1val2val3...
	//#Greens:
	//val1val2val3...
	//Blues:
	//val1val2val3...
	//
	//assert(inf);	
	if(!inf)
		return false;
	int num_cb = 3;
	int locRES_MU_S_BIN = RES_MU_S_BIN;
	int locRES_R_BIN = RES_R_BIN;
	int locRES_DIR_BIN = RES_DIR_BIN;
	int locRES_DIR_BIN_TOTAL = RES_DIR_BIN_TOTAL;

	inf.read(reinterpret_cast<char *>(&num_cb), sizeof(int));
	inf.read(reinterpret_cast<char *>(&locRES_MU_S_BIN), sizeof(int));
	inf.read(reinterpret_cast<char *>(&locRES_R_BIN), sizeof(int));
	inf.read(reinterpret_cast<char *>(&locRES_DIR_BIN), sizeof(int));
	inf.read(reinterpret_cast<char *>(&locRES_DIR_BIN_TOTAL), sizeof(int));

	_global_photon_counts->destroy();
	_global_photon_counts = m_context->createBuffer( RT_BUFFER_INPUT);//RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL);
	_global_photon_counts->setFormat( RT_FORMAT_UNSIGNED_INT3 );

	_global_photon_counts->setSize( locRES_MU_S_BIN*locRES_R_BIN*locRES_DIR_BIN_TOTAL );
	memset( _global_photon_counts->map(), 0, locRES_MU_S_BIN*locRES_R_BIN*locRES_DIR_BIN_TOTAL*sizeof(optix::uint3) );
	_global_photon_counts->unmap();
	m_context["rtpass_global_photon_counts"]->set( _global_photon_counts );
	m_context["RES_MU_S_BIN"]->setInt(locRES_MU_S_BIN);
	m_context["RES_R_BIN"]->setInt(locRES_R_BIN);
	m_context["RES_DIR_BIN_TOTAL"]->setInt(locRES_DIR_BIN_TOTAL);
	m_context["RES_DIR_BIN"]->setInt(locRES_DIR_BIN);

	optix::uint3* photoncounts_data = reinterpret_cast<optix::uint3*>( _global_photon_counts->map() );
	for(int l_cb=0; l_cb < 3; l_cb++){
		unsigned int val = 0;
		unsigned int total = 0;
		unsigned int final_total = 0;
		for(int i=0; i < locRES_MU_S_BIN ; i++){
			for(int k = 0; k < locRES_R_BIN; k++){
				for(int j = 0; j < locRES_DIR_BIN_TOTAL; j++){
					inf.read(reinterpret_cast<char *>(&val), sizeof(unsigned int));
					switch(l_cb){
						case 0:
							//if(photoncounts_data[i*locRES_R_BIN*locRES_DIR_BIN_TOTAL + k*locRES_DIR_BIN_TOTAL+j].x < 1000000)
								photoncounts_data[i*locRES_R_BIN*locRES_DIR_BIN_TOTAL + k*locRES_DIR_BIN_TOTAL+j].x = val;
							//else
							//	photoncounts_data[i*locRES_R_BIN*locRES_DIR_BIN_TOTAL + k*locRES_DIR_BIN_TOTAL+j].x = 1000000;
							break;
						case 1:
							//if(photoncounts_data[i*locRES_R_BIN*locRES_DIR_BIN_TOTAL + k*locRES_DIR_BIN_TOTAL+j].y < 100000)
								photoncounts_data[i*locRES_R_BIN*locRES_DIR_BIN_TOTAL + k*locRES_DIR_BIN_TOTAL+j].y = val;
							//else
							//	photoncounts_data[i*locRES_R_BIN*locRES_DIR_BIN_TOTAL + k*locRES_DIR_BIN_TOTAL+j].y = 100000;
							break;
						case 2:
							//if(photoncounts_data[i*locRES_R_BIN*locRES_DIR_BIN_TOTAL + k*locRES_DIR_BIN_TOTAL+j].z < 100000)
								photoncounts_data[i*locRES_R_BIN*locRES_DIR_BIN_TOTAL + k*locRES_DIR_BIN_TOTAL+j].z = val;
							//else
							//	photoncounts_data[i*locRES_R_BIN*locRES_DIR_BIN_TOTAL + k*locRES_DIR_BIN_TOTAL+j].z = 100000;
							break;
						default:
							break;
					}
					
				}
			}
		}
	}
	_global_photon_counts->unmap();
	inf.close();

	return true;
}


// 
//	-save photon_counts to files
//	-update global photon counts table
//
void ProgressivePhotonScene::savePhotonBinCounts(Buffer pdata, int size, string fname, int cb, bool bSave)
{
	//unsigned int* photons_data    = reinterpret_cast<unsigned int*>( pdata->map() );
	PhotonCountRecord* photons_data    = reinterpret_cast<PhotonCountRecord*>( pdata->map() );
	optix::uint3* photoncounts_data    = reinterpret_cast<optix::uint3*>( _global_photon_counts->map() );
	//PhotonRecord* temp_photons = new PhotonRecord[RES_MU_S_BIN*RES_R_BIN*RES_DIR_BIN_TOTAL];
	//stringstream st;
    //st << nc;
	string fname_tbl = "C:\\SVN\\ascii_table_vol_count.txt";
    //fname_tbl += st.str();
	//fname_tbl += ".txt";

	for(int i=0; i < size; i++){
		//int pos_dir_loc = photons_data[i].pos_dir_index;
		//int dir_count = photons_data[i].dir_count;
		//int cuda_index = photons_data[i].cuda_index;
		switch(cb){
			case 0:
				if(photons_data[i].dir_count != 0){
					photoncounts_data[photons_data[i].pos_dir_index].x += 1;//photons_data[i].dir_count; 
				}
				break;
			case 1:
				if(photons_data[i].dir_count != 0){
					photoncounts_data[photons_data[i].pos_dir_index].y += 1;//photons_data[i].dir_count; 
				}
				break;
			case 2:
				if(photons_data[i].dir_count != 0){
					photoncounts_data[photons_data[i].pos_dir_index].z += 1;//photons_data[i].dir_count; 
				}
				break;
			default:
				break;
		}
		//outf.write(reinterpret_cast<char *>(&pos_dir_loc), sizeof(int));
		//outf.write(reinterpret_cast<char *>(&cuda_index), sizeof(int));
		//outf.write(reinterpret_cast<char *>(&dir_count), sizeof(int));
	}
	memset( photons_data, 0, NUM_VOL_PHOTONS*sizeof( PhotonCountRecord ) );

	
	if(bSave){
		ofstream outf(fname.c_str(), ios::binary);
		//#FORMAT:
		//num_cb
		//RES_MU_S_BIN
		//RES_R_BIN
		//RES_DIR_BIN
		//RES_DIR_BIN_TOTAL
		//#Reds:
		//val1val2val3...
		//#Greens:
		//val1val2val3...
		//Blues:
		//val1val2val3...
		//
		assert(outf);	
		if(!outf)
			return;

		int num_cb = 3;
		int locRES_MU_S_BIN = RES_MU_S_BIN;
		int locRES_R_BIN = RES_R_BIN;
		int locRES_DIR_BIN = RES_DIR_BIN;
		int locRES_DIR_BIN_TOTAL = RES_DIR_BIN_TOTAL;
		outf.write(reinterpret_cast<char *>(&num_cb), sizeof(int));
		outf.write(reinterpret_cast<char *>(&locRES_MU_S_BIN), sizeof(int));
		outf.write(reinterpret_cast<char *>(&locRES_R_BIN), sizeof(int));
		outf.write(reinterpret_cast<char *>(&locRES_DIR_BIN), sizeof(int));
		outf.write(reinterpret_cast<char *>(&locRES_DIR_BIN_TOTAL), sizeof(int));

		FILE* fp = fopen(fname_tbl.c_str(),"w");
		for(int l_cb=0; l_cb < 3; l_cb++){

			fprintf(fp,"color band = %d\n", l_cb);
			unsigned int val = 0;
			unsigned int total = 0;
			unsigned int final_total = 0;
			for(int i=0; i < locRES_MU_S_BIN ; i++){
				for(int k = 0; k < locRES_R_BIN; k++){
					total = 0;
					for(int j = 0; j < locRES_DIR_BIN_TOTAL; j++){
						switch(l_cb){
							case 0:
								val = (unsigned int)photoncounts_data[i*locRES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].x;
								break;
							case 1:
								val = (unsigned int)photoncounts_data[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].y;
								break;
							case 2:
								val = (unsigned int)photoncounts_data[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].z;
								break;
							default:
								break;
						}
						outf.write(reinterpret_cast<char *>(&val), sizeof(unsigned int));
						//temp_photons[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].position = make_float3();
						//photoncounts_data[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j] = 11;
						total += val;
						//outf.write(reinterpret_cast<char *>(&val), sizeof(unsigned int));
						//if(j < RES_DIR_BIN_TOTAL-1)
						//	fprintf(fp,"[%d, %d, %d (ind:%d)] %u, ", i, k, j,  (i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j), val);
						//else
						//	fprintf(fp,"[%d, %d, %d] %u\n", i, k, j, val);
					}
					//fprintf(fp,"BIN [muS: %d, r: %d] TOTAL = %u\n", i, k, total);
					final_total += total;
				}
			}
			fprintf(fp,"FULL TOTAL = %u, color_band = %d\n", final_total, l_cb);
		}
		fclose(fp);

		outf.close();
	}

	
/*
	for(int i=0; i < num_photons; i++){
		float3 pos = (*temp_photons[i]).position;
		outf.write(reinterpret_cast<char *>(&pos.x), sizeof(float));
		outf.write(reinterpret_cast<char *>(&pos.y), sizeof(float));
		outf.write(reinterpret_cast<char *>(&pos.z), sizeof(float));

		float3 dir = (*temp_photons[i]).ray_dir;
		float3 energy = (*temp_photons[i]).energy;
		//for(int j = 0; j < COLOR_SAMPLES; j++){
		//	float c = 1.0;
		outf.write(reinterpret_cast<char *>(&dir.x), sizeof(float));
		outf.write(reinterpret_cast<char *>(&dir.y), sizeof(float));
		outf.write(reinterpret_cast<char *>(&dir.z), sizeof(float));
		//}
	}
*/
	
	//delete [] temp_photons;
	_global_photon_counts->unmap();
	pdata->unmap();
}


void ProgressivePhotonScene::savePhotonBinCountsHist(string fname, string fname_tbl, bool bSave)
{
	optix::uint3* photoncounts_data    = reinterpret_cast<optix::uint3*>( _global_photon_counts->map() );
	//string fname_tbl = "C:\\SVN\\ascii_vol_count_hist.txt";

	ifstream inf(fname.c_str(), ios::binary);
	if(!inf)
		return;
	int num_cb = 3;
	int locRES_MU_S_BIN = RES_MU_S_BIN;
	int locRES_R_BIN = RES_R_BIN;
	int locRES_DIR_BIN = RES_DIR_BIN;
	int locRES_DIR_BIN_TOTAL = RES_DIR_BIN_TOTAL;

	inf.read(reinterpret_cast<char *>(&num_cb), sizeof(int));
	inf.read(reinterpret_cast<char *>(&locRES_MU_S_BIN), sizeof(int));
	inf.read(reinterpret_cast<char *>(&locRES_R_BIN), sizeof(int));
	inf.read(reinterpret_cast<char *>(&locRES_DIR_BIN), sizeof(int));
	inf.read(reinterpret_cast<char *>(&locRES_DIR_BIN_TOTAL), sizeof(int));
	inf.close();

	if(bSave){
		int num_cb = 3;
		//int locRES_MU_S_BIN = RES_MU_S_BIN;
		//int locRES_R_BIN = RES_R_BIN;
		//int locRES_DIR_BIN = RES_DIR_BIN;
		//int locRES_DIR_BIN_TOTAL = RES_DIR_BIN_TOTAL;
		
		FILE* fp = fopen(fname_tbl.c_str(),"w");
		for(int l_cb=0; l_cb < num_cb; l_cb++){

			fprintf(fp,"color band = %d\n", l_cb);
			unsigned int val = 0;
			unsigned int total = 0;
			unsigned int final_total = 0;
			for(int i=0; i < locRES_MU_S_BIN ; i++){
				for(int k = 0; k < locRES_R_BIN; k++){
					total = 0;
					for(int j = 0; j < locRES_DIR_BIN_TOTAL; j++){
						switch(l_cb){
							case 0:
								val = (unsigned int)photoncounts_data[i*locRES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].x;
								break;
							case 1:
								val = (unsigned int)photoncounts_data[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].y;
								break;
							case 2:
								val = (unsigned int)photoncounts_data[i*RES_R_BIN*RES_DIR_BIN_TOTAL + k*RES_DIR_BIN_TOTAL+j].z;
								break;
							default:
								break;
						}
						total += val;
						
					}
					//fprintf(fp,"[%d, %d]=%u\n", i, k, total);
					final_total += total;
				}
			}
			fprintf(fp,"FULL TOTAL = %u, color_band = %d\n", final_total, l_cb);
		}
		fclose(fp);
	}

	_global_photon_counts->unmap();
}

void ProgressivePhotonScene::createPhotonRecordMap(Buffer pdata, Buffer pmap, int size, int sizemap, string fname, bool bsave, bool bAccelStruct)
{
	PhotonRecord* photons_data    = reinterpret_cast<PhotonRecord*>( pdata->map() );
	PhotonRecord* photon_map_data = NULL;
	if(bAccelStruct){
		photon_map_data = reinterpret_cast<PhotonRecord*>( pmap->map() );
		for( unsigned int i = 0; i < sizemap; ++i ) {
			photon_map_data[i].energy = make_float3( 0.0f );
		}
	}

	// Push all valid photons to front of list
	unsigned int valid_photons = 0;
	PhotonRecord** temp_photons = new PhotonRecord*[size];
	for( unsigned int i = 0; i < size; ++i ) {
		if( fmaxf( photons_data[i].energy ) > 0.0f ) {
			temp_photons[valid_photons++] = &photons_data[i];
		}
	}
	if ( _display_debug_buffer ) {
		std::cerr << " ** valid_photon/NUM_PHOTONS =  " 
			<< valid_photons<<"/"<<size
			<<" ("<<valid_photons/static_cast<float>(size)<<")\n";
	}

	// Make sure we arent at most 1 less than power of 2
	valid_photons = valid_photons >= sizemap ? sizemap : valid_photons;

	float3 bbmin = make_float3(0.0f);
	float3 bbmax = make_float3(0.0f);
	if( _split_choice == LongestDim ) {
		bbmin = make_float3(  std::numeric_limits<float>::max() );
		bbmax = make_float3( -std::numeric_limits<float>::max() );
		// Compute the bounds of the photons
		for(unsigned int i = 0; i < valid_photons; ++i) {
			float3 position = (*temp_photons[i]).position;
			bbmin = fminf(bbmin, position);
			bbmax = fmaxf(bbmax, position);
		}
	}

	if(bsave)
		SavePhotonMap(temp_photons, valid_photons, fname.c_str());//"c:\\SVN\\lightmap.bin");
	// Now build KD tree
	if(bAccelStruct)
		buildKDTree( temp_photons, 0, valid_photons, 0, photon_map_data, 0, _split_choice, bbmin, bbmax );

	delete[] temp_photons;
	if(photon_map_data)
		pmap->unmap();
	pdata->unmap();
}

void ProgressivePhotonScene::createVolPhotonMap()
{
	PhotonRecord* photons_data    = reinterpret_cast<PhotonRecord*>( _vol_photons->map() );
	PhotonRecord* photon_map_data = reinterpret_cast<PhotonRecord*>( _vol_photon_map->map() );

	for( unsigned int i = 0; i < _vol_photon_map_size; ++i ) {
		photon_map_data[i].energy = make_float3( 0.0f );
	}

	// Push all valid photons to front of list
	unsigned int valid_photons = 0;
	PhotonRecord** temp_photons = new PhotonRecord*[NUM_VOL_PHOTONS];
	for( unsigned int i = 0; i < NUM_VOL_PHOTONS; ++i ) {
		if( fmaxf( photons_data[i].energy ) > 0.0f ) {
			temp_photons[valid_photons++] = &photons_data[i];
		}
	}
	if ( _display_debug_buffer ) {
		std::cerr << " ** valid_photon/NUM_PHOTONS =  " 
			<< valid_photons<<"/"<<NUM_VOL_PHOTONS
			<<" ("<<valid_photons/static_cast<float>(NUM_VOL_PHOTONS)<<")\n";
	}

	// Make sure we arent at most 1 less than power of 2
	valid_photons = valid_photons >= _vol_photon_map_size ? _vol_photon_map_size : valid_photons;

	float3 bbmin = make_float3(0.0f);
	float3 bbmax = make_float3(0.0f);
	if( _split_choice == LongestDim ) {
		bbmin = make_float3(  std::numeric_limits<float>::max() );
		bbmax = make_float3( -std::numeric_limits<float>::max() );
		// Compute the bounds of the photons
		for(unsigned int i = 0; i < valid_photons; ++i) {
			float3 position = (*temp_photons[i]).position;
			bbmin = fminf(bbmin, position);
			bbmax = fmaxf(bbmax, position);
		}
	}
	/*
	for(unsigned int i = 0; i < valid_photons; ++i) {
	float3 position = (*temp_photons[i]).position;
	std::cerr << "position[" << i << "]= " 
	<< position.x <<", " <<  position.y << ", " << position.z << "\n";
	}
	std::cerr << "_____________" << "\n";
	*/

	SavePhotonMap(temp_photons, valid_photons, "c:\\SVN\\volumemap_v.bin");
	// Now build KD tree
	buildKDTree( temp_photons, 0, valid_photons, 0, photon_map_data, 0, _split_choice, bbmin, bbmax );

	delete[] temp_photons;
	_vol_photon_map->unmap();
	_vol_photons->unmap();
}


void ProgressivePhotonScene::createRTPointMap()
{
	PhotonRecord* photons_data    = reinterpret_cast<PhotonRecord*>( _rt_points->map() );
	PhotonRecord* photon_map_data = reinterpret_cast<PhotonRecord*>( _rt_point_map->map() );

	for( unsigned int i = 0; i < _vol_photon_map_size; ++i ) {
		photon_map_data[i].energy = make_float3( 0.0f );
	}

	// Push all valid photons to front of list
	unsigned int valid_photons = 0;
	PhotonRecord** temp_photons = new PhotonRecord*[NUM_VOL_PHOTONS];
	for( unsigned int i = 0; i < NUM_VOL_PHOTONS; ++i ) {
		if( fmaxf( photons_data[i].energy ) > 0.0f ) {
			temp_photons[valid_photons++] = &photons_data[i];
		}
	}
	if ( _display_debug_buffer ) {
		std::cerr << " ** valid_photon/NUM_PHOTONS =  " 
			<< valid_photons<<"/"<<NUM_VOL_PHOTONS
			<<" ("<<valid_photons/static_cast<float>(NUM_VOL_PHOTONS)<<")\n";
	}

	// Make sure we arent at most 1 less than power of 2
	valid_photons = valid_photons >= _rt_point_map_size ? _rt_point_map_size : valid_photons;

	float3 bbmin = make_float3(0.0f);
	float3 bbmax = make_float3(0.0f);
	if( _split_choice == LongestDim ) {
		bbmin = make_float3(  std::numeric_limits<float>::max() );
		bbmax = make_float3( -std::numeric_limits<float>::max() );
		// Compute the bounds of the photons
		for(unsigned int i = 0; i < valid_photons; ++i) {
			float3 position = (*temp_photons[i]).position;
			bbmin = fminf(bbmin, position);
			bbmax = fmaxf(bbmax, position);
		}
	}
	/*
	for(unsigned int i = 0; i < valid_photons; ++i) {
	float3 position = (*temp_photons[i]).position;
	std::cerr << "position[" << i << "]= " 
	<< position.x <<", " <<  position.y << ", " << position.z << "\n";
	}
	std::cerr << "_____________" << "\n";
	*/

	SavePhotonMap(temp_photons, valid_photons, "c:\\SVN\\volumemap_rt.bin");
	// Now build KD tree
	buildKDTree( temp_photons, 0, valid_photons, 0, photon_map_data, 0, _split_choice, bbmin, bbmax );

	delete[] temp_photons;
	_rt_point_map->unmap();
	_rt_points->unmap();
}

void ProgressivePhotonScene::setObjFilename(string filename) 
{
	_objfilename = filename;	
}

void ProgressivePhotonScene::setPhotonFilename(string filename) 
{
	_photonfilename = filename;	
}

void ProgressivePhotonScene::SavePhotonMap(PhotonRecord** temp_photons, int num_photons, const char* filename)
{
	if(!temp_photons || !filename)
		return;
	//char filename[1024] = "c:\\SVN\\volumemap.bin";

	ofstream outf(filename, ios::binary);
	assert(outf);	
	if(!outf)
		return;

	int size = num_photons;
	const int COLOR_SAMPLES = 3;
	int cs = COLOR_SAMPLES;
	outf.write(reinterpret_cast<char *>(&size), sizeof(int));
	outf.write(reinterpret_cast<char *>(&cs), sizeof(int));

	for(int i=0; i < num_photons; i++){
		float3 pos = (*temp_photons[i]).position;
		outf.write(reinterpret_cast<char *>(&pos.x), sizeof(float));
		outf.write(reinterpret_cast<char *>(&pos.y), sizeof(float));
		outf.write(reinterpret_cast<char *>(&pos.z), sizeof(float));

		float3 dir = (*temp_photons[i]).ray_dir;
		float3 energy = (*temp_photons[i]).energy;
		//for(int j = 0; j < COLOR_SAMPLES; j++){
		//	float c = 1.0;
		outf.write(reinterpret_cast<char *>(&dir.x), sizeof(float));
		outf.write(reinterpret_cast<char *>(&dir.y), sizeof(float));
		outf.write(reinterpret_cast<char *>(&dir.z), sizeof(float));
		//}
	}
	//outf.write(reinterpret_cast<char *>(&x_start), sizeof(int));
	//outf.write(reinterpret_cast<char *>(&x_end), sizeof(int));	
	//outf.write(reinterpret_cast<char *>(&y_start), sizeof(int));

	outf.close();
}

void ProgressivePhotonScene::optix_display(float3 in_c, float3 in_s, const float* ip, const float* iv)
{
	try {
		// render the scene
		float3 eye, U, V, W;
		disp->_camera->getEyeUVW( eye, U, V, W );
		// Don't be tempted to just start filling in the values outside of a constructor, 
		// because if you add a parameter it's easy to forget to add it here.
		SampleScene::RayGenCameraData camera_data( eye, U, V, W );

		_iviewf = Matrix4x4(iv);
		_iproj = Matrix4x4(ip);
		_cam_pos = make_float4(in_c.x,in_c.y,in_c.z,1.0);
		trace( camera_data );

		// Always count rendered frames
		//++_frame_count;

		// Not strictly necessary
		//glClearColor(0.0, 0.0, 0.0, 0.0);
		//glClear(GL_COLOR_BUFFER_BIT);

		//if( _display_frames ) {
		displayFrame();
		//}
	} catch( Exception& e ){
		sutilReportError( e.getErrorString().c_str() );
		//exit(2);
	}

	// Do not draw text on 1st frame -- issue on linux causes problems with 
	// glDrawPixels call if drawText glutBitmapCharacter is called on first frame.
#ifdef DISPLAY_FRAME_INFO
	if ( _display_fps && _cur_continuous_mode != CDNone && _frame_count > 1 ) {
		// Output fps 
		double current_time;
		sutilCurrentTime( &current_time );
		double dt = current_time - _last_frame_time;
		if( dt > _fps_update_threshold ) {
			sprintf( _fps_text, "fps: %7.2f", (_frame_count - _last_frame_count) / dt );

			_last_frame_time = current_time;
			_last_frame_count = _frame_count;
		} else if( _frame_count == 1 ) {
			sprintf( _fps_text, "fps: %7.2f", 0.f );
		}

		drawText( _fps_text, 10.0f, 10.0f, GLUT_BITMAP_8_BY_13 );
	}

	if( _print_mem_usage ) {
		// Output memory
		std::ostringstream str;
		DeviceMemoryLogger::logCurrentMemoryUsage(_scene->getContext(), str);
		drawText( str.str(), 10.0f, 26.0f, GLUT_BITMAP_8_BY_13 );
	}
#endif //DISPLAY_FRAME_INFO
}

void ProgressivePhotonScene::trace( const RayGenCameraData& camera_data )
{
	if (! m_camera_changed ){
		return;
	}
	_print_timings = true;
	m_context["iview_matrix_row_0"]->setFloat(_iviewf[0*4+0], _iviewf[0*4+1], _iviewf[0*4+2], _iviewf[0*4+3]);
	m_context["iview_matrix_row_1"]->setFloat(_iviewf[1*4+0], _iviewf[1*4+1], _iviewf[1*4+2], _iviewf[1*4+3]);
	m_context["iview_matrix_row_2"]->setFloat(_iviewf[2*4+0], _iviewf[2*4+1], _iviewf[2*4+2], _iviewf[2*4+3]);
	m_context["iview_matrix_row_3"]->setFloat(_iviewf[3*4+0], _iviewf[3*4+1], _iviewf[3*4+2], _iviewf[3*4+3]);

	m_context["iproj_matrix_row_0"]->setFloat(_iproj[0*4+0], _iproj[0*4+1], _iproj[0*4+2], _iproj[0*4+3]);
	m_context["iproj_matrix_row_1"]->setFloat(_iproj[1*4+0], _iproj[1*4+1], _iproj[1*4+2], _iproj[1*4+3]);
	m_context["iproj_matrix_row_2"]->setFloat(_iproj[2*4+0], _iproj[2*4+1], _iproj[2*4+2], _iproj[2*4+3]);
	m_context["iproj_matrix_row_3"]->setFloat(_iproj[3*4+0], _iproj[3*4+1], _iproj[3*4+2], _iproj[3*4+3]);

	m_context["new_eye"]->setFloat(_cam_pos);


	Buffer output_buffer = m_context["rtpass_output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	output_buffer->getSize( buffer_width, buffer_height );


	_frame_number = m_camera_changed ? 0u : _frame_number+1;
	m_context["frame_number"]->setFloat( static_cast<float>(_frame_number) );
	m_context["full_gather"]->setInt((int)_full_gather);
	m_context["exposure"]->setFloat(_exposure);
	m_context["singleOnly"]->setInt(_singleOnly);
	m_context["hdrOn"]->setInt((int)_hdrOn);
	if ( m_camera_changed ) {
		//m_camera_changed = false;
		m_context["rtpass_eye"]->setFloat( camera_data.eye );
		m_context["rtpass_U"]->setFloat( camera_data.U );
		m_context["rtpass_V"]->setFloat( camera_data.V );
		m_context["rtpass_W"]->setFloat( camera_data.W );

		_iteration_count=1;
	}
	//m_context["filter_type"]->setInt( FILTER_GAUSSIAN );
	//m_context["filter_width"]->setFloat( 1.0f );
	//m_context["gaussian_alpha"]->setFloat( 0.1f );

	// Trace photons
	if(_update_pmap && !isSingle()&& !_finishedPhotonTracing){
		//string fname = "c:\\SVN\\vol_bin_counts.bin";
		if(!loadPhotonBinCounts(_photonfilename)){
			double t0, t1;
			sutilCurrentTime( &t0 );

			int _iter_photon_map = 0;
			int NUM_PHOTON_ITER = 50;//100;
			if (_print_timings) std::cerr << "Starting photon pass   ... ";
			for(int _iter_photon_map=0; _iter_photon_map < NUM_PHOTON_ITER; _iter_photon_map++){
				cout << "Photon pass: " << _iter_photon_map << endl;
				for(int cb =0; cb < 3; cb++){
					cout << "---->color pass: " << cb << endl;
					m_context["color_band"]->setInt( cb );
					Buffer photon_rnd_seeds = m_context["photon_rnd_seeds"]->getBuffer();
					uint2* seeds = reinterpret_cast<uint2*>( photon_rnd_seeds->map() );
					for ( unsigned int i = 0; i < PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT*MAX_VOL_PHOTON_COUNT; ++i )
						seeds[i] = random2u();
					photon_rnd_seeds->unmap();
					
					m_context["photon_rnd_seeds"]->set( photon_rnd_seeds );

					memset( _vol_counts_photons->map(), 0, NUM_VOL_PHOTONS*sizeof( PhotonCountRecord ) );
					_vol_counts_photons->unmap();
					m_context["ppass_vol_counts_output_buffer"]->set( _vol_counts_photons );
					
					


					m_context->launch( ppass,
							static_cast<unsigned int>(PHOTON_LAUNCH_WIDTH),
							static_cast<unsigned int>(PHOTON_LAUNCH_HEIGHT) );

					// By computing the total number of photons as an unsigned long long we avoid 32 bit
					// floating point addition errors when the number of photons gets sufficiently large
					// (the error of adding two floating point numbers when the mantissa bits no longer
					// overlap).
					m_context["total_emitted"]->setFloat( static_cast<float>((unsigned long long)_iteration_count*PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT) );
					//sutilCurrentTime( &t1 );
					//if (_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

					// Build KD tree 
					if (_print_timings) std::cerr << "Starting kd_tree build ... ";
					sutilCurrentTime( &t0 );
					createPhotonMap();
					createVolPhotonMap();

					createPhotonRecordMap(_light_sample, _light_sample_map, 
						PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT, 
						PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT, 
						"c:\\SVN\\lightmap.bin", false, false);

					createPhotonRecordMap(_lightplane_sample, _lightplane_sample_map, 
						PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT, 
						PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT, 
						"c:\\SVN\\lightplanemap.bin", false, false);

					createPhotonRecordMap(_vol_rotate, _vol_rotate_map, 
						NUM_VOL_PHOTONS, 
						NUM_VOL_PHOTONS, 
						"c:\\SVN\\vol_rotate.bin", false, false);
					
					savePhotonBinCounts(_vol_counts_photons, 
						NUM_VOL_PHOTONS,
						"c:\\SVN\\vol_bin_counts.bin", cb, (_iter_photon_map == NUM_PHOTON_ITER-1 && cb == 2));

					//createRTPointMap();
					


					////////////////////////////////////////////////////////////////////////////////////
					/*
					if(_iter_photon_map == NUM_PHOTON_ITER-1 && cb == 2){
						if (_print_timings) std::cerr << "Starting pbin pass   ... ";
						sutilCurrentTime( &t0 );

						m_context->launch( pbinpass,
								static_cast<unsigned int>(RES_MU_S * RES_NU),
								static_cast<unsigned int>(RES_MU) );
						// By computing the total number of photons as an unsigned long long we avoid 32 bit
						// floating point addition errors when the number of photons gets sufficiently large
						// (the error of adding two floating point numbers when the mantissa bits no longer
						// overlap).
						sutilCurrentTime( &t1 );
						if (_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;
					}
					*/

					//savePhotonBins(_pbin_buffer, 
					//	RES_R, RES_MU_S * RES_NU, RES_MU,
					//	"c:\\SVN\\ascii_vol_bin_counts.txt");
				} //color_bands
			} //number of photon passes
			sutilCurrentTime( &t1 );
			if (_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;
		}

		string fname_tbl = "C:\\SVN\\ascii_vol_count_hist.txt";
		savePhotonBinCountsHist(_photonfilename, fname_tbl, true);
		_finishedPhotonTracing = true;

		
		{
		

		//_pbin_buffer
		float* tmp = new float[RES_MU_S * RES_NU * RES_MU * RES_R * 4];
		
		for(int r=0; r < RES_R; r++){
			m_context["gR_ind"]->setInt(r);
			cout << "view ray texture: altitude " << r << endl;
			double t0, t1;
			if (_print_timings) std::cerr << "Starting pbin pass   ... ";
			sutilCurrentTime( &t0 );

			m_context->launch( pbinpass,
					static_cast<unsigned int>(RES_MU_S * RES_NU),
					static_cast<unsigned int>(RES_MU));
			// By computing the total number of photons as an unsigned long long we avoid 32 bit
			// floating point addition errors when the number of photons gets sufficiently large
			// (the error of adding two floating point numbers when the mantissa bits no longer
			// overlap).
			sutilCurrentTime( &t1 );
			if (_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;
			loadBinCountsTexture(tmp, r,  RES_MU_S * RES_NU, RES_MU, RES_R, 4);
		}
		create3DOptixTexFromBuffer("inscatterPhotonSampler", tmp, RES_MU_S * RES_NU, RES_MU, RES_R, 4, 1, false);
		delete [] tmp;
		//loadBinCountsTexture();
		
		}

	}

	if ( m_camera_changed ) {
		//cout << "Camera changed!!!" << endl;
		m_camera_changed = false;
		m_context["rtpass_eye"]->setFloat( camera_data.eye );
		m_context["rtpass_U"]->setFloat( camera_data.U );
		m_context["rtpass_V"]->setFloat( camera_data.V );
		m_context["rtpass_W"]->setFloat( camera_data.W );
		
		if(_finishedPhotonTracing){
			m_context["rtpass_global_photon_counts"]->set( _global_photon_counts );
			//testPhotonBinCounts();
		}

		// Trace viewing rays
		if (_print_timings) std::cerr << "Starting RT pass ... ";
		std::cerr.flush();
		double t0, t1;
		sutilCurrentTime( &t0 );
		m_context->launch( rtpass,
			static_cast<unsigned int>(buffer_width*sqrt_samples_per_pixel),
			static_cast<unsigned int>(buffer_height*sqrt_samples_per_pixel) );
		sutilCurrentTime( &t1 );
		//if(_finishedPhotonTracing){
		//	testPhotonBinCounts();
		//}
		if (_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;
		m_context["total_emitted"]->setFloat(  0.0f );
		//_iteration_count=1;



		//m_context->launch(gather_samples, static_cast<unsigned int>(buffer_width), 
		//								 static_cast<unsigned int>(buffer_height) );
	}

	double t0, t1;
	// Shade view rays by gathering photons
	if (_print_timings) std::cerr << "Starting gather pass   ... ";
	sutilCurrentTime( &t0 );
	m_context->launch( gather,
		static_cast<unsigned int>(buffer_width),
		static_cast<unsigned int>(buffer_height) );
	sutilCurrentTime( &t1 );
	if (_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

	// Debug output
	if( _display_debug_buffer ) {
		sutilCurrentTime( &t0 );
		float4* debug_data = reinterpret_cast<float4*>( _debug_buffer->map() );
		Buffer hit_records = m_context["rtpass_output_buffer"]->getBuffer();
		HitRecord* hit_record_data = reinterpret_cast<HitRecord*>( hit_records->map() );
		float4 avg  = make_float4( 0.0f );
		float4 minv = make_float4( std::numeric_limits<float>::max() );
		float4 maxv = make_float4( 0.0f );
		float counter = 0.0f;
		for( unsigned int j = 0; j < buffer_height; ++j ) {
			for( unsigned int i = 0; i < buffer_width; ++i ) {
				///*
				if( i < 10 && j < 10 && 0) {
					fprintf( stderr, " %08.4f %08.4f %08.4f %08.4f\n", debug_data[j*buffer_width+i].x,
						debug_data[j*buffer_width+i].y,
						debug_data[j*buffer_width+i].z,
						debug_data[j*buffer_width+i].w );
				}
				//*/


				if( hit_record_data[j*buffer_width+i].flags & PPM_HIT ) {
					float4 val = debug_data[j*buffer_width+i];
					avg += val;
					minv = fminf(minv, val);
					maxv = fmaxf(maxv, val);
					counter += 1.0f;
				}
			}
		}
		_debug_buffer->unmap();
		hit_records->unmap();

		avg = avg / counter; 
		sutilCurrentTime( &t1 );
		if (_print_timings) std::cerr << "Stat collection time ...           " << t1 - t0 << std::endl;
		std::cerr << "(min, max, average):"
			<< " loop iterations: ( "
			<< minv.x << ", "
			<< maxv.x << ", "
			<< avg.x << " )"
			<< " radius: ( "
			<< minv.y << ", "
			<< maxv.y << ", "
			<< avg.y << " )"
			<< " N: ( "
			<< minv.z << ", "
			<< maxv.z << ", "
			<< avg.z << " )"
			<< " M: ( "
			<< minv.w << ", "
			<< maxv.w << ", "
			<< avg.w << " )";
		std::cerr << ", total_iterations = "<<_iteration_count;
		std::cerr << std::endl;
	}
	_iteration_count++;
}

void ProgressivePhotonScene::displayFrame()
{
  GLboolean sRGB = GL_FALSE;
  //if (_use_sRGB && _sRGB_supported) {
  //  glGetBooleanv( GL_FRAMEBUFFER_SRGB_CAPABLE_EXT, &sRGB );
  //  if (sRGB) {
  //    glEnable(GL_FRAMEBUFFER_SRGB_EXT);
  //  }
  //}

  // Draw the resulting image
  Buffer buffer = getOutputBuffer(); 
  RTsize buffer_width_rts, buffer_height_rts;
  buffer->getSize( buffer_width_rts, buffer_height_rts );
  int buffer_width  = static_cast<int>(buffer_width_rts);
  int buffer_height = static_cast<int>(buffer_height_rts);
  RTformat buffer_format = buffer->getFormat();

  //if( _save_frames_to_file ) {
  //  static char fname[128];
  //  std::string basename = _save_frames_basename.empty() ? "frame" : _save_frames_basename;
  //  sprintf(fname, "%s_%05d.ppm", basename.c_str(), _frame_count);
  //  sutilDisplayFilePPM( fname, buffer->get() );
  //}

  unsigned int vboId = 0;
  if( usesVBOBuffer() == true )
    vboId = buffer->getGLBOId();
  //vboId = 0;
  if (vboId)
  {
	//cout << "ERROR: Fix this!!! we are in ProgressivePhotonScene::displayFrame() vboId option" << endl;
	 
    if (!_texId)
    {
	  glActiveTexture(GL_TEXTURE0 + _texUnit);
      glGenTextures( 1, &_texId );
      glBindTexture( GL_TEXTURE_2D, _texId);

      // Change these to GL_LINEAR for super- or sub-sampling
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

      // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

      glBindTexture( GL_TEXTURE_2D, 0);
    }

	glActiveTexture(GL_TEXTURE0 + _texUnit);
    glBindTexture( GL_TEXTURE_2D, _texId );

    // send pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vboId);

    RTsize elementSize = buffer->getElementSize();
    if      ((elementSize % 8) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ((elementSize % 4) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ((elementSize % 2) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                             glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if(buffer_format == RT_FORMAT_UNSIGNED_BYTE4) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, buffer_width, buffer_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    } else if(buffer_format == RT_FORMAT_FLOAT4) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, buffer_width, buffer_height, 0, GL_RGBA, GL_FLOAT, 0);
    } else if(buffer_format == RT_FORMAT_FLOAT3) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, buffer_width, buffer_height, 0, GL_RGB, GL_FLOAT, 0);
    } else if(buffer_format == RT_FORMAT_FLOAT) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, buffer_width, buffer_height, 0, GL_LUMINANCE, GL_FLOAT, 0);
    } else {
      assert(0 && "Unknown buffer format");
    }

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
/*
    glEnable(GL_TEXTURE_2D);

    // Initialize offsets to pixel center sampling.

    float u = 0.5f/buffer_width;
    float v = 0.5f/buffer_height;

    glBegin(GL_QUADS);
    glTexCoord2f(u, v);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, v);
    glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f - u, 1.0f - v);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(u, 1.0f - v);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    glDisable(GL_TEXTURE_2D);
	*/
	
  } else {
    GLvoid* imageData = buffer->map();
    assert( imageData );

    GLenum gl_data_type = GL_FALSE;
    GLenum gl_format = GL_FALSE;

    switch (buffer_format) {
          case RT_FORMAT_UNSIGNED_BYTE4:
            gl_data_type = GL_UNSIGNED_BYTE;
            gl_format    = GL_BGRA;
            break;

          case RT_FORMAT_FLOAT:
            gl_data_type = GL_FLOAT;
            gl_format    = GL_LUMINANCE;
            break;

          case RT_FORMAT_FLOAT3:
            gl_data_type = GL_FLOAT;
            gl_format    = GL_RGB;
            break;

          case RT_FORMAT_FLOAT4:
            gl_data_type = GL_FLOAT;
            gl_format    = GL_RGBA;
            break;

          default:
            fprintf(stderr, "Unrecognized buffer data type or format.\n");
            exit(2);
            break;
    }

    glDrawPixels( static_cast<GLsizei>( buffer_width),
      static_cast<GLsizei>( buffer_height ),
      gl_format, gl_data_type, imageData);

    buffer->unmap();
  }
  //if (_use_sRGB && _sRGB_supported && sRGB) {
  //  glDisable(GL_FRAMEBUFFER_SRGB_EXT);
  //}
}

void ProgressivePhotonScene::doResize( unsigned int width, unsigned int height )
{
	// display buffer resizing handled in base class
	m_context["rtpass_output_buffer"]->getBuffer()->setSize( width, height );
	//m_context["rtpass_output_samples"]->getBuffer()->setSize( width*sqrt_samples_per_pixel, height*sqrt_samples_per_pixel );
	m_context["output_buffer"       ]->getBuffer()->setSize( width, height );
	m_context["image_rnd_seeds"     ]->getBuffer()->setSize( width, height );
	m_context["debug_buffer"        ]->getBuffer()->setSize( width, height );

	Buffer image_rnd_seeds = m_context["image_rnd_seeds"]->getBuffer();
	uint2* seeds = reinterpret_cast<uint2*>( image_rnd_seeds->map() );
	for ( unsigned int i = 0; i < width*height; ++i )  
		seeds[i] = random2u();
	image_rnd_seeds->unmap();
}

GeometryInstance ProgressivePhotonScene::createCone(const float3& color, const float radius, const float height, int up_cone, bool bGather)
{
	const float3 blue = make_float3(0.05f, 0.05f, 0.8f );
	Geometry sphere = m_context->createGeometry();
	sphere->setPrimitiveCount( 1u );
	std::string ptx_path = ptxpath( "isgShadows", "sphere.cu" );
	//Program sphere_bounding_box = m_context->createProgramFromPTXFile( ptx_path, "bounds_atmosphere" );
	//Program sphere_intersection = m_context->createProgramFromPTXFile( ptx_path, "intersect_atmosphere" );
	Program sphere_bounding_box    = m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "sphere.cu" ), "cone_bounds" );
	Program sphere_intersection = m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "sphere.cu" ), "cone_intersect" );
	sphere->setIntersectionProgram( sphere_intersection );
	sphere->setBoundingBoxProgram( sphere_bounding_box );

	//sphere["sphere"]->setFloat( 200.0f, 50.0f, 150.0f, radius);
	//sphere["sphere"]->setFloat( 0.0f, 0.0f, 0.0f, radius);
	sphere["cone_data"]->setFloat(make_float3(radius, height, M_PI*2));
	sphere["up_cone"]->setInt(up_cone);
	sphere["Rg"]->setFloat(Rg);
	/*
	float3 normal = normalize( cross( offset1, offset2 ) );
	float d       = dot( normal, anchor );
	float4 plane  = make_float4( normal, d );

	float3 v1 = offset1 / dot( offset1, offset1 );
	float3 v2 = offset2 / dot( offset2, offset2 );

	sphere["plane"]->setFloat( plane );
	sphere["anchor"]->setFloat( anchor );
	sphere["v1"]->setFloat( v1 );
	sphere["v2"]->setFloat( v2 );
	*/
	Material sphere_matl = m_context->createMaterial();
	sphere_matl["max_depth"]->setUint( 20 );
	sphere_matl["radiance_ray_type"]->setUint( 0u );
	sphere_matl["shadow_ray_type"]->setUint( 1u );
	sphere_matl["scene_epsilon"]->setFloat( 1.e-3f );

	if(bGather){
		sphere_matl["Rt"]->setFloat(Rt);
		sphere_matl["Rg"]->setFloat(Rg);
		sphere_matl["RL"]->setFloat(RL);
		sphere_matl["TRANSMITTANCE_INTEGRAL_SAMPLES"]->setFloat(TRANSMITTANCE_INTEGRAL_SAMPLES);
		sphere_matl->setClosestHitProgram( rtpass, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
		"transit_closest_hit_ext") );
		sphere_matl->setAnyHitProgram(shadow, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
		"transit_any_hit") );
	}
	else{
		sphere_matl->setClosestHitProgram( rtpass, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
		"rtpass_closest_hit") );//"rtpass_mesh_closest_hit") );
	
	//sphere_matl->setClosestHitProgram( ppass,  m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_ppass.cu"),
	//	"ppass_closest_hit") );
	sphere_matl->setAnyHitProgram(     gather,  m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_gather.cu"),
	                                 "gather_any_hit") );
	sphere_matl->setAnyHitProgram(shadow, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
		"shadow_any_hit") );
	}
	//box_matl["Kd"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	sphere_matl["Kd"]->setFloat( blue );
	sphere_matl["Ks"]->setFloat( make_float3( 0.2f, 0.2f, 0.2f ) );
	sphere_matl["Ka"]->setFloat( make_float3( 0.05f, 0.05f, 0.05f ) );
	sphere_matl[ "emissive" ]->setFloat( 0.0f, 0.0f, 0.0f );
	sphere_matl["phong_exp"]->setFloat(0.0f);
	sphere_matl["refraction_index"]->setFloat( 1.2f );
	//sphere_matl["ambient_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.2f, 0.2f, 0.2f ) ) );
	//sphere_matl["diffuse_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.8f, 0.8f, 0.8f ) ) );
	//sphere_matl["specular_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.0f, 0.0f, 0.0f ) ) );
	sphere_matl["shadow_attenuation"]->setFloat( 0.4f, 0.7f, 0.4f );
	float color_scale = 1;
	float3 Kd = make_float3( color_scale, 1.0f - color_scale, 1.0f );
	//sphere_matl["transmissive_map"]->setTextureSampler( loadTexture( m_context, "", Kd ) );




	GeometryInstance gi = m_context->createGeometryInstance();

	gi->setGeometry( sphere );
	gi->setMaterialCount( 1 );
	gi->setMaterial( 0, sphere_matl );

	gi["Kd"]->setFloat( color );
	gi["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
	gi["use_grid"]->setUint( 0u );
	gi["grid_color"]->setFloat( make_float3( 0.0f ) );
	gi["emitted"]->setFloat( 0.0f, 0.0f, 0.0f );
	return gi;
}

GeometryInstance ProgressivePhotonScene::createSphere(const float3& color, const float radius, bool bGather)
{
	const float3 white = make_float3( 0.8f, 0.8f, 0.8f );
	const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
	const float3 blue = make_float3(0.05f, 0.05f, 0.8f );
	const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
	const float3 black = make_float3( 0.0f, 0.0f, 0.0f );
	const float3 light = make_float3( 15.0f, 15.0f, 5.0f );

	Geometry sphere = m_context->createGeometry();
	sphere->setPrimitiveCount( 1u );
	std::string ptx_path = ptxpath( "isgShadows", "sphere.cu" );
	//Program sphere_bounding_box = m_context->createProgramFromPTXFile( ptx_path, "bounds_atmosphere" );
	//Program sphere_intersection = m_context->createProgramFromPTXFile( ptx_path, "intersect_atmosphere" );
	Program sphere_bounding_box    = m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "sphere.cu" ), "tex_bounds" );
	Program sphere_intersection = m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "sphere.cu" ), "tex_intersect" );
	sphere->setIntersectionProgram( sphere_intersection );
	sphere->setBoundingBoxProgram( sphere_bounding_box );

	//sphere["sphere"]->setFloat( 200.0f, 50.0f, 150.0f, radius);
	//sphere["up_cone"]->setInt(0);
	sphere["sphere"]->setFloat( 0.0f, 0.0f, 0.0f, radius);

	/*
	float3 normal = normalize( cross( offset1, offset2 ) );
	float d       = dot( normal, anchor );
	float4 plane  = make_float4( normal, d );

	float3 v1 = offset1 / dot( offset1, offset1 );
	float3 v2 = offset2 / dot( offset2, offset2 );

	sphere["plane"]->setFloat( plane );
	sphere["anchor"]->setFloat( anchor );
	sphere["v1"]->setFloat( v1 );
	sphere["v2"]->setFloat( v2 );
	*/
	Material sphere_matl = m_context->createMaterial();

	sphere_matl["max_depth"]->setUint( 20 );
	sphere_matl["radiance_ray_type"]->setUint( 0u );
	sphere_matl["shadow_ray_type"]->setUint( 1u );
	sphere_matl["scene_epsilon"]->setFloat( 1.e-3f );
	
	if(bGather){
		sphere_matl["Rt"]->setFloat(Rt);
		sphere_matl["Rg"]->setFloat(Rg);
		sphere_matl["RL"]->setFloat(RL);
		sphere_matl["TRANSMITTANCE_INTEGRAL_SAMPLES"]->setFloat(TRANSMITTANCE_INTEGRAL_SAMPLES);
		sphere_matl->setClosestHitProgram( rtpass, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
		"transit_closest_hit_ext") );
	}
	else{
		sphere_matl->setClosestHitProgram( rtpass, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
			"rtpass_closest_hit") );
	
	sphere_matl->setClosestHitProgram( ppass,  m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_ppass.cu"),
		"ppass_closest_hit") );
	//sphere_matl->setAnyHitProgram(     gather,  m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_gather.cu"),
	//                                 "gather_any_hit") );
	sphere_matl->setAnyHitProgram(shadow, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
		"shadow_any_hit") );
	}
	//box_matl["Kd"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	sphere_matl["Kd"]->setFloat( blue );
	sphere_matl["Ks"]->setFloat( make_float3( 0.2f, 0.2f, 0.2f ) );
	sphere_matl["Ka"]->setFloat( make_float3( 0.05f, 0.05f, 0.05f ) );
	sphere_matl[ "emissive" ]->setFloat( 0.0f, 0.0f, 0.0f );
	sphere_matl["phong_exp"]->setFloat(0.0f);
	sphere_matl["refraction_index"]->setFloat( 1.2f );
	//sphere_matl["ambient_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.2f, 0.2f, 0.2f ) ) );
	//sphere_matl["diffuse_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.8f, 0.8f, 0.8f ) ) );
	//sphere_matl["specular_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.0f, 0.0f, 0.0f ) ) );
	sphere_matl["shadow_attenuation"]->setFloat( 0.4f, 0.7f, 0.4f );
	float color_scale = 1;
	float3 Kd = make_float3( color_scale, 1.0f - color_scale, 1.0f );
	sphere_matl["transmissive_map"]->setTextureSampler( loadTexture( m_context, "", Kd ) );




	GeometryInstance gi = m_context->createGeometryInstance();

	gi->setGeometry( sphere );
	gi->setMaterialCount( 1 );
	gi->setMaterial( 0, sphere_matl );

	gi["Kd"]->setFloat( color );
	gi["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
	gi["use_grid"]->setUint( 0u );
	gi["grid_color"]->setFloat( make_float3( 0.0f ) );
	gi["emitted"]->setFloat( 0.0f, 0.0f, 0.0f );
	return gi;
}

GeometryInstance ProgressivePhotonScene::createVolSphere(const float3& color, const float radiusT, const float radiusG)
{
	const float3 white = make_float3( 0.8f, 0.8f, 0.8f );
	const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
	const float3 blue = make_float3(0.05f, 0.05f, 0.8f );
	const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
	const float3 black = make_float3( 0.0f, 0.0f, 0.0f );
	const float3 light = make_float3( 15.0f, 15.0f, 5.0f );

	Geometry sphere = m_context->createGeometry();
	sphere->setPrimitiveCount( 1u );
	std::string ptx_path = ptxpath( "isgShadows", "sphere.cu" );
	Program sphere_bounding_box    = m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "sphere.cu" ), "bounds_atmosphere" );
	Program sphere_intersection = m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "sphere.cu" ), "intersect_atmosphere" );
	sphere->setIntersectionProgram( sphere_intersection );
	sphere->setBoundingBoxProgram( sphere_bounding_box );

	//sphere["up_cone"]->setInt(0);
	sphere["sphere"]->setFloat( 0.0f, 0.0f, 0.0f, radiusT);


	Material sphere_matl = m_context->createMaterial();

	sphere_matl["Rt"]->setFloat(radiusT);
	sphere_matl["Rg"]->setFloat(radiusG);
	sphere_matl["RL"]->setFloat(RL);
	sphere_matl["TRANSMITTANCE_INTEGRAL_SAMPLES"]->setFloat(TRANSMITTANCE_INTEGRAL_SAMPLES);
	sphere_matl["max_depth"]->setUint( 20 );
	sphere_matl["radiance_ray_type"]->setUint( 0u );
	sphere_matl["shadow_ray_type"]->setUint( 1u );
	sphere_matl["scene_epsilon"]->setFloat( 1.e-3f );

	sphere_matl->setClosestHitProgram( rtpass, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
		"rtpass_vol_closest_hit") );
	sphere_matl->setClosestHitProgram( ppass,  m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_ppass.cu"),
		"ppass_vol_closest_hit_counts") ); //"ppass_vol_closest_hit_ver2"));
	sphere_matl->setAnyHitProgram(     gather,  m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_gather.cu"),
	                                "gather_vol_any_hit") );
	sphere_matl->setAnyHitProgram(shadow, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
		"shadow_any_hit") );

	sphere_matl["Kd"]->setFloat( blue );
	sphere_matl["Ks"]->setFloat( make_float3( 0.2f, 0.2f, 0.2f ) );
	sphere_matl["Ka"]->setFloat( make_float3( 0.05f, 0.05f, 0.05f ) );
	sphere_matl[ "emissive" ]->setFloat( 0.01f, 0.01f, 0.01f );
	sphere_matl["phong_exp"]->setFloat(0.0f);
	sphere_matl["refraction_index"]->setFloat( 1.2f );
	sphere_matl["ambient_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.2f, 0.2f, 0.2f ) ) );
	sphere_matl["diffuse_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.8f, 0.8f, 0.8f ) ) );
	sphere_matl["specular_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.0f, 0.0f, 0.0f ) ) );
	sphere_matl["shadow_attenuation"]->setFloat( 0.4f, 0.7f, 0.4f );
	float color_scale = 1;
	float3 Kd = make_float3( color_scale, 1.0f - color_scale, 1.0f );

	std::string texture_path = std::string( "C:/SVN/Dev/inria/PrecomputedAtmosphericScattering/transparency/" );
	//sphere_matl["transmissive_map"]->setTextureSampler( loadTexture( m_context, texture_path + "sphere_texture.ppm",
	//	make_float3( 0.0f, 0.0f, 0.0f ) ) );

    sphere_matl["transmissive_map"]->setTextureSampler( loadTexture( m_context, "", Kd ) );


	GeometryInstance gi = m_context->createGeometryInstance();

	gi->setGeometry( sphere );
	gi->setMaterialCount( 1 );
	gi->setMaterial( 0, sphere_matl );

	gi["Kd"]->setFloat( color );
	gi["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
	gi["use_grid"]->setUint( 0u );
	gi["grid_color"]->setFloat( make_float3( 0.0f ) );
	gi["emitted"]->setFloat( 0.0f, 0.0f, 0.0f );
	return gi;
}



void ProgressivePhotonScene::create2DOptixTex(std::string texName, unsigned int glTex , optix::TextureSampler & texSampler)
{
	try{
	texSampler = m_context->createTextureSamplerFromGLImage(glTex, RT_TARGET_GL_TEXTURE_2D);
    texSampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    texSampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    texSampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    texSampler->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
    texSampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
    texSampler->setMaxAnisotropy(1.0f);
    texSampler->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
    m_context[texName.c_str()]->setTextureSampler(texSampler);  //Input texture/sampler
	} catch ( optix::Exception& e ) {
    sutilReportError( e.getErrorString().c_str() );
    exit(-1);
	}
}

void ProgressivePhotonScene::create3DOptixTex(std::string texName, unsigned int glTex , optix::TextureSampler &texSampler)
{
	try{
	glDisable(GL_TEXTURE_3D);
	texSampler = m_context->createTextureSamplerFromGLImage(glTex, RT_TARGET_GL_TEXTURE_3D);
    texSampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    texSampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    texSampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    texSampler->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
    texSampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
    texSampler->setMaxAnisotropy(1.0f);
    texSampler->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
    m_context[texName.c_str()]->setTextureSampler(texSampler);  //Input texture/sampler
	glEnable(GL_TEXTURE_3D);
	} catch ( optix::Exception& e ) {
    sutilReportError( e.getErrorString().c_str() );
    exit(-1);
	}
}

void ProgressivePhotonScene::create2DOptixTexFromBuffer(std::string texName, float* data, int width, int height, int dim, bool debug)
{	
	m_context[texName.c_str()]->setTextureSampler(loadTexture2fext(	data, 
																	width, 
																	height,
																	dim,
																	debug));

}

void ProgressivePhotonScene::create2DOptixTexFromBufferuc(std::string texName, unsigned char* data, int width, int height, int dim)
{
	m_context[texName.c_str()]->setTextureSampler(loadTexture2ucext(	data, 
																width, 
																height,
																dim));

}


void ProgressivePhotonScene::create3DOptixTexFromBuffer(std::string texName, float* data, int width, int height, int depth, int dim, int test, bool debug)
{
	m_context[texName.c_str()]->setTextureSampler(loadTexture3f(	data, 
																width, 
																height,
																depth,
																dim,
																test,
																debug));
}

bool ProgressivePhotonScene::loadObjFile_materials(const std::string& model_filename, GeometryGroup geometry_group)
{
	try
	{
		if(model_filename.empty())
			return false;
		optix::Material opaque = m_context->createMaterial();
		opaque->setClosestHitProgram( rtpass, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
			"rtpass_mesh_closest_hit") );
		opaque->setAnyHitProgram(shadow, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
			"shadow_any_hit") );
		ObjLoaderEx loader = ObjLoaderEx( model_filename, m_context, geometry_group, opaque, true); 
		loader.load();

	} catch ( optix::Exception& e ) {
		sutilReportError( e.getErrorString().c_str() );
		exit(-1);
	}
	return true;

}

bool ProgressivePhotonScene::loadObjFile(const std::string& model_filename)
{
	try
	{
		if(model_filename.empty())
			return false;
		
		// Loading model
		_model = new nv::Model();
		if(!_model->loadModelFromFile(model_filename.c_str())) {
			std::cerr << "Unable to load model '" << model_filename << "'" << std::endl;
			exit(-1);
		}

		//Do not forget to handle case of zUp (right now we are assumming yUp)
		
		nv::vec3f* vertices = (nv::vec3f*)_model->getPositions();
		float3 max_old = make_float3(-1000000.0);
		float3 min_old = make_float3(1000000.0);
		float3 max = make_float3(-1000000.0);
		float3 min = make_float3(1000000.0);
		float3 scalef = make_float3(20.0,20.0,20.0);//make_float3(2.0,2.0,2.0);
		for(int i = 0; i < _model->getPositionCount(); ++i) {
			float& x = vertices[i].x;
			float& y = vertices[i].y;
			float& z = vertices[i].z;
			x /= scalef.x;
			y /= scalef.y;
			z /= scalef.z;
			if(max_old.x < x) max_old.x = x;
			if(max_old.y < y) max_old.y = y;
			if(max_old.z < z) max_old.z = z;

			if(min_old.x > x) min_old.x = x;
			if(min_old.y > y) min_old.y = y;
			if(min_old.z > z) min_old.z = z;
		}
		//nv::vec3f* normals  = (nv::vec3f*)_model->getNormals();
		
		for(int i = 0; i < _model->getPositionCount(); ++i) {
			float& x = vertices[i].x;
			float& y = vertices[i].y;
			float& z = vertices[i].z;
			
			y = y + Rg+(Rt-Rg)/100.0 -min_old.y;
			if(max_old.y < y) max_old.y = y;
			if(min_old.y > y) min_old.y = y;

			//float v_len = sqrtf(x*x+(y)*(y)+z*z);
			//float vx = x/v_len;
			//float vy = (y)/v_len;
			//float vz = z/v_len;

			//vx/vz
			//float u = (atanf(vx/vz)+M_PI)/(2*M_PI);
			//float v = (asinf(vy)+M_PI/2.0)/M_PI;
			
			float r = y;
			//stereographic projection
			//http://kartoweb.itc.nl/geometrics/Map%20projections/body.htm
			float u = -2*atanf(sqrtf(x*x+z*z)/(2*r))+M_PI/2.0;
			float v = atanf(x/(-z));
			//float r = sqrt(x*x+y*y+z*z);
			
			y = r*cos(u) * cos(v);
    		x = r*cos(u) * sin(v);
    		y = r*sin(u);
			
			if(max.x < x) max.x = x;
			if(max.y < y) max.y = y;
			if(max.z < z) max.z = z;

			if(min.x > x) min.x = x;
			if(min.y > y) min.y = y;
			if(min.z > z) min.z = z;
			
		}

		float3 center = (max + min)/2.0; //make_float3(0.0,Rg,0.0);//(max + min)/2;
		float center_len = length(center);
		float center_s = sqrtf(center.x*center.x+center.y*center.y);
		//float lon = atanf(center.y / center.x);
		//float lat = atanf(center_s / center.z);
		// remember 90 - lat!!!!
		//float lat = acosf(center.y/length(center));
		//float lon = (center.x <= 0) ? asinf(center_s/center.z) : M_PI-asinf(center_s/center.z);
		float lat = (center_s!= 0) ? atan2f(center.z,center_s) : ((center.z>=0) ? M_PI/2.0 : -M_PI/2.0);
		float lon = (center.x != 0) ? atan2f(center.y,center.x) : ((center.y>=0)? M_PI/2.0 : -M_PI/2.0);

		float co = cosf(lon);
		float so = sinf(lon);
		float ca = cosf(lat);
		float sa = sinf(lat);
		float3 po = make_float3(co*ca, so*ca, sa) * Rg;
		//float3 po = make_float3(co*sa, so*sa, ca) * Rg;
		cout << "center: " << center.x << ", " << center.y << ", " << center.z << endl;
		cout << "data file cente lat: " << lat << ", lon: " <<  lon << endl;
		cout << "p0: "  << po.x << ", " << po.y << ", " << po.z << endl;

		_model->removeDegeneratePrims();
		_model->computeNormals();

		_model->clearTexCoords();
		_model->clearColors();
		_model->clearTangents();

	/*
		if(zUp) {
			nv::vec3f* vertices = (nv::vec3f*)_model->getPositions();
			nv::vec3f* normals  = (nv::vec3f*)_model->getNormals();
			for(int i = 0; i < _model->getPositionCount(); ++i) {
			  std::swap(vertices[i].y, vertices[i].z);
			  vertices[i].z *= -1;

			  std::swap(normals[i].y, normals[i].z);
			  normals[i].z *= -1;
			}
		}
	*/
		_model->compileModel();

		char fn[MAX_PATH];
		sprintf(fn,"C:\\SVN\\objfile.xyz");
		FILE* fp = fopen(fn,"w");
		if(fp){
			nv::vec3f* vertices = (nv::vec3f*)_model->getPositions();
			//nv::vec3f* normals  = (nv::vec3f*)_model->getNormals();
			for(int i = 0; i < _model->getPositionCount(); ++i) {
				fprintf(fp,"%f, %f, %f\n",vertices[i].x,vertices[i].y,vertices[i].z);
			}

			fclose(fp);
		}
		
		glGenBuffers(1, &_modelVB);
		glBindBuffer(GL_ARRAY_BUFFER, _modelVB);
		glBufferData(GL_ARRAY_BUFFER,
				   _model->getCompiledVertexCount()*_model->getCompiledVertexSize()*sizeof(float),
				   _model->getCompiledVertices(), GL_STATIC_READ);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glGenBuffers(1, &_modelIB);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _modelIB);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
				   _model->getCompiledIndexCount()*sizeof(int),
				   _model->getCompiledIndices(), GL_STATIC_READ);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		float diag;
		/*
		_model->computeBoundingBox(modelBBMin, modelBBMax);
		std::cerr << "modelBBMin[" << modelBBMin.x << ", " << modelBBMin.y << ", " << modelBBMin.z << "]" << endl;
		std::cerr << "modelBBMax[" << modelBBMax.x << ", " << modelBBMax.y << ", " << modelBBMax.z << "]" << endl;

		modelBBCenter = (modelBBMin + modelBBMax) * 0.5;
		std::cerr << "modelBBCenter[" << modelBBCenter.x << ", " << modelBBCenter.y << ", " << modelBBCenter.z << "]" << endl;
		std::cerr << "modelBBMax-modelBBMin[" << modelBBMax.x-modelBBMin.x << ", " << modelBBMax.y-modelBBMin.y << ", " << modelBBMax.z-modelBBMin.z << "]" << endl;
		diag = nv::length(modelBBMax - modelBBMin);
		*/

		//m_context->setRayTypeCount(1);
		//m_context->setEntryPointCount(1);

		//m_context["shadow_ray_type"]->setUint(0u);
		//m_context["scene_epsilon"]->setFloat(scene_epsilon);


		optix::Material opaque = m_context->createMaterial();
		opaque->setClosestHitProgram( rtpass, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
			"rtpass_mesh_closest_hit") );
		opaque->setAnyHitProgram(shadow, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
			"shadow_any_hit") );

		optix::Geometry rtModel = m_context->createGeometry();
		rtModel->setPrimitiveCount( _model->getCompiledIndexCount()/3 );
		rtModel->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath("isgShadows", "triangle_mesh_fat.cu"), "mesh_intersect" ) );
		rtModel->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath("isgShadows", "triangle_mesh_fat.cu"), "mesh_bounds" ) );

		int num_vertices = _model->getCompiledVertexCount();
		optix::Buffer vertex_buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, _modelVB);
		vertex_buffer->setFormat(RT_FORMAT_USER);
		vertex_buffer->setElementSize(3*2*sizeof(float));
		vertex_buffer->setSize(num_vertices);
		rtModel["vertex_buffer"]->setBuffer(vertex_buffer);


		optix::Buffer index_buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, _modelIB);
		index_buffer->setFormat(RT_FORMAT_INT3);
		index_buffer->setSize(_model->getCompiledIndexCount()/3);
		rtModel["index_buffer"]->setBuffer(index_buffer);

		optix::Buffer material_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
		material_buffer->setFormat(RT_FORMAT_UNSIGNED_INT);
		material_buffer->setSize(_model->getCompiledIndexCount()/3);
		void* material_data = material_buffer->map();
		memset(material_data, 0, _model->getCompiledIndexCount()/3*sizeof(unsigned int));
		material_buffer->unmap();
		rtModel["material_buffer"]->setBuffer(material_buffer);

	/*	gi->setMaterialCount(1);
		gi->setMaterial(0, opaque);
		gi->setGeometry(rtModel);
	*/

		_gi->setMaterialCount(1);
		_gi->setMaterial(0, opaque);
		_gi->setGeometry(rtModel);


		//m_context["shadow_casters"]->set(geometrygroup);

		//m_context->setStackSize(2048);
		//m_context->validate();
	} catch ( optix::Exception& e ) {
		sutilReportError( e.getErrorString().c_str() );
		exit(-1);
	}
	return true;
	//return gi;
}

//TODO: modify your algorithm: you have to create Optix sampler right after populating OpenGL textures
void ProgressivePhotonScene::loadTexToOptix()
{

	//create1DOptixTex("SRPhaseFuncSampler", getTex(G_SRUnit) , texSRPhaseFuncSampler);
	//create1DOptixTex("SMPhaseFuncSampler", getTex(G_SMUnit) , texSMPhaseFuncSampler);
	//create1DOptixTex("SRPhaseFuncIndSampler", getTex(G_SRDensUnit) , texSRPhaseFuncIndSampler);
	//create1DOptixTex("SMPhaseFuncIndSampler", getTex(G_SMDensUnit) , texSMPhaseFuncIndSampler);
	
	
	//create2DOptixTex("trans_texture", getTex(G_transmittanceUnit) , textrans_texture);
	//create2DOptixTex("earth_texture", getTex(G_reflectanceUnit) , texearth_texture);
	//create3DOptixTex("inscatterSampler", getTex(G_inscatterUnit) , texinscatterSampler);
	//create3DOptixTex("raySampler", getTex(G_deltaSRUnit) , texraySampler);
	//create3DOptixTex("mieSampler", getTex(G_deltaSMUnit) , texmieSampler);
	


	//m_context["SRPhaseFuncSampler"]->setTextureSampler(loadTexture2f(disp->get_sr_phase_func(),disp->getDimX(),1));
	//m_context["SMPhaseFuncSampler"]->setTextureSampler(loadTexture2f(disp->get_sm_phase_func(),disp->getDimX(),1));
	//m_context["SRPhaseFuncIndSampler"]->setTextureSampler(loadTexture2f(disp->get_sr_phase_func_ind(),disp->getDimX(),1));
	//m_context["SMPhaseFuncIndSampler"]->setTextureSampler(loadTexture2f(disp->get_sm_phase_func_ind(),disp->getDimX(),1));
/*
	m_context["trans_texture"]->setTextureSampler(loadTexture2f(disp->getTransmittance(), disp->getTransmittanceWidth(), 
																disp->getTransmittanceHeight()));
*/
	m_context->validate();
	m_context->compile();
}

GeometryInstance ProgressivePhotonScene::createParallelogram( const float3& anchor,
															 const float3& offset1,
															 const float3& offset2,
															 const float3& color )
{
	Geometry parallelogram = m_context->createGeometry();
	parallelogram->setPrimitiveCount( 1u );
	std::string ptx_path = ptxpath( "isgShadows", "parallelogram.cu" );
	parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( ptx_path, "intersect" ));
	parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptx_path, "bounds" ));

	float3 normal = normalize( cross( offset1, offset2 ) );
	float d       = dot( normal, anchor );
	float4 plane  = make_float4( normal, d );

	float3 v1 = offset1 / dot( offset1, offset1 );
	float3 v2 = offset2 / dot( offset2, offset2 );

	parallelogram["plane"]->setFloat( plane );
	parallelogram["anchor"]->setFloat( anchor );
	parallelogram["v1"]->setFloat( v1 );
	parallelogram["v2"]->setFloat( v2 );

	Material material = m_context->createMaterial();
	material->setClosestHitProgram( rtpass, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
		"rtpass_closest_hit") );
	//sphere_matl->setClosestHitProgram( ppass,  m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_ppass.cu"),
	//	"ppass_closest_hit") );
	material->setAnyHitProgram(     gather,  m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_gather.cu"),
	                                 "gather_any_hit") );
	material->setAnyHitProgram(shadow, m_context->createProgramFromPTXFile( ptxpath( "isgShadows", "ppm_rtpass.cu"),
		"shadow_any_hit") );

	material["Ka"]->setFloat( make_float3( 0.05f, 0.05f, 0.05f ) );
	material[ "emissive" ]->setFloat( 0.0f, 0.0f, 0.0f );
	material["phong_exp"]->setFloat(0.0f);
	material["refraction_index"]->setFloat( 1.2f );
	material["ambient_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.2f, 0.2f, 0.2f ) ) );
	material["diffuse_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.8f, 0.8f, 0.8f ) ) );
	material["specular_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.0f, 0.0f, 0.0f ) ) );
	material["shadow_attenuation"]->setFloat( 0.4f, 0.7f, 0.4f );
	float color_scale = 1;
	float3 Kd = make_float3( color_scale, 1.0f - color_scale, 1.0f );
	material["transmissive_map"]->setTextureSampler( loadTexture( m_context, "", Kd ) );

	GeometryInstance gi = m_context->createGeometryInstance();
	gi->setGeometry( parallelogram );
	gi->setMaterialCount( 1 );
	gi->setMaterial( 0, material );

	gi["Kd"]->setFloat( color );
	gi["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
	gi["use_grid"]->setUint( 0u );
	gi["grid_color"]->setFloat( make_float3( 0.0f ) );
	gi["emitted"]->setFloat( 0.0f, 0.0f, 0.0f );
	return gi;
}


void ProgressivePhotonScene::loadObjGeometry( const std::string& filename, optix::Aabb& bbox )
{
	GeometryGroup geometry_group = m_context->createGeometryGroup();
	std::string full_path = std::string( sutilSamplesDir() ) + "/progressivePhotonMap/wedding-band.obj";
	PpmObjLoader loader( full_path, m_context, geometry_group );
	loader.load();
	bbox = loader.getSceneBBox();

	m_context["top_object"]->set( geometry_group );
	m_context["top_shadower"]->set( geometry_group );
}


void ProgressivePhotonScene::createTestGeometryEarthAtm(optix::Aabb& bbox)
{
	const float3 white = make_float3( 1.8f, 0.8f, 0.8f );
	const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
	const float3 blue = make_float3(0.05f, 0.05f, 0.8f );
	const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
	const float3 black = make_float3( 0.0f, 0.0f, 0.0f );
	const float3 brown = make_float3(165.0/255.0,42.0/255.0,42.0/255.0);

	//const float3 light = make_float3( 15.0f, 15.0f, 5.0f );

	// create geometry instances
	std::vector<GeometryInstance> gis;
	std::vector<GeometryInstance> gis_ground;


	m_context["Rt"]->setFloat(Rt);
	m_context["Rg"]->setFloat(Rg);
	m_context["RL"]->setFloat(RL);
	m_context["TRANSMITTANCE_INTEGRAL_SAMPLES"]->setInt(TRANSMITTANCE_INTEGRAL_SAMPLES);
	m_context["INSCATTER_INTEGRAL_SAMPLES"]->setInt(INSCATTER_INTEGRAL_SAMPLES);
	m_context["betaR"]->setFloat( betaR);
	m_context["betaMEx"]->setFloat( betaMEx );
	m_context["betaMSca"]->setFloat( betaMSca );

	m_context["RES_R"]->setInt(RES_R);
	m_context["RES_MU"]->setInt(RES_MU);
	m_context["RES_MU_S"]->setInt(RES_MU_S);
	m_context["RES_MU_S_BIN"]->setInt(RES_MU_S_BIN);
	m_context["RES_R_BIN"]->setInt(RES_R_BIN);
	m_context["RES_NU"]->setInt(RES_NU);
	m_context["RES_DIR_BIN_TOTAL"]->setInt(RES_DIR_BIN_TOTAL);
	m_context["RES_DIR_BIN"]->setInt(RES_DIR_BIN);
	m_context["mieG"]->setFloat(mieG);

	m_context["HR"]->setFloat( HR);
	m_context["HM"]->setFloat( HM );
/*
	m_context["SRPhaseFuncSampler"]->setTextureSampler(loadTexture2f(disp->get_sr_phase_func(),disp->getDimX(),1));
	m_context["SMPhaseFuncSampler"]->setTextureSampler(loadTexture2f(disp->get_sm_phase_func(),disp->getDimX(),1));
	m_context["SRPhaseFuncIndSampler"]->setTextureSampler(loadTexture2f(disp->get_sr_phase_func_ind(),disp->getDimX(),1));
	m_context["SMPhaseFuncIndSampler"]->setTextureSampler(loadTexture2f(disp->get_sm_phase_func_ind(),disp->getDimX(),1));
	m_context["trans_texture"]->setTextureSampler(loadTexture2f(disp->getTransmittance(), disp->getTransmittanceWidth(), disp->getTransmittanceHeight()));
	
	m_context["inscatterSampler"]->setTextureSampler(loadTexture3f(disp->getInscatter1(), disp->getInscatter1Width(), 
	disp->getInscatter1Height(),disp->getInscatter1Depth()));
	m_context["raySampler"]->setTextureSampler(loadTexture3f(disp->getDataRay(), disp->getRayWidth(), 
	disp->getRayHeight(),disp->getRayDepth()));
	m_context["mieSampler"]->setTextureSampler(loadTexture3f(disp->getDataMie(), disp->getMieWidth(), 
	disp->getMieHeight(),disp->getMieDepth()));
	*/
	/*
	m_context["earth_texture"]->setTextureSampler(loadTexture2uc(pr.getEarthTex(), EARTH_TEX_WIDTH, EARTH_TEX_HEIGHT));
	m_context["trans_texture"]->setTextureSampler(loadTexture2f(pr.tr.getTransmittance(), pr.tr.getWidth(), pr.tr.getHeight()));
	m_context["inscatterSampler"]->setTextureSampler(loadTexture3f(pr.inscatter1.getInscatter(), pr.inscatter1.getWidth(), 
	pr.inscatter1.getHeight(),pr.inscatter1.getDepth()));
	m_context["raySampler"]->setTextureSampler(loadTexture3f(pr.inscatter1.getDataRay(), pr.inscatter1.getWidth(), 
	pr.inscatter1.getHeight(),pr.inscatter1.getDepth()));
	m_context["mieSampler"]->setTextureSampler(loadTexture3f(pr.inscatter1.getDataMie(), pr.inscatter1.getWidth(), 
	pr.inscatter1.getHeight(),pr.inscatter1.getDepth()));\
	*/

	// Sphere geometry.
	//gis.push_back( createCone(blue , Rg, Rg));
	//gis_ground.push_back( createSphere(brown , Rg));
	gis.push_back( createVolSphere(white, Rt, Rg));
	//gis.push_back( createSphere(green , (Rg+(Rt-Rg)*4/(RES_R/4)), true));


	//////////////////////////////////////////////////////////
	// X-Y plane and Y-Z plane
/*
	gis.push_back( createParallelogram( make_float3(0.0,0.0,0.0),
                                      make_float3(0.0,Rt,0.0),
                                      make_float3(Rt,0.0,0.0),
                                      white ));
	gis.push_back( createParallelogram( make_float3(0.0,0.0,0.0),
                                      make_float3(0.0,Rt,0.0),
                                      make_float3(0.0,0.0,Rt),
                                      red ));
*/	
	//bbox.include(make_float3( 0.0f, 0.0f, 0.0f ));
	//bbox.include(make_float3( 556.0f, 548.8f, 559.2f ));	
	bbox.include(make_float3( -Rt-100, -Rt-100, -Rt-100 ));
	bbox.include(make_float3( Rt+100, Rt+100, Rt+100 ));

	// Create geometry group
	
	GeometryGroup geometrygroup = m_context->createGeometryGroup();
	geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
	for(size_t i = 0; i < gis.size(); ++i) {
		geometrygroup->setChild( (int)i, gis[i] );
	}
	geometrygroup->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );
	
	GeometryGroup geometrygroup_ground = m_context->createGeometryGroup();
	geometrygroup_ground->setChildCount( static_cast<unsigned int>(gis_ground.size()) );
	for(size_t i = 0; i < gis_ground.size(); ++i) {
		geometrygroup_ground->setChild( (int)i, gis_ground[i] );
	}
	geometrygroup_ground->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	_gi =m_context->createGeometryInstance();
	optix::GeometryGroup geometrygroup_obj = m_context->createGeometryGroup();
	
	if(loadObjFile_materials(_objfilename, geometrygroup_obj)){
	}
	
	//if(loadObjFile( _objfilename)){
	//	geometrygroup_obj->setChildCount(1);
	//	geometrygroup_obj->setChild(0, _gi);
	//}
	else
		geometrygroup_obj->setChildCount(0);
    geometrygroup_obj->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	std::vector<GeometryInstance> gis_cones;
	std::vector<GeometryInstance> gis_cones_inv;
	std::vector<GeometryInstance> gis_spheres;
	
	//for(int i = 1; i < RES_R; i++){
	//	gis_spheres.push_back( createSphere(green , (Rg+(Rt-Rg)*(float)i/(float)(RES_R-1)), true));
	//}
	//gis_spheres.push_back( createSphere(green , Rg, true));
	//gis_spheres.push_back( createSphere(green , (Rg+(Rt-Rg)*(float)12/(float)(RES_R_BIN)), true));
	//gis_spheres.push_back( createSphere(green , (Rg+(Rt-Rg)*(float)16/(float)(RES_R_BIN)), true));
	//gis_spheres.push_back( createSphere(green , (Rg+(Rt-Rg)*(float)24/(float)(RES_R_BIN)), true));
	gis_spheres.push_back( createSphere(green , Rt, true));

	GeometryGroup geometrygroup_spheres = m_context->createGeometryGroup();
	geometrygroup_spheres->setChildCount( static_cast<unsigned int>(gis_spheres.size()) );
	for(size_t i = 0; i < gis_spheres.size(); ++i) {
		geometrygroup_spheres->setChild( (int)i, gis_spheres[i] );
	}
	geometrygroup_spheres->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	

	GeometryGroup geometrygroup_cones = m_context->createGeometryGroup();
	geometrygroup_cones->setChildCount( static_cast<unsigned int>(gis_cones.size()) );
	for(size_t i = 0; i < gis_cones.size(); ++i) {
		geometrygroup_cones->setChild( (int)i, gis_cones[i] );
	}
	geometrygroup_cones->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	GeometryGroup geometrygroup_cones_inv = m_context->createGeometryGroup();
	geometrygroup_cones_inv->setChildCount( static_cast<unsigned int>(gis_cones_inv.size()) );
	for(size_t i = 0; i < gis_cones_inv.size(); ++i) {
		geometrygroup_cones_inv->setChild( (int)i, gis_cones_inv[i] );
	}
	geometrygroup_cones_inv->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	
	_transform_inner_obj->setChild(geometrygroup_cones );
	_transform_inner_obj_inv->setChild(geometrygroup_cones_inv );

	optix::Matrix<4,4> Translate = optix::Matrix<4,4>::translate( make_float3(0.0,0.0,-sinf(M_PI/4)*Rg));
	float3 Axis = normalize( make_float3(1.0,0.0,0.0) );
    optix::Matrix4x4 Rotate = optix::Matrix4x4::rotate( M_PI, Axis );
    //optix::Matrix<4,4> Comp = Rotate*Translate;
	optix::Matrix<4,4> Comp =  Rotate*Translate;
	//_transform_inner_obj->setMatrix( false, Comp.getData( ), 0 );
/*
	geometrygroup->setChildCount(ind+1);
	geometrygroup->setChild( ind,_gi);
	geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );
*/

	//add geometry from obj file


	//Transform transform_inner_obj = m_context->createTransform( );
	//transform_inner_obj->setChild(geometrygroup_obj );

	_ggroup = m_context->createGroup();
	_ggroup->setChildCount(4);
	//_ggroup->setChild( 0, geometrygroup_obj);
	_ggroup->setChild( 0, _transform_inner_obj);
	_ggroup->setChild( 1, _transform_inner_obj_inv);
	_ggroup->setChild( 2, geometrygroup_spheres);
	//ggroup->setChild( 3, geometrygroup);
	_ggroup->setChild( 3, geometrygroup_ground);
	_ggroup->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );
	m_context["shadow_casters"]->set(_ggroup);
	m_context["top_object"]->set( _ggroup);

	_ggroup_single = m_context->createGroup();
	_ggroup_single->setChildCount(3);
	_ggroup_single->setChild( 0, geometrygroup_obj);
	_ggroup_single->setChild( 1, geometrygroup);
	_ggroup_single->setChild( 2, geometrygroup_ground);
	_ggroup_single->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );
	m_context["bruneton_single"]->setInt((int)_bruneton_single);
	m_context["jm_mult"]->setInt((int)_jm_mult);
			
	m_context["shadow_casters"]->set(_ggroup_single);
	m_context["single_top_object"]->set( _ggroup_single);

	_ppm_ggroup = m_context->createGroup();
	_ppm_ggroup->setChildCount(2);
	//_ppm_ggroup->setChild( 0, geometrygroup_obj);
	//ppm_ggroup->setChild( 1, _transform_inner_obj);
	//ppm_ggroup->setChild( 2, _transform_inner_obj_inv);
	_ppm_ggroup->setChild( 0, geometrygroup);
	_ppm_ggroup->setChild( 1, geometrygroup_ground);
	_ppm_ggroup->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	m_context["geometry_choice"]->setInt(1.0);
	m_context["ppm_top_object"]->set( _ppm_ggroup);

	//ggroup->setChild( 0, geometrygroup);
	//ggroup->setChild( 0, transform_inner_obj);
	//ggroup->setChild( 1, geometrygroup);
	//ggroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

	//optix::Matrix<4,4> Translate = optix::Matrix<4,4>::translate( make_float3(0.0,Rg,0.0));
	//optix::Matrix<4,4> Comp = Translate;// * Rotate;
	//optix::Matrix<4,4> Scale = optix::Matrix<4,4>::scale(make_float3(0.25,0.25,0.25));
	//optix::Matrix<4,4> Comp = Translate;// * Rotate;

	//transform_inner_obj->setMatrix( false, Comp.getData( ), 0 );
	//gis.push_back(gi);


	m_context["sigma_s"]->setFloat( betaR);
	m_context["sigma_a"]->setFloat( make_float3(0.0f,0.0f,0.0f) );

	m_context["filter_type"]->setInt( FILTER_GAUSSIAN );
	m_context["filter_width"]->setFloat( 3.0f );//1.0
	m_context["gaussian_alpha"]->setFloat( 0.1f );

	m_context["gR_ind"]->setInt(0);
	m_context->setStackSize(2048);
	//m_context->setStackSize(20048);
	//m_context->setStackSize(10048);
    m_context->validate();
	//gis[gis.size() - 1]["phong_exp"]->setFloat(0.0f);

/*
	Buffer output_samples = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	output_samples->setFormat( RT_FORMAT_USER );
	output_samples->setElementSize( sizeof( float3 ) );
	output_samples->setSize( TRANSMITTANCE_W, TRANSMITTANCE_H );
	m_context["transmittance_samples"]->set( output_samples);
*/
}

void ProgressivePhotonScene::switchAtmosphereMode()
{
	_bruneton_single = !_bruneton_single;
	m_context["bruneton_single"]->setInt((int)_bruneton_single);
}

void ProgressivePhotonScene::switchAtmosphereMode1()
{
	_jm_mult = !_jm_mult;
	m_context["jm_mult"]->setInt((int)_jm_mult);
}

bool ProgressivePhotonScene::isSingle()
{
	return _bruneton_single;
}

void ProgressivePhotonScene::switchSunControlState() 
{
	_bSunActivate = !_bSunActivate;
	string status = _bSunActivate ? "enabled" : "disabled";
	cout << "_bSunActivate " << status << endl;
}

bool ProgressivePhotonScene::isSunControlActivated() 
{	
	return _bSunActivate;
}

//-----------------------------------------------------------------------------
//
// Main driver
//
//-----------------------------------------------------------------------------

#ifdef PHOTON_TEST

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
	std::cerr
		<< "Usage  : " << argv0 << " [options]\n"
		<< "App options:\n"
		<< "  -h  | --help                               Print this usage message\n"
		<< "  -c  | --cornell-box                        Display Cornell Box scene\n"
		<< "  -t  | --timeout <sec>                      Seconds before stopping rendering. Set to 0 for no stopping.\n"
#ifndef RELEASE
		<< "  -pt | --print-timings                      Print timing information\n"
		<< " -ddb | --display-debug-buffer               Display the debug buffer information\n"
#endif
		<< std::endl;
	GLUTDisplay::printUsage();

	std::cerr
		<< "App keystrokes:\n"
		<< "  w Move light up\n"
		<< "  a Move light left\n"
		<< "  s Move light down\n"
		<< "  d Move light right\n"
		<< std::endl;

	if ( doExit ) exit(1);
}



int main( int argc, char** argv )
{

	GLUTDisplay::init( argc, argv );

	bool print_timings = false;
	bool display_debug_buffer = false;
	bool cornell_box = false;
	bool distant_test = false;
	float timeout = -1.0f;

	for ( int i = 1; i < argc; ++i ) {
		std::string arg( argv[i] );
		if ( arg == "--help" || arg == "-h" ) {
			printUsageAndExit( argv[0] );
		} else if ( arg == "--print-timings" || arg == "-pt" ) {
			print_timings = true;
		} else if ( arg == "--display-debug-buffer" || arg == "-ddb" ) {
			display_debug_buffer = true;
		} else if ( arg == "--cornell-box" || arg == "-c" ) {
			cornell_box = true;
		} else if ( arg == "--distant" || arg == "-d" ) {
			distant_test = true;
		} else if ( arg == "--timeout" || arg == "-t" ) {
			if(++i < argc) {
				timeout = static_cast<float>(atof(argv[i]));
			} else {
				std::cerr << "Missing argument to "<<arg<<"\n";
				printUsageAndExit(argv[0]);
			}
		} else {
			std::cerr << "Unknown option: '" << arg << "'\n";
			printUsageAndExit( argv[0] );
		}
	}

	if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

	try {
		//Precomp pr;
		//pr.precompute();

		ProgressivePhotonScene scene;
		if (print_timings) scene.printTimings();
		if (display_debug_buffer) scene.displayDebugBuffer();
		if (cornell_box ) scene.setSceneCornellBox();
		if (distant_test ) scene.setDistantTest();
		timeout = 30;
		GLUTDisplay::setProgressiveDrawingTimeout(timeout);
		GLUTDisplay::setUseSRGB(true);
		GLUTDisplay::run( "ProgressivePhotonScene", &scene, GLUTDisplay::CDNone );//GLUTDisplay::CDProgressive );
	} catch( Exception& e ){
		sutilReportError( e.getErrorString().c_str() );
		exit(1);
	}

	return 0;
}

#endif