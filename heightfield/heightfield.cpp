
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

//-----------------------------------------------------------------------------
//
//  heightfield.cpp: render a simple heightfield
//
//  Options        : [ sinxy | sync | plane | plane2 | file_name ]
//
//-----------------------------------------------------------------------------


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <fstream>
#include <iostream>
#include <float.h>
#include <GLUTDisplay.h>
#include "commonStructs.h"
#include <cstdlib>
#include <cstring>
#include "perlin.h"

using namespace optix;

//-----------------------------------------------------------------------------
//
// Heightfield data creation routines
//
//-----------------------------------------------------------------------------

namespace {
  inline float plane(float x, float y)
  {
    return 0.5f;
  }

  inline float plane2(float x, float y)
  {
    return y;
  }

  inline float sinxy(float x, float y)
  {
    float r2 = 8.0f*(x*x+y*y);
    if(r2 > -1.e-12f && r2 < 1.e-12f)
      return 1.0f;
    else
      return (1.2f*sinf(x*15.0f)*sinf(y*30.0f)/(r2*sqrtf(r2)+0.47f));
  }

  inline float sinc(float x, float y)
  {
    float r = sqrtf(x*x+y*y)*20;
    return (r==0.0f?1.0f:sinf(r)/r)+1.0f;
  }

  template<typename F>
    static void fillBuffer(F& f, Context& context, Buffer& databuffer, int nx, int ny)
    {
      databuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT, nx+1, ny+1 );
      float* p = reinterpret_cast<float*>(databuffer->map());
      for(int i = 0; i<= nx; i++){
        float x = float(i)/nx * 2 - 1;
        for(int j = 0; j<= ny; j++){
          float y = float(j)/ny * 2 - 1;
          *p++ = f(x, y);
        }
      }
      databuffer->unmap();
    }

    static void fillBufferPerlin(Context& context, Buffer& databuffer, int nx, int ny)
    {
	  //PerlinNoise pn(0.5, 4, 128,7, 5);
	  
	  PerlinNoise pn(0.5, 1.2, 128,7, 5);
	  double res =0.0;
	  double min_res = pn.GetHeight(0.0, 0.0);
	  double max_res = pn.GetHeight(0.0, 0.0);
	  std::vector<double> pn_arr;
	  for (int i = 0; i <= nx; i++){
	    float x = float(i)/nx * 8 - 1;
		for (int j = 0; j <= ny; j++){
			float y = float(j)/ny * 8 - 1;
			res = pn.GetHeight(x, y);
			pn_arr.push_back(res);
			if(res < min_res)
				min_res = res;
			if(res > max_res)
				max_res = res;
		}
	  }
      databuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT, nx+1, ny+1 );
      float* p = reinterpret_cast<float*>(databuffer->map());
      for(int i = 0; i<= nx*ny; i++){
		*p++ = (pn_arr[i]-min_res)/(max_res-min_res);
      }
      databuffer->unmap();
    }

	static void fillBufferPerlinGPU(Context& context, Buffer& databuffer, int nx, int ny)
	{
		PerlinNoiseGPU pngp;
		/*
		char *pixels;
		int i,j;

		glGenTextures(1, texID); // Generate a unique texture ID
		glBindTexture(GL_TEXTURE_2D, *texID); // Bind the texture to texture unit 0

		pixels = (char*)malloc( 256*256*4 );
		for(i = 0; i<256; i++)
		for(j = 0; j<256; j++) {
		  int offset = (i*256+j)*4;
		  char value = perm[(j+perm[i]) & 0xFF];
		  pixels[offset] = grad3[value & 0x0F][0] * 64 + 64;   // Gradient x
		  pixels[offset+1] = grad3[value & 0x0F][1] * 64 + 64; // Gradient y
		  pixels[offset+2] = grad3[value & 0x0F][2] * 64 + 64; // Gradient z
		  pixels[offset+3] = value;                     // Permuted index
		}


		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
		*/
	}
}

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

static float rand_range(float min, float max)
{
  return min + (max - min) * (float) rand() / (float) RAND_MAX;
}


//-----------------------------------------------------------------------------
// 
// Heightfield Scene
//
//-----------------------------------------------------------------------------

class HeightfieldScene : public SampleScene
{
public:
  HeightfieldScene(const std::string& dataname) : dataname(dataname) {}

  // From SampleScene
  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer();

  void createGeometry();
  void createData();
  GeometryInstance getGeo(float3 min, float3 max, bool bTest);
  void getParallelogramVector(std::vector<GeometryInstance>& gis);
  void setupNoiseTexture();

private:
  Buffer      databuffer;
  float       ymin, ymax;
  std::string dataname;

  static unsigned int WIDTH;
  static unsigned int HEIGHT;
};

unsigned int HeightfieldScene::WIDTH  = 512u;
unsigned int HeightfieldScene::HEIGHT = 384u;


void HeightfieldScene::initScene( InitialCameraData& camera_data )
{
  try {
    // Setup state
    m_context->setRayTypeCount( 2 );
    m_context->setEntryPointCount( 1 );
    m_context->setStackSize(560);

    m_context["radiance_ray_type"]->setUint( 0u );
    m_context["scene_epsilon"]->setFloat( 1.e-3f );
    m_context["max_depth"]->setInt(5);
    m_context["radiance_ray_type"]->setUint(0);
    m_context["shadow_ray_type"]->setUint(1);
    m_context["output_buffer"]->set( createOutputBuffer( RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT ) );

    // Set up camera
    camera_data = InitialCameraData( make_float3( 4.0f, 4.0f, 4.0f ), // eye
                                     make_float3( 0.0f, 0.0f, 0.3f ), // lookat
                                     make_float3( 0.0f, 1.0f, 0.0f ), // up
                                     45.0f );                         // vfov

    // Declare camera variables.  The values do not matter, they will be overwritten in trace.
    m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

    m_context["bad_color"]->setFloat( 1.0f, 1.0f, 0.0f );
    m_context["bg_color"]->setFloat( make_float3(.1f, 0.2f, 0.4f) * 0.5f );

    // Ray gen program
    std::string ptx_path = ptxpath( "heightfield", "pinhole_camera.cu" );
    Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
    m_context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
    m_context->setExceptionProgram( 0, exception_program );
    m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptxpath( "heightfield", "constantbg.cu" ), "miss" ) );

    // Set up light buffer
    m_context["ambient_light_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    BasicLight lights[] = { 
      { { 4.0f, 12.0f, 10.0f }, { 1.0f, 1.0f, 1.0f }, 1 }
    };

    Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(BasicLight));
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    m_context["lights"]->set(light_buffer);

    createData();
    createGeometry();

    // Finalize
    m_context->validate();
    m_context->compile();

  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
}


Buffer HeightfieldScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void HeightfieldScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( 0, 
                    static_cast<unsigned int>(buffer_width),
                    static_cast<unsigned int>(buffer_height)
                    );
}

void HeightfieldScene::setupNoiseTexture()
{
	// 3D solid noise buffer, 1 float channel, all entries in the range [0.0, 1.0].
  srand(0); // Make sure the pseudo random numbers are the same every run.

  int tex_width  = 164;
  int tex_height = 164;
  int tex_depth  = 64;
  Buffer noiseBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, tex_width, tex_height);//, tex_depth);
  float *tex_data = (float *) noiseBuffer->map();
  bool bPerlin = true;

  PerlinNoise pn(0.5, 4, 128,7, 5);
  if (bPerlin)
  {
    // Distances to Voronoi control points (repeated version, taking the 26 surrounding cubes into account!)
    // Voronoi_repeat(16, tex_width, tex_height, tex_depth, tex_data);


    std::vector<double> pn_arr;
	double res =0.0;
	double min_res = pn.GetHeight(0.0, 0.0);
	double max_res = pn.GetHeight(0.0, 0.0);
	for (int x = tex_width; x > 0; x--){
		for (int y = tex_height; y > 0; y--){
			// One channel 3D noise in [0.0, 1.0] range.
			res = pn.GetHeight(x, y);
			pn_arr.push_back(res);
			if(res < min_res)
				min_res = res;
			if(res > max_res)
				max_res = res;
		}
    }

	for (int i = 0; i < pn_arr.size(); i++){
			*tex_data++ = (pn_arr[i]-min_res)/(max_res-min_res);
			//std::cout << res;
    }
  }
  else
  {
    // Random noise in range [0, 1]
    for (int i = tex_width * tex_height/* * tex_depth*/;  i > 0; i--)
    {
      // One channel 3D noise in [0.0, 1.0] range.
      *tex_data++ = rand_range(0.0f, 1.0f);
    }
  }
  noiseBuffer->unmap(); 


  // Noise texture sampler
  TextureSampler noiseSampler = m_context->createTextureSampler();

  noiseSampler->setWrapMode(0, RT_WRAP_REPEAT);
  noiseSampler->setWrapMode(1, RT_WRAP_REPEAT);
  noiseSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
  noiseSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
  noiseSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
  noiseSampler->setMaxAnisotropy(1.0f);
  noiseSampler->setMipLevelCount(1);
  noiseSampler->setArraySize(1);
  noiseSampler->setBuffer(0, 0, noiseBuffer);

  m_context["noise_texture"]->setTextureSampler(noiseSampler);
}

void HeightfieldScene::getParallelogramVector(std::vector<GeometryInstance>& gis)
{
	// old code
	RTsize nx, nz;
	databuffer->getSize(nx, nz);

	float3 min = make_float3(0.0,0.0,0.0);
	float3 max = make_float3(1.0,1.0,1.0);
	// If buffer is nx by nz, we have nx-1 by nz-1 cells;
	float3 cellsize = (max - min) / (make_float3(static_cast<float>(nx-1), 1.0f, static_cast<float>(nz-1)));
	cellsize.y = 1;
	float3 inv_cellsize = make_float3(1)/cellsize;

	// Create material
	//Program phong_ch = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "phong.cu" ), "closest_hit_radiance" );
	//Program phong_ah = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "phong.cu" ), "any_hit_shadow" );
	Program phong_ch = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "phong.cu" ), "noise_closest_hit_radiance" );
	Material parallelogram_matl = m_context->createMaterial();
	parallelogram_matl->setClosestHitProgram( 0, phong_ch );
	//parallelogram_matl->setAnyHitProgram( 1, phong_ah );

	parallelogram_matl["Ka"]->setFloat(6.0f, 0.3f, 0.1f);
	parallelogram_matl["Kd"]->setFloat(0.1f, 0.7f, 0.2f);
	parallelogram_matl["Ks"]->setFloat(0.6f, 0.6f, 0.6f);
	parallelogram_matl["phong_exp"]->setFloat(132);
	parallelogram_matl["reflectivity"]->setFloat(0, 0, 0);

	
	Program pgram_intersect  = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "heightfield.cu" ), "parallelogram_intersect" );
	Program pgram_bounds     = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "heightfield.cu" ), "parallelogram_bounds" );
	float3 anchor;
	float3 v2;
	float3 v1;
	std::vector<float3> anchor_arr;
	std::vector<float3> v1_arr;
	std::vector<float3> v2_arr;
	anchor_arr.push_back(make_float3(0.0f + 1.0f, 0.00f, 0.0f));
	/*
	anchor_arr.push_back(make_float3(0.0f + 1.0f, 0.00f, 0.0f));
	//anchor_arr.push_back(make_float3(0.0f + 1.0f, 0.00f, 0.0f));
	anchor_arr.push_back(make_float3(0.0f + 2.0f, 0.00f+1.0f, 0.0f+1.0f));
	anchor_arr.push_back(make_float3(0.0f + 2.0f, 0.00f+1.0f, 0.0f+1.0f));
	//anchor_arr.push_back(make_float3(0.0f + 2.0f, 0.00f+1.0f, 0.0f+1.0f));
    */
	v1_arr.push_back(make_float3( 0.0f, 1.0f, 0.0f));
	//v1_arr.push_back(make_float3( 1.0f, 0.0f, 0.0f));
	//v1_arr.push_back(make_float3( 1.0f, 0.0f, 0.0f));
    /*
	
    v1_arr.push_back(make_float3( -1.0f, 0.0f, 0.0f));
	v1_arr.push_back(make_float3( 0.0f, -1.0f, 0.0f));
    //v1_arr.push_back(make_float3( -1.0f, 0.0f, -0.0f));
    */
	v2_arr.push_back(make_float3( 0.0f, 0.0f, 1.0f));
	//v2_arr.push_back(make_float3( 0.0f, 1.0f, 0.0f));
	//v2_arr.push_back(make_float3( 0.0f, 0.0f, 1.0f));
    /*
	
    v2_arr.push_back(make_float3( 0.0f, 0.0f, -1.0f));
	v2_arr.push_back(make_float3( 0.0f, 0.0f, -1.0f));
    //v2_arr.push_back(make_float3( 0.0f, -1.0f, 0.0f));
    */
	for( unsigned int i = 0u; i < v1_arr.size() ; i++ ) {
		Geometry parallelogram = m_context->createGeometry();
		parallelogram->setPrimitiveCount( 1u );
		parallelogram->setBoundingBoxProgram( pgram_bounds );
		parallelogram->setIntersectionProgram( pgram_intersect ); 
		
		anchor = anchor_arr[i];
		//float3 anchor = make_float3(-1.0f, 0.00f, 1.0f);
		v1 = v1_arr[i];
		v2 = v2_arr[i];
		
		float3 normal = cross( v1, v2 );
		normal = normalize( normal );
		float d = dot( normal, anchor );
		v1 *= 1.0f/dot( v1, v1 );
		v2 *= 1.0f/dot( v2, v2 );
		float4 plane = make_float4( normal, d );
		parallelogram["plane"]->setFloat( plane );
		parallelogram["v1"]->setFloat( v1 );
		parallelogram["v2"]->setFloat( v2 );
		parallelogram["anchor"]->setFloat( anchor );

		parallelogram["boxmin"]->setFloat(min);
		parallelogram["boxmax"]->setFloat(max);
		parallelogram["cellsize"]->setFloat(cellsize);
		parallelogram["inv_cellsize"]->setFloat(inv_cellsize);
		parallelogram["data"]->setBuffer(databuffer);

		gis.push_back( m_context->createGeometryInstance( parallelogram, &parallelogram_matl, &parallelogram_matl +1 ) );
	}
	
}

GeometryInstance HeightfieldScene::getGeo(float3 min, float3 max, bool bTest)
{
  Geometry heightfield = m_context->createGeometry();
  heightfield->setPrimitiveCount( 1u );

  heightfield->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath( "heightfield", "heightfield.cu" ), "bounds" ) );
  if(bTest)
	 heightfield->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( "heightfield", "heightfield.cu" ), "intersect_test" ) );
  else
     heightfield->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( "heightfield", "heightfield.cu" ), "intersect" ) );
  //float3 min = make_float3(-2, ymin, -2);
  //float3 max = make_float3( 2, ymax,  2);
  RTsize nx, nz;
  databuffer->getSize(nx, nz);
  
  // If buffer is nx by nz, we have nx-1 by nz-1 cells;
  float3 cellsize = (max - min) / (make_float3(static_cast<float>(nx-1), 1.0f, static_cast<float>(nz-1)));
  cellsize.y = 1;
  float3 inv_cellsize = make_float3(1)/cellsize;
  heightfield["boxmin"]->setFloat(min);
  heightfield["boxmax"]->setFloat(max);
  heightfield["cellsize"]->setFloat(cellsize);
  heightfield["inv_cellsize"]->setFloat(inv_cellsize);
  heightfield["data"]->setBuffer(databuffer);

  // Create material
  Program phong_ch = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "phong.cu" ), "closest_hit_radiance" );
  Program phong_ah = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "phong.cu" ), "any_hit_shadow" );
  Material heightfield_matl = m_context->createMaterial();
  heightfield_matl->setClosestHitProgram( 0, phong_ch );
  heightfield_matl->setAnyHitProgram( 1, phong_ah );

  heightfield_matl["Ka"]->setFloat(0.0f, 0.3f, 0.1f);
  heightfield_matl["Kd"]->setFloat(0.1f, 0.7f, 0.2f);
  heightfield_matl["Ks"]->setFloat(0.6f, 0.6f, 0.6f);
  heightfield_matl["phong_exp"]->setFloat(132);
  heightfield_matl["reflectivity"]->setFloat(0, 0, 0);

  GeometryInstance gi = m_context->createGeometryInstance( heightfield, &heightfield_matl, &heightfield_matl+1 );
  return gi;  
}

void HeightfieldScene::createGeometry()
{
  Geometry sphere = m_context->createGeometry();
  sphere->setPrimitiveCount( 1u );
  std::string ptx_path = ptxpath( "heightfield", "heightfield.cu" );
  //Program sphere_bounding_box = m_context->createProgramFromPTXFile( ptx_path, "bounds_atmosphere" );
  //Program sphere_intersection = m_context->createProgramFromPTXFile( ptx_path, "intersect_atmosphere" );
  Program sphere_bounding_box    = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "heightfield.cu" ), "sphere_bounds" );
  Program sphere_intersection = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "heightfield.cu" ), "sphere_intersect" );
  sphere->setIntersectionProgram( sphere_intersection );
  sphere->setBoundingBoxProgram( sphere_bounding_box );
	
  
  // Create material
  float radiusT = 2.0;
  sphere["sphere"]->setFloat( 0.0f, -2.0f, 0.0f, radiusT);
  Program phong_ch = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "phong.cu" ), "closest_hit_radiance" );
  //Program phong_ah = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "phong.cu" ), "any_hit_shadow" );
  Material sphere_matl = m_context->createMaterial();
  sphere_matl["Ka"]->setFloat(0.0f, 0.3f, 0.1f);
  sphere_matl["Kd"]->setFloat(0.1f, 0.7f, 0.2f);
  sphere_matl["Ks"]->setFloat(0.6f, 0.6f, 0.6f);
  sphere_matl["phong_exp"]->setFloat(132);
  sphere_matl["reflectivity"]->setFloat(0, 0, 0);
  sphere_matl->setClosestHitProgram( 0, phong_ch );
  //sphere_matl->setAnyHitProgram( 1, phong_ah );

  GeometryInstance gi_sphere = m_context->createGeometryInstance();

  gi_sphere->setGeometry( sphere );
  gi_sphere->setMaterialCount( 1 );
  gi_sphere->setMaterial( 0, sphere_matl );
  
  
  float3 min = make_float3(-2, ymin, -2);
  float3 max = make_float3( 2, ymax,  2);
  GeometryInstance gi = getGeo(min, max, false);
  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(1));
  geometrygroup->setChild( 0, gi );
  /*
  geometrygroup->setChild( 1, gi_sphere );
  */

  /*
  std::vector<GeometryInstance> gis;
  getParallelogramVector(gis);
  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()));
  //geometrygroup->setChild( 0, gi_sphere );
  for( unsigned int i = 0; i < gis.size(); ++i ) {
    geometrygroup->setChild( i, gis[i] );
  }
*/
  geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );
  //setupNoiseTexture();
  m_context["top_object"]->set( geometrygroup );
  m_context["top_shadower"]->set( geometrygroup );
}


void HeightfieldScene::createData()
{
  if(dataname == "sinxy"){
    fillBuffer(sinxy, m_context, databuffer, 100, 100);
  } else if(dataname == "plane"){
    fillBuffer(plane, m_context, databuffer, 10, 10);
  } else if(dataname == "plane2"){
    fillBuffer(plane2, m_context, databuffer, 10, 10);
  } else if(dataname == "sinc"){
    fillBuffer(sinc, m_context, databuffer, 500, 500);
  } else if(dataname == "perlin"){
    fillBufferPerlin(m_context, databuffer, 1200, 1200);
  } else {
    // Try to open as a file
    std::ifstream in(dataname.c_str(), std::ios::binary);
    if(!in){
      std::cerr << "Error opening '" << dataname << "'\n";
      exit(1);
    }
    int nx, nz;
    in >> nx >> nz;
    if(!in){
      std::cerr << "Error reading header from '" << dataname << "'\n";
      exit(1);
    }
    in.get();
    databuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT, nx+1, nz+1 );
    float* p = reinterpret_cast<float*>(databuffer->map());
    in.read(reinterpret_cast<char*>(p), sizeof(float)*(nx+1)*(nz+1));
    if(!in){
      std::cerr << "Error reading data from '" << dataname << "'\n";
      exit(1);
    }
    databuffer->unmap();
  }

  // Compute data range
  ymin = FLT_MAX;
  ymax = -FLT_MAX;
  RTsize width, height;
  databuffer->getSize(width, height);
  RTsize size = width * height;
  float* p = reinterpret_cast<float*>(databuffer->map());
  for(RTsize i=0;i<size; i++){
    float value = *p++;
    ymin = fminf(ymin, value);
    ymax = fmaxf(ymax, value);
  }
  ymin -= 1.e-6f;
  ymax += 1.e-6f;
  databuffer->unmap();
}


//-----------------------------------------------------------------------------
//
// main 
//
//-----------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"

    << "  -h  | --help                               Print this usage message\n"
    << "  -ds | --dataset <data set>                 Specify data set to render\n"
    << "\n"
    << "<data set>: sinxy | plane | plane2 | sinc | filename\n"
    << std::endl;
  GLUTDisplay::printUsage();

  if ( doExit ) exit(1);
}


int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  // Process command line options
  std::string dataset( "perlin" );
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else if ( arg == "-ds" || arg == "--dataset" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      dataset = argv[++i];
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    HeightfieldScene scene( dataset );
    GLUTDisplay::run( "HeightfieldScene", &scene );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}

