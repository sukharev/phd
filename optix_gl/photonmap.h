#pragma once


#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixpp_namespace.h>
#include <iostream>
#include <GLUTDisplay.h>
#include <sutil.h>
#include <ImageLoader.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include "ppm.h"
#include "select.h"
#include "PpmObjLoader.h"
#include "random.h"
#include "commonStructs.h"
#include <iostream>
#include <fstream>
#include "consts.h"
#include <nvModel.h>

using namespace std;
using namespace optix;


enum SplitChoice {
  RoundRobin,
  HighestVariance,
  LongestDim
};

//-----------------------------------------------------------------------------
//
// Whitted Scene
//
//-----------------------------------------------------------------------------

class ProgressivePhotonScene : public SampleScene
{
public:
  ProgressivePhotonScene() : SampleScene()
    , _frame_number( 0 )
    , _display_debug_buffer( false )
    , _print_timings ( false )
    , _cornell_box( false )
	, _distant_test( false )
    , _light_phi(M_PIf/2.0)//(-M_PIf/4.0)/*_light_phi( 2.19f )*/
    , _light_theta(M_PIf/2.0f)//(-M_PIf)/*_light_theta( 1.15f )*/
    , _split_choice(LongestDim)
	, _camera_index(0)
	, _update_pmap(true)
	, _full_gather(false)
	, _exposure(0.4)
	, _singleOnly(1)
	, _hdrOn(true)
	, _texUnit(18)
	, _bSunActivate(true)
	, _thetaOnly(false)
	, _angleIndex(0)
	, _ctrlFactor(true)
	, _bruneton_single(true)
	, _jm_mult(true)
	, _finishedPhotonTracing(false)
	, _testRtPassLevel(0)
  {
	  _transform_inner_obj = m_context->createTransform( );
	  _transform_inner_obj_inv = m_context->createTransform( );
  }

  void optix_display(float3 in_c, float3 in_s, const float* iproj, const float* iview);
  const char* const ptxpath( const std::string& target, const std::string& base );

  void signalScaleFactorChange(float sfactor);
  void signalTestRtPassChange();
  void signalLightChanged(float3 new_light_dir);
  void getHVAngles(float & vangle, float & hangle);
  void updateGeometry();

  void loadTexToOptix();
  // From SampleScene
  void   signalMouseCoordChanged(Matrix4x4& iviewf, Matrix4x4& iproj, float4 cam_pos);
  void   initScene( InitialCameraData& camera_data );
  bool   keyPressed(unsigned char key, int x, int y);
  void   trace( const RayGenCameraData& camera_data );
  void   doResize( unsigned int width, unsigned int height );
  Buffer getOutputBuffer();

  int getTexUnit() { return _texUnit;}
  void SavePhotonMap(PhotonRecord** temp_photons, int num_photons, const char* filename);

  void setSceneCornellBox() { _cornell_box = true; }
  void setDistantTest()		{ _distant_test = true; _cornell_box = true;}
  void setSceneOBJ()        { _cornell_box = false; }
  void printTimings()       { _print_timings = true; }
  void displayDebugBuffer() { _display_debug_buffer = true; }

  //debugging
  void testPhotonBinCounts();
  void loadBinCountsTexture(float* pTexBuffer, int r,  int width, int height, int depth, int dim);
  optix::TextureSampler loadTexture2uc( unsigned char* data, 
										const unsigned int width, 
										const unsigned int height);
  optix::TextureSampler loadTexture2ucext( unsigned char* data, 
															 const unsigned int width, 
															 const unsigned int height,
														  const unsigned int dim);
  optix::TextureSampler loadTexture2f( float* data, 
							    	  const unsigned int width, 
									  const unsigned int height);
  optix::TextureSampler loadTexture2fext( float* data, 
							    	  const unsigned int width, 
									  const unsigned int height,
									  const unsigned int dim,
									  bool debug = false);
  optix::TextureSampler loadTexture3f( float* data, 
							    	  const unsigned int width, 
									  const unsigned int height,
									  const unsigned int depth,
									  const unsigned int dim,
									  int test = -1,
									  bool debug = false);
  void displayFrame();

  void create2DOptixTex(std::string texName, unsigned int glTex , optix::TextureSampler &texSampler);
  void create3DOptixTex(std::string texName, unsigned int glTex , optix::TextureSampler &texSampler);
  void create2DOptixTexFromBuffer(std::string texName, float* data, int width, int height, int dim, bool debug = false);
  void create2DOptixTexFromBufferuc(std::string texName, unsigned char* data, int width, int height, int dim);
  void create3DOptixTexFromBuffer(std::string texName, float* data, int width, int height, int depth, int dim, int test = -1, bool debug=false);

  bool loadObjFile(const std::string& model_filename);
  bool loadObjFile_materials(const std::string& model_filename, GeometryGroup geometry_group);
  void setObjFilename(string filename);
  void setPhotonFilename(string filename);
  
  void switchAtmosphereMode();
  void switchAtmosphereMode1();
  bool isSingle();
  void switchSunControlState();
  bool isSunControlActivated();
  void switchThetaOnly() 
  { 
	  _thetaOnly = !_thetaOnly;
	  string status = _thetaOnly ? "enabled" : "disabled";
	  cout << "_thetaOnly " << status << endl;
  }
  bool isThetaOnly() { return _thetaOnly;}
  void switchCtrlFactor() 
  { 
	  _ctrlFactor = ! _ctrlFactor;
	  string status = _ctrlFactor ? "enabled" : "disabled";
	  cout << "_ctrlFactor " << status << endl;

  }
  bool isCtrlFactorEnabled() { return _ctrlFactor;}


private:

  void createPhotonRecordMap(Buffer pdata, Buffer pmap, int size, int sizemap, string fname, bool bsave, bool bAccelStruct);
  void createPhotonMap();
  void createVolPhotonMap();
  void createRTPointMap();
  void savePhotonBinCounts(Buffer pdata, int size, string fname, int nc, bool bSave);
  void savePhotonBinCountsHist(string fname, string fname_tbl, bool bSave);
  bool loadPhotonBinCounts(string fname);
  void savePhotonBins(Buffer pdata, int sizex, int sizey, int sizez, string fname);
  void loadObjGeometry( const std::string& filename, optix::Aabb& bbox );
  void createTestGeometryEarthAtm(optix::Aabb& bbox);
  GeometryInstance createParallelogram( const float3& anchor,
                                        const float3& offset1,
                                        const float3& offset2,
                                        const float3& color );
  GeometryInstance createVolSphere(const float3& color, const float radiusT, const float radiusG);
  GeometryInstance createSphere(const float3& color, const float radius, bool bGather=false);
  GeometryInstance createCone(const float3& color, const float radius, const float height, int up_cone, bool bGather=false);


  enum ProgramEnum {
    ppass,
    rtpass,
	//gather_samples,
    //ppass,
    gather,
	pbinpass,
    shadow,
	numPrograms
  };

  Transform _transform_inner_obj;
  Transform _transform_inner_obj_inv;
  static unsigned int   _texId;
  const int _texUnit;
  Acceleration	_sphere_accel; // usef to keep track of inner sphere acceleration structure
  bool			_update_pmap; // controlled by keys 'p' and 'P': forces new photon tracing
  bool			_full_gather; // activates full gather shader (default: false)
  bool			_hdrOn;		  // activates high dynamic range (default: on)
  float			_exposure;
  int			_singleOnly;
  unsigned int  _frame_number;
  bool          _display_debug_buffer;
  bool          _print_timings;
  bool          _cornell_box;
  bool			_distant_test;
  Program       _pgram_bounding_box;
  Program       _pgram_intersection;
  Material      _material;
  Buffer        _display_buffer;
  Buffer        _photons;
  
  Buffer		_vol_photons;
  Buffer		_vol_counts_photons;  //volume photon counts (CUDA indexed table)
  Buffer		_global_photon_counts; //global photon table (_vol_counts_photons updates this table)

  Buffer		_light_sample;
  Buffer		_lightplane_sample;
  Buffer		_rt_points;
  Buffer		_vol_rotate;
  Buffer		_vol_rotate_map;
  Buffer		_photon_table;
  Buffer		_photon_table_map;

  Buffer        _photon_map;
  Buffer        _vol_photon_map;
  Buffer        _rt_point_map;
  Buffer		_light_sample_map;
  Buffer		_lightplane_sample_map;

  Buffer        _debug_buffer;
  Buffer		_pbin_buffer;
  float         _light_phi;
  float         _light_theta;
  unsigned int  _iteration_count;
  unsigned int  _photon_map_size;
  unsigned int  _vol_photon_map_size;
  unsigned int	_rt_point_map_size;
  unsigned int	_rt_light_sample_map_size;
  unsigned int	_rt_lightplane_sample_map_size;
  unsigned int	_vol_rotate_map_size;
  unsigned int	_vol_photon_table_size;
  SplitChoice   _split_choice;
  PPMLight      _light;

  const static unsigned int sqrt_samples_per_pixel;
  const static unsigned int WIDTH;
  const static unsigned int HEIGHT;
  const static unsigned int MAX_PHOTON_COUNT;
  const static unsigned int MAX_VOL_PHOTON_COUNT;
  const static unsigned int PHOTON_LAUNCH_WIDTH;
  const static unsigned int PHOTON_LAUNCH_HEIGHT;
  //const static unsigned int VOL_PHOTON_LAUNCH_WIDTH;
  //const static unsigned int VOL_PHOTON_LAUNCH_HEIGHT;
  const static unsigned int NUM_PHOTONS;
  const static unsigned int NUM_VOL_PHOTONS;

  //used to precompute values
  //Precomp pr;

  unsigned int _camera_index;
  InitialCameraData _camera[3];



  // experiment with changing camera view and projection
  Matrix4x4 _iviewf;
  Matrix4x4 _iproj;
  float4 _cam_pos;
  void initExperimentViewProj();

  nv::Model* _model; // mesh model 
  GLuint _modelVB;
  GLuint _modelIB;

  bool _bSunActivate; //enable/disable sun position change
  bool _thetaOnly;
  bool _ctrlFactor;

  int _angleIndex;
public:
  //textures
  optix::TextureSampler texSRPhaseFuncSampler;
  optix::TextureSampler texSMPhaseFuncSampler;
  optix::TextureSampler texSRPhaseFuncIndSampler;
  optix::TextureSampler texSMPhaseFuncIndSampler;
  optix::TextureSampler textrans_texture;
  optix::TextureSampler texinscatterSampler;
  optix::TextureSampler texraySampler;
  optix::TextureSampler texmieSampler;
  optix::TextureSampler texearth_texture;

  std::string _objfilename;
  std::string _photonfilename;


  GeometryGroup _geometry_group_obj;
  GeometryInstance _gi;

  Group _ggroup;
  Group _ppm_ggroup;
  Group _ggroup_single;

  bool _bruneton_single;
  bool _jm_mult;
  bool _finishedPhotonTracing;
  int _testRtPassLevel;
};