#pragma once
#include "pas.h"
#include "vec3.h"
#include "photonmap.h"
#include "mouse.h"
#include "string"
#include <nvModel.h>
#include <nvTime.h>
#include <framebufferObject.h>
#include <nvGlutManipulators.h>

#ifdef _OLD
#include <Cg/cg.h>
#include <Cg/cgGL.h>
#endif
using namespace std;

class IsgShadow{
public:
	IsgShadow();
	~IsgShadow();

	void init_optix();
#ifdef _OLD
	void draw_earth(CGtechnique& tech);
	void draw_ground(CGtechnique& tech);
	void draw_model(CGtechnique& tech, bool bGLSL = false);
	void draw_scene(CGtechnique& current_technique);
#endif
	void init_gl();
	void init_scene(const char* model_filename);
	GeometryInstance createSphere(const float3& color, const float radius);
public:
	nv::GlutExamine manipulator;

	nv::Model* model;
	GLuint modelVB;
	GLuint modelIB;
	nv::vec3f modelBBMin, modelBBMax, modelBBCenter;
	nv::vec3f lightPos;
	float lightRadius;
	GLuint shadowMapTex;
	GLuint groundTex;

	GLuint worldSpaceTex;
	FramebufferObject* worldSpaceFBO;

	unsigned int warmup_frames, timed_frames;
	unsigned int initialWindowWidth;
	unsigned int initialWindowHeight;
	unsigned int rasterWidth, rasterHeight;
	int logShadowSamplingRate;

	GLuint shadowMapPBO;

	// OptiX
	unsigned int traceWidth, traceHeight;

	optix::Context        rtContext;
	optix::TextureSampler rtWorldSpaceTexture;
	optix::Buffer         rtShadowBuffer;


	bool animate;
	bool moveLight;
	bool doBlur;
	bool doISG;
	bool zUp;
	bool stripNormals;

	bool fixedCamera;
	nv::matrix4f fixedCameraMatrix;

	bool drawFps;

	float scene_epsilon;
};

extern IsgShadow* isg;


#define CHECK_ERRORS()         \
  do {                         \
    GLenum err = glGetError(); \
    if (err) {                                                       \
      printf( "GL Error %d at line %d\n", (int)err, __LINE__);       \
      /*exit(-1);*/                                                      \
    }                                                                \
  } while(0)