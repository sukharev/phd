#pragma once
#include "pas.h"
#include "vec3.h"
#include "photonmap.h"
#include "mouse.h"
#include "string"
using namespace std;

void redisplayFunc_pas();
void reshapeFunc_pas(int x, int y);
void mouseClickFunc_pas(int button, int state, int x, int y);
void mouseMotionFunc_pas(int x, int y);
void specialKeyFunc_pas(int c, int x, int y);
void keyboardFunc_pas(unsigned char c, int x, int y);
void idleFunc_pas();
void loadData();
void precompute();
void updateView();
void init_pas();

extern int PhotonBufferSize;// = BufferSize;
extern float TotalPhotonNum;// = 0.0f;
extern float* BufInfo;//[4];
extern unsigned int FrameCount;// = 0;
extern float* TempData; //[BufferSize * BufferSize * 4];
extern float InitialRadius;// = 0.0f;

void InitializePPMData();
extern vec3f s; // sun direction

#define G_reflectanceUnit		0
#define G_transmittanceUnit		15
#define G_irradianceUnit		2
#define G_inscatterUnit			3
#define G_deltaEUnit			4
#define G_deltaSRUnit			5
#define G_deltaSMUnit			6
#define G_deltaJUnit			7
#define G_photonUnit			8
#define G_randomUnit			9
#define G_photonFluxUnit		10
#define G_SRUnit				11//uint 11
#define G_SRDensUnit			12//uint 12
#define G_SMUnit				13//uint 13
#define G_SMDensUnit			14//uint 14
#define G_testRenderTextureUnit  1
#define G_renderTextureIDUnit	16
#define G_geomRenderTextureUnit 17

unsigned int getTex(unsigned int id);

class GLDisplay{
public:
	GLDisplay(string test= "") 
	{
		_debugstr = test;
		_camera = NULL;

		_sr_phase_func = NULL;
		_sr_phase_func_ind = NULL;
		_sm_phase_func = NULL;
		_sm_phase_func_ind = NULL;
		_phase_func_length = 0;
		_phase_func_dimx = 0;

		_transmittance = NULL;
		_transHeight = 0;
		_transWidth = 0;
	}

	PinholeCamera* _camera;
	ProgressivePhotonScene _scene;
	string _debugstr;

	// pre-computed textures
	float* get_sr_phase_func() { return _sr_phase_func; }
	float* get_sr_phase_func_ind() { return _sr_phase_func_ind; }
	float* get_sm_phase_func() { return _sm_phase_func; }
	float* get_sm_phase_func_ind() { return _sm_phase_func_ind; }
	void store_var(float* data, int size, float* & dest_var)
	{
		if(dest_var)
			delete [] dest_var;
		dest_var = new float[size];
		for(int i = 0; i<size; i++){
			dest_var[i] = data[i];
		}	
		_phase_func_length = size;
		_phase_func_dimx = size/3;
	}
	int getDimX() { return _phase_func_dimx;}

	float* _sr_phase_func;
	float* _sr_phase_func_ind;

	float* _sm_phase_func;
	float* _sm_phase_func_ind;
	int _phase_func_length;
	int _phase_func_dimx;

	// transmittance
	float* getTransmittance() { return _transmittance;}
	unsigned int getTransmittanceHeight() { return _transHeight;}
	unsigned int getTransmittanceWidth() { return _transWidth;}

	void store_2Dvar(float* data, int sizex, int sizey, int dim, float* & dest_var, unsigned int &varsizex, unsigned int& varsizey)
	{
		int size = sizex*sizey*dim;
		if(dest_var)
			delete [] dest_var;
		dest_var = new float[size];
		for(int i = 0; i<size; i++){
			dest_var[i] = data[i];
		}	
		varsizex = sizex;
		varsizey = sizey;
	}

	
	float* _transmittance;
	unsigned int _transHeight;
	unsigned int _transWidth;

};

extern GLDisplay* disp;