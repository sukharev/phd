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

#include <cstdlib>

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined(_WIN32)
#    include <GL/wglew.h>
#  endif
#  include <GL/glut.h>
#endif

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
//#include <cutil_inline.h>

//#include <GL/glew.h>
//#include <glee.h>
//#ifdef __gl_h_
//#undef __gl_h_
//#endif

//#include "GLTools.h"
//#include <GL/glut.h>
//#include <GL/glext.h>
#include "tiffio.h"

#include "vec3.h"
#include "mat4.h"
#include "pas.h"
#include "pas_api.h"
#include "isgshadow.h"

#ifdef _OLD
CGtechnique emptyPlaceholder;
#endif

GLint maxTexSize;                       // maximum allowed size for our 2D textures
GLint maxTexUnits;                      // maximum number of texture image units
GLint maxDrawBuffers;                   // maximum number of drawbuffers supported
GLint maxColorAttachments;              // maximum number of FBO color attachments
GLuint framebufferID[2];                // FBO names
GLuint renderbufferID;                  // renderbuffer object name
GLint maxRenderbufferSize;              // maximum allowed size for FBO renderbuffer
GLint fboWidth = 512;//1024;                     // set based on window size
GLint fboHeight = 512;//800;
#define NUM_TEX_FBO	2
GLuint renderTextureID[NUM_TEX_FBO];              // 1 in 1st pass and up to 4 simultaneous render targets in 2nd
bool bUpdateRendering = true;
void test_renderer();

//#include <cutil_inline.h>
using namespace std;

#define MAX_PATH_LENGTH 1	//photon tracing

//#define NEYRET_MULT_SCATTERING  //enable bruneton/neyret multiple scattering incremental method

float* getTextureData(GLenum id, unsigned texId, GLenum format, int texX, int texY, int sizeDim, bool bIgnore=true);

#ifdef TEST_PAS
const int PhotonBufferSize = BufferSize;
float TotalPhotonNum = 0.0f;
float BufInfo[4];
unsigned int FrameCount = 0;
float TempData[BufferSize * BufferSize * 4];
float InitialRadius = 0.0f;

void InitializePPMData();
vec3f s(0.0, -1.0, 0.0); // sun position
#else

int PhotonBufferSize = BufferSize;
float TotalPhotonNum = 0.0f;
unsigned int FrameCount = 0;
float InitialRadius = 0.0f;
float* TempData = new float[BufferSize * BufferSize * 4];
float* BufInfo = new float[4];
vec3f s = vec3f(0.0, -1.0, 0.0); // sun position

#endif
// ----------------------------------------------------------------------------
// TOOLS
// ----------------------------------------------------------------------------

namespace XORShift
{
	// XOR shift PRNG
	unsigned int x = 123456789;
	unsigned int y = 362436069;
	unsigned int z = 521288629;
	unsigned int w = 88675123; 

	inline float frand()
	{ 
		unsigned int t;
	 
		t = x ^ (x << 11);
		x = y; y = z; z = w;
		return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))) * (1.0f / 4294967295.0f); 
	}
}


void loadTIFF(char *name, unsigned char *tex)
{
    tstrip_t strip = 0;
    tsize_t off = 0;
    tsize_t n = 0;
    TIFF* tf = TIFFOpen(name, "r");
    while ((n = TIFFReadEncodedStrip(tf, strip, tex + off, (tsize_t) -1)) > 0) {
    	strip += 1;
        off += n;
    };
    TIFFClose(tf);
}

string* loadFile(const string &fileName)
{
    string* result = new string();
    ifstream file(fileName.c_str());
    if (!file) {
        std::cerr << "Cannot open file " << fileName << endl;
        throw exception();
    }
    string line;
    while (getline(file, line)) {
        *result += line;
        *result += '\n';
    }
    file.close();
    return result;
}

void printShaderLog(int shaderId)
{
    int logLength;
    glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 0) {
        char *log = new char[logLength];
        glGetShaderInfoLog(shaderId, logLength, &logLength, log);
        cout << string(log);
        delete[] log;
    }
}

unsigned int loadProgram(const vector<string> &files)
{
    unsigned int programId = glCreateProgram();
    unsigned int vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
    unsigned int fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
    glAttachShader(programId, vertexShaderId);
    glAttachShader(programId, fragmentShaderId);

    int n = files.size();
    string **strs = new string*[n];
    const char** lines = new const char*[n + 1];
    cout << "loading program " << files[n - 1] << "..." << endl;
    bool geo = false;
    for (int i = 0; i < n; ++i) {
        string* s = loadFile(files[i]);
        strs[i] = s;
        lines[i + 1] = s->c_str();
        if (strstr(lines[i + 1], "_GEOMETRY_") != NULL) {
            geo = true;
        }
    }

    lines[0] = "#define _VERTEX_\n";
    glShaderSource(vertexShaderId, n + 1, lines, NULL);
    glCompileShader(vertexShaderId);
    printShaderLog(vertexShaderId);

    if (geo) {
        unsigned geometryShaderId = glCreateShader(GL_GEOMETRY_SHADER_EXT);
        glAttachShader(programId, geometryShaderId);
        lines[0] = "#define _GEOMETRY_\n";
        glShaderSource(geometryShaderId, n + 1, lines, NULL);
        glCompileShader(geometryShaderId);
        printShaderLog(geometryShaderId);
        glProgramParameteriEXT(programId, GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
        glProgramParameteriEXT(programId, GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
        glProgramParameteriEXT(programId, GL_GEOMETRY_VERTICES_OUT_EXT, 3);
    }

    lines[0] = "#define _FRAGMENT_\n";
    glShaderSource(fragmentShaderId, n + 1, lines, NULL);
    glCompileShader(fragmentShaderId);
    printShaderLog(fragmentShaderId);

    glLinkProgram(programId);

    for (int i = 0; i < n; ++i) {
        delete strs[i];
    }
    delete[] strs;
    delete[] lines;

    return programId;
}

void drawQuad()
{
    glBegin(GL_TRIANGLE_STRIP);
    glVertex2f(-1.0, -1.0);
    glVertex2f(+1.0, -1.0);
    glVertex2f(-1.0, +1.0);
    glVertex2f(+1.0, +1.0);
    glEnd();
}

// ----------------------------------------------------------------------------
// PRECOMPUTATIONS
// ----------------------------------------------------------------------------


const int reflectanceUnit = G_reflectanceUnit;
const int transmittanceUnit = G_transmittanceUnit;
const int irradianceUnit = G_irradianceUnit;
const int inscatterUnit = G_inscatterUnit;
const int deltaEUnit = G_deltaEUnit;
const int deltaSRUnit = G_deltaSRUnit;
const int deltaSMUnit = G_deltaSMUnit;
const int deltaJUnit = G_deltaJUnit;
const int photonUnit = G_photonUnit;
const int randomUnit = G_randomUnit;
const int photonFluxUnit = G_photonFluxUnit;
const int SRUnit = G_SRUnit;//uint 11
const int SRDensUnit = G_SRDensUnit;//uint 12
const int SMUnit = G_SMUnit;//uint 13
const int SMDensUnit = G_SMDensUnit;//uint 14
const int testRenderTextureUnit = G_testRenderTextureUnit;
const int renderTextureIDUnit = G_renderTextureIDUnit;
const int geomRenderTextureUnit = G_geomRenderTextureUnit;

unsigned int reflectanceTexture;//unit 0, ground reflectance texture
unsigned int transmittanceTexture;//unit 1, T table
unsigned int irradianceTexture;//unit 2, E table
unsigned int inscatterTexture;//unit 3, S table
unsigned int deltaETexture;//unit 4, deltaE table
unsigned int deltaSRTexture;//unit 5, deltaS table (Rayleigh part)
unsigned int deltaSMTexture;//unit 6, deltaS table (Mie part)
unsigned int deltaJTexture;//unit 7, deltaJ table

//unsigned int photonPositionTexture;//unit 8
//unsigned int randomTexture;//unit 9
//unsigned int photonFluxTexture;//unit 10

// these textures used to calculate tables
// for converting random [0,1) into non-uniform densities of phase functions
unsigned int SRTexture;//unit 8
unsigned int SRDensTexture;//unit 9
unsigned int SMTexture;//unit 10
unsigned int SMDensTexture;//unit 11
unsigned int testRenderTexture; // unit 15
unsigned int geomRenderTexture; // unit 17

GLuint PhotonRayTraceSurface;

unsigned int transmittanceProg;
unsigned int irradiance1Prog;
unsigned int inscatter1Prog;
unsigned int copyIrradianceProg;
unsigned int copyInscatter1Prog;
unsigned int jProg;
unsigned int irradianceNProg;
unsigned int inscatterNProg;
unsigned int copyInscatterNProg;
unsigned int photonProg;

unsigned int phaseFuncProg;
unsigned int phaseDensFuncProg;

unsigned int fbo;
unsigned int fbo1;

unsigned int drawProg;
unsigned int drawGeometryProg;
unsigned int combineProg;

unsigned int getTex(unsigned int id)
{
	switch(id){
		case G_reflectanceUnit:
			return reflectanceTexture;
		case G_transmittanceUnit:
			return transmittanceTexture;
		case G_irradianceUnit:
			return irradianceTexture;
		case G_inscatterUnit:
			return inscatterTexture;
		case G_deltaEUnit:
			return deltaETexture;
		case G_deltaSRUnit:
			return deltaSRTexture;
		case G_deltaSMUnit:
			return deltaSMTexture;
		case G_deltaJUnit:
			return deltaJTexture;
		//case G_photonUnit:
		//	return photonTexture;
		case G_randomUnit:
			return reflectanceTexture;
		//case G_photonFluxUnit:
		//	return photonFluxTexture;
		case G_SRUnit:
			return SRTexture;
		case G_SRDensUnit:
			return SRDensTexture;
		case G_SMUnit:
			return reflectanceTexture;
		case G_SMDensUnit:
			return SMTexture;
		case G_testRenderTextureUnit:
			return testRenderTexture;
		case G_geomRenderTextureUnit:
			return geomRenderTexture;
		default:
			return 0;
	}
}

void setLayer(unsigned int prog, int layer)
{
    double r = layer / (RES_R - 1.0);
    r = r * r;
    r = sqrt(Rg * Rg + r * (Rt * Rt - Rg * Rg)) + (layer == 0 ? 0.01 : (layer == RES_R - 1 ? -0.001 : 0.0));
    double dmin = Rt - r;
    double dmax = sqrt(r * r - Rg * Rg) + sqrt(Rt * Rt - Rg * Rg);
    double dminp = r - Rg;
    double dmaxp = sqrt(r * r - Rg * Rg);
    glUniform1f(glGetUniformLocation(prog, "r"), float(r));
    glUniform4f(glGetUniformLocation(prog, "dhdH"), float(dmin), float(dmax), float(dminp), float(dmaxp));
    glUniform1i(glGetUniformLocation(prog, "layer"), layer);
}

#define TEX2OPTIX
void loadData()
{
    cout << "loading Earth texture..." << endl;
    glActiveTexture(GL_TEXTURE0 + reflectanceUnit);
    glGenTextures(1, &reflectanceTexture);
    glBindTexture(GL_TEXTURE_2D, reflectanceTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    unsigned char *tex = new unsigned char[2500 * 1250 * 4];
    //unsigned char *tex = new unsigned char[5400 * 2700 * 4];
	//unsigned char *tex = new unsigned char[2642 * 1488 * 4];
	loadTIFF("earth.tiff", tex);
	//loadTIFF("earth_b.tiff", tex);
	//loadTIFF("test.tiff", tex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 2500, 1250, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 5400, 2700, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2642, 1488, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);
    glGenerateMipmapEXT(GL_TEXTURE_2D);
#ifdef TEX2OPTIX
	disp->_scene.create2DOptixTexFromBufferuc("earth_texture", tex, 2500, 1250, 4);
	
	//disp->_scene.create2DOptixTex("earth_texture", reflectanceTexture , disp->_scene.texearth_texture);
#endif
    delete[] tex;
}


void SavePhotonMap(GLenum id_pos, GLenum id_dir, int texX, int texY, int sizeDim, char* filename)
{
	if(!filename)
		return;
	float* pTexBufferPos = new float[texX * texY * sizeDim];
	glReadBuffer(id_pos);
	glReadPixels(0, 0, texX, texY, GL_RGB, GL_FLOAT, pTexBufferPos);

	float* pTexBufferDir = new float[texX * texY * sizeDim];
	glReadBuffer(id_dir);
	glReadPixels(0, 0, texX, texY, GL_RGB, GL_FLOAT, pTexBufferDir);
	//char filename[1024] = "c:\\SVN\\volumemap.bin";
	
	ofstream outf(filename, ios::binary);
	assert(outf);	
	if(!outf)
		return;

	int size = texX*texY;
	const int COLOR_SAMPLES = 3;
	int cs = COLOR_SAMPLES;
	outf.write(reinterpret_cast<char *>(&size), sizeof(int));
	outf.write(reinterpret_cast<char *>(&cs), sizeof(int));


	for(int i=0; i<texX; i++){
		for(int j=0; j<texY; j++){
			for(int k=0; k<sizeDim; k++){
				outf.write(reinterpret_cast<char *>(&pTexBufferPos[i*texY*sizeDim + j*sizeDim+k]), sizeof(float));
			}
			for(int k=0; k<sizeDim; k++){
				outf.write(reinterpret_cast<char *>(&pTexBufferDir[i*texY*sizeDim + j*sizeDim+k]), sizeof(float));
			}
		}
	}
    //outf.write(reinterpret_cast<char *>(&x_start), sizeof(int));
	//outf.write(reinterpret_cast<char *>(&x_end), sizeof(int));	
	//outf.write(reinterpret_cast<char *>(&y_start), sizeof(int));
	
	outf.close();
}

#define MAX_PATH 512

void saveTextureToXYZ_3Dslice(GLenum id, int texX, int texY, int depth, int sizeDim, char* filename, bool bDebug = false)
{
	//int dim = 3; //texDim
	float* pTexBuffer = new float[texX * texY * sizeDim];
	glReadBuffer(id);
	glReadPixels(0, 0, texX, texY, GL_RGB, GL_FLOAT, pTexBuffer);
	//glCopyTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, 0, 0, RES_MU_S * RES_NU, RES_MU, 0 );
	//glGetTexImage( GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pTexBuffer ); 
	FILE* fp = fopen(filename,"w");
	if(!fp){
		printf("\nERROR: cannot open file %s\n",filename);
		return;
	}
	for(int i=0; i<texX; i++){
		for(int j=0; j<texY; j++){
			for(int k=0; k<sizeDim; k++){
				if(k < sizeDim -1)
					fprintf(fp,"%f ",pTexBuffer[i*texY*sizeDim + j*sizeDim+k]);
				else
					fprintf(fp,"%f",pTexBuffer[i*texY*sizeDim + j*sizeDim+k]);
			}
			fprintf(fp,"\n");
		}
	}
	fclose(fp);
	
	if(bDebug){
		float max_x = -100.0;
		float max_y = -100.0;
		float max_z = -100.0;
		float max_w = -100.0;
		//assert(sizeDim == sizeDim);
		sizeDim = 3;
		unsigned char *a_f =  (unsigned char*)malloc(sizeof(unsigned char)*sizeDim*texY*texX);
		for(int i = 0; i < texX; i++){
			for(int j = 0; j < texY; j++){
				if(max_x < pTexBuffer[i*texY*sizeDim + j*sizeDim]) max_x = pTexBuffer[i*texY*sizeDim + j*sizeDim];
				if(max_y < pTexBuffer[i*texY*sizeDim + j*sizeDim + 1]) max_y = pTexBuffer[i*texY*sizeDim + j*sizeDim + 1];
				if(sizeDim >= 3)
					if(max_z < pTexBuffer[i*texY*sizeDim + j*sizeDim + 2]) max_z = pTexBuffer[i*texY*sizeDim + j*sizeDim + 2];
				if(sizeDim == 4)
					if(max_z < pTexBuffer[i*texY*sizeDim + j*sizeDim + 3]) max_z = pTexBuffer[i*texY*sizeDim + j*sizeDim + 3];
			}
		}

		for(int i = 0; i < texX; i++){
			for(int j = 0; j < texY; j++){
				a_f[i*texY*sizeDim + j*sizeDim] = (unsigned char)((255)*(pTexBuffer[i*texY*sizeDim + j*sizeDim]/max_x));
				a_f[i*texY*sizeDim + j*sizeDim + 1] = (unsigned char)((255)*(pTexBuffer[i*texY*sizeDim + j*sizeDim + 1]/max_y));
				if(sizeDim >= 3)
					a_f[i*texY*sizeDim + j*sizeDim + 2] = (unsigned char)((255)*(pTexBuffer[i*texY*sizeDim + j*sizeDim + 2]/max_z));
				if(sizeDim == 4)
					a_f[i*texY*sizeDim + j*sizeDim + 3] = (unsigned char)((255)*(pTexBuffer[i*texY*sizeDim + j*sizeDim + 3]/max_z));
			}
		}
		char debugfn[MAX_PATH];
		sprintf(debugfn,"%s.ppm",filename);
		//cutSavePPMub( debugfn, a_f, texX, texY);
	}
	delete [] pTexBuffer;
}

void saveTextureToXYZ3D(GLenum id, GLuint texture, int texX, int texY, int texZ, int sizeDim, char* filename, bool bDebug = false)
{
	//int dim = 3; //texDim
	float* pTexBuffer = new float[texX * texY * texZ * sizeDim];
	//glReadBuffer(id);
	glBindTexture(GL_TEXTURE_3D, texture);
	glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, pTexBuffer);
	
	char fn[MAX_PATH];
	sprintf(fn,"%s.xyz",filename);
	FILE* fp = fopen(fn,"w");
	if(!fp){
		printf("\nERROR: cannot open file %s\n",fn);
		return;
	}
	for(int d = 0; d < texZ; d++){
		for(int i=0; i<texX; i++){
			for(int j=0; j<texY; j++){
				for(int k=0; k<sizeDim; k++){
					if(k < sizeDim -1)
						fprintf(fp,"%f ",pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim+k]);
					else
						fprintf(fp,"%f",pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim+k]);
				}
				fprintf(fp,"\n");
			}
		}
		fprintf(fp,"____d=%d________________\n",d);
	}
	fclose(fp);

	if(bDebug){
		for(int d = 0; d < texZ; d++){
			float max_x = -100.0;
			float max_y = -100.0;
			float max_z = -100.0;
			float max_w = -100.0;
			//assert(sizeDim == sizeDim);
			//sizeDim = 3;
			unsigned char *a_f =  (unsigned char*)new unsigned char[sizeDim*texY*texX];
			for(int i = 0; i < texX; i++){
				for(int j = 0; j < texY; j++){
					if(max_x < pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim]) max_x = pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim];
					if(max_y < pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim + 1]) max_y = pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim + 1];
					if(sizeDim >= 3)
						if(max_z < pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim + 2]) max_z = pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim + 2];
					//if(sizeDim == 4)
					//	if(max_z < pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim + 3]) max_z = pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim + 3];
				}
			}

			for(int i = 0; i < texX; i++){
				for(int j = 0; j < texY; j++){
					a_f[i*texY*sizeDim + j*sizeDim] = (unsigned char)((255)*(pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim]/max_x));
					a_f[i*texY*sizeDim + j*sizeDim + 1] = (unsigned char)((255)*(pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim + 1]/max_y));
					if(sizeDim >= 3)
						a_f[i*texY*sizeDim + j*sizeDim + 2] = (unsigned char)((255)*(pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim + 2]/max_z));
					if(sizeDim == 4)
						a_f[i*texY*sizeDim + j*sizeDim + 3] = (unsigned char)((255)*(pTexBuffer[d*texX*texY*sizeDim + i*texY*sizeDim + j*sizeDim + 3]/max_z));
				}
			}
			char debugfn[MAX_PATH];
			sprintf(debugfn,"%s_%d.ppm",filename,d);
			//cutSavePPMub( debugfn, a_f, texX, texY);
			delete [] a_f;
		}
	}
	delete [] pTexBuffer;
}

void getMinMax(GLenum id, int texX, int texY, int sizeDim, float &minval, float& maxval)
{
	float* pTexBuffer = new float[texX * texY * sizeDim];
	glReadBuffer(id);
	glReadPixels(0, 0, texX, texY, GL_RGB, GL_FLOAT, pTexBuffer);
	minval = pTexBuffer[0];
	maxval = pTexBuffer[0];
	for(int i=0; i<texX; i++){
		for(int j=0; j<texY; j++){
			float v = pTexBuffer[i*texY*sizeDim + j*sizeDim+0];
			if(v < minval)
				minval = v;
			if(v > maxval)
				maxval = v;
		}
	}
	delete [] pTexBuffer;

}

float* getTextureData(GLenum id, unsigned texId, GLenum format, int texX, int texY, int sizeDim, bool bIgnore)
{
	float* pTexBuffer = new float[texX * texY * sizeDim];
	if(bIgnore)
		glReadBuffer(id);
	glBindTexture(GL_TEXTURE_2D, texId);
	glReadPixels(0, 0, texX, texY, format, GL_FLOAT, pTexBuffer);
	return pTexBuffer;
}

float* getTextureData3D(GLenum id, unsigned texId, int texX, int texY, int texZ, int sizeDim, GLenum format)
{
	glReadBuffer(id);
	float* pTexBuffer = new float[texX * texY * texZ * sizeDim];
	glBindTexture(GL_TEXTURE_3D, texId);
	glGetTexImage(GL_TEXTURE_3D, 0, format, GL_FLOAT, pTexBuffer);
	return pTexBuffer;

}

//assumptions: texture id is 2D texture
void saveTextureToXYZ(GLenum id, GLenum format, int texX, int texY, int sizeDim, char* filename, bool bDebug = false)
{
	//int dim = 3; //texDim
	float* pTexBuffer = new float[texX * texY * sizeDim];
	glReadBuffer(id);
	glReadPixels(0, 0, texX, texY, format, GL_FLOAT, pTexBuffer);
	//glCopyTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, 0, 0, RES_MU_S * RES_NU, RES_MU, 0 );
	//glGetTexImage( GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pTexBuffer ); 
	FILE* fp = fopen(filename,"w");
	if(!fp){
		delete [] pTexBuffer;
		printf("\nERROR: cannot open file %s\n",filename);
		return;
	}
	for(int i=0; i<texX; i++){
		for(int j=0; j<texY; j++){
			for(int k=0; k<sizeDim; k++){
				if(k < sizeDim -1)
					fprintf(fp,"%f ",pTexBuffer[i*texY*sizeDim + j*sizeDim+k]);
				else
					fprintf(fp,"%f",pTexBuffer[i*texY*sizeDim + j*sizeDim+k]);
			}
			fprintf(fp,"\n");
		}
	}
	fclose(fp);
	
	if(bDebug){
		float max_x = -100.0;
		float max_y = -100.0;
		float max_z = -100.0;
		float max_w = -100.0;
		//assert(sizeDim == sizeDim);
		sizeDim = 3;
		unsigned char *a_f =  (unsigned char*)new unsigned char[sizeDim*texY*texX];
		for(int i = 0; i < texX; i++){
			for(int j = 0; j < texY; j++){
				if(max_x < pTexBuffer[i*texY*sizeDim + j*sizeDim]) max_x = pTexBuffer[i*texY*sizeDim + j*sizeDim];
				if(max_y < pTexBuffer[i*texY*sizeDim + j*sizeDim + 1]) max_y = pTexBuffer[i*texY*sizeDim + j*sizeDim + 1];
				if(sizeDim >= 3)
					if(max_z < pTexBuffer[i*texY*sizeDim + j*sizeDim + 2]) max_z = pTexBuffer[i*texY*sizeDim + j*sizeDim + 2];
				if(sizeDim == 4)
					if(max_z < pTexBuffer[i*texY*sizeDim + j*sizeDim + 3]) max_z = pTexBuffer[i*texY*sizeDim + j*sizeDim + 3];
			}
		}

		for(int i = 0; i < texX; i++){
			for(int j = 0; j < texY; j++){
				a_f[i*texY*sizeDim + j*sizeDim] = (unsigned char)((255)*(pTexBuffer[i*texY*sizeDim + j*sizeDim]/max_x));
				a_f[i*texY*sizeDim + j*sizeDim + 1] = (unsigned char)((255)*(pTexBuffer[i*texY*sizeDim + j*sizeDim + 1]/max_y));
				if(sizeDim >= 3)
					a_f[i*texY*sizeDim + j*sizeDim + 2] = (unsigned char)((255)*(pTexBuffer[i*texY*sizeDim + j*sizeDim + 2]/max_z));
				if(sizeDim == 4)
					a_f[i*texY*sizeDim + j*sizeDim + 3] = (unsigned char)((255)*(pTexBuffer[i*texY*sizeDim + j*sizeDim + 3]/max_z));
			}
		}
		char debugfn[MAX_PATH];
		sprintf(debugfn,"%s.ppm",filename);
		//cutSavePPMub( debugfn, a_f, texX, texY);
		delete [] a_f;
	}
	delete [] pTexBuffer;
}

void precompute()
{
    glActiveTexture(GL_TEXTURE0 + transmittanceUnit);
    glGenTextures(1, &transmittanceTexture);
    glBindTexture(GL_TEXTURE_2D, transmittanceTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, fboWidth, fboHeight, 0, GL_RGB, GL_FLOAT, NULL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, TRANSMITTANCE_W, TRANSMITTANCE_H, 0, GL_RGB, GL_FLOAT, NULL);

    glActiveTexture(GL_TEXTURE0 + irradianceUnit);
    glGenTextures(1, &irradianceTexture);
    glBindTexture(GL_TEXTURE_2D, irradianceTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, SKY_W, SKY_H, 0, GL_RGB, GL_FLOAT, NULL);

    glActiveTexture(GL_TEXTURE0 + inscatterUnit);
    glGenTextures(1, &inscatterTexture);
    glBindTexture(GL_TEXTURE_3D, inscatterTexture);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    //glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F_ARB, RES_MU_S * RES_NU, RES_MU, RES_R, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, RES_MU_S * RES_NU, RES_MU, RES_R, 0, GL_RGBA, GL_FLOAT, NULL);

    glActiveTexture(GL_TEXTURE0 + deltaEUnit);
    glGenTextures(1, &deltaETexture);
    glBindTexture(GL_TEXTURE_2D, deltaETexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, SKY_W, SKY_H, 0, GL_RGB, GL_FLOAT, NULL);

    glActiveTexture(GL_TEXTURE0 + deltaSRUnit);
    glGenTextures(1, &deltaSRTexture);
    glBindTexture(GL_TEXTURE_3D, deltaSRTexture);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB16F_ARB, RES_MU_S * RES_NU, RES_MU, RES_R, 0, GL_RGB, GL_FLOAT, NULL);

    glActiveTexture(GL_TEXTURE0 + deltaSMUnit);
    glGenTextures(1, &deltaSMTexture);
    glBindTexture(GL_TEXTURE_3D, deltaSMTexture);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB16F_ARB, RES_MU_S * RES_NU, RES_MU, RES_R, 0, GL_RGB, GL_FLOAT, NULL);

    glActiveTexture(GL_TEXTURE0 + deltaJUnit);
    glGenTextures(1, &deltaJTexture);
    glBindTexture(GL_TEXTURE_3D, deltaJTexture);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB16F_ARB, RES_MU_S * RES_NU, RES_MU, RES_R, 0, GL_RGB, GL_FLOAT, NULL);

	glActiveTexture(GL_TEXTURE0 + SRUnit);
    glGenTextures(1, &SRTexture);
    glBindTexture(GL_TEXTURE_1D, SRTexture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F_ARB,  PHASE_DISTR_RES, 0, GL_RGB, GL_FLOAT, NULL);

	glActiveTexture(GL_TEXTURE0 + SMUnit);
    glGenTextures(1, &SMTexture);
    glBindTexture(GL_TEXTURE_1D, SMTexture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F_ARB,  PHASE_DISTR_RES, 0, GL_RGB, GL_FLOAT, NULL);

	glActiveTexture(GL_TEXTURE0 + SRDensUnit);
    glGenTextures(1, &SRDensTexture);
    glBindTexture(GL_TEXTURE_1D, SRDensTexture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F_ARB,  PHASE_DISTR_RES, 0, GL_RGB, GL_FLOAT, NULL);

	glActiveTexture(GL_TEXTURE0 + SMDensUnit);
    glGenTextures(1, &SMDensTexture);
    glBindTexture(GL_TEXTURE_1D, SMDensTexture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F_ARB,  PHASE_DISTR_RES, 0, GL_RGB, GL_FLOAT, NULL);


	
	glActiveTexture(GL_TEXTURE0 + testRenderTextureUnit);
    glGenTextures(1, &testRenderTexture);
    glBindTexture(GL_TEXTURE_2D, testRenderTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, fboWidth, fboHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	glActiveTexture(GL_TEXTURE0 + geomRenderTextureUnit);
    glGenTextures(1, &geomRenderTexture);
    glBindTexture(GL_TEXTURE_2D, geomRenderTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, fboWidth, fboHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, fboWidth, fboHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	// Set up the render textures: 1 for 1st pass, up to 4 for 2nd
    //glGenTextures(4, renderTextureID);
/*
    for (int i = 0; i < NUM_TEX_FBO; i++)
    {
		assert(GL_MAX_TEXTURE_UNITS > GL_TEXTURE0 + renderTextureIDUnit + i);
		glActiveTexture(GL_TEXTURE0 + renderTextureIDUnit + i);
    
		glGenTextures(1, &renderTextureID[i]);
		glBindTexture(GL_TEXTURE_2D, renderTextureID[i]);
		
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
        if (i == 0)
        {
            // The 1st pass texture needs to be mipmapped for the enhanced blur effect
            //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        }

        // this may change with window size changes
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, fboWidth, fboHeight, 0, GL_RGB, GL_FLOAT, 0);
    }
*/
	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &maxTexUnits);

    vector<string> files;
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("transmittance.glsl");
    transmittanceProg = loadProgram(files);

    files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("irradiance1.glsl");
    irradiance1Prog = loadProgram(files);

    files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("inscatter1.glsl");
    inscatter1Prog = loadProgram(files);

    files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("copyIrradiance.glsl");
    copyIrradianceProg = loadProgram(files);

    files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("copyInscatter1.glsl");
    copyInscatter1Prog = loadProgram(files);

    files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("inscatterS.glsl");
    jProg = loadProgram(files);
/*
    files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("photonTracing.glsl");
    photonProg = loadProgram(files);
*/
	files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("irradianceN.glsl");
    irradianceNProg = loadProgram(files);

    files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("inscatterN.glsl");
    inscatterNProg = loadProgram(files);

    files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("copyInscatterN.glsl");
    copyInscatterNProg = loadProgram(files);

    files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("earth.glsl");
    drawProg = loadProgram(files);
    

	
	files.clear();
	files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("geom.glsl");
    drawGeometryProg = loadProgram(files);
    

	files.clear();
	files.push_back("combine.glsl");
	combineProg = loadProgram(files);
    //glUseProgram(combineProg);

	files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("phasefunc.glsl");
    phaseFuncProg = loadProgram(files);

	files.clear();
    files.push_back("pas.h");
    files.push_back("common.glsl");
    files.push_back("phasedensfunc.glsl");
	phaseDensFuncProg = loadProgram(files);
	
	glUseProgram(drawProg);
    glUniform1i(glGetUniformLocation(drawProg, "reflectanceSampler"), reflectanceUnit);
    glUniform1i(glGetUniformLocation(drawProg, "transmittanceSampler"), transmittanceUnit);
    glUniform1i(glGetUniformLocation(drawProg, "irradianceSampler"), irradianceUnit);
    glUniform1i(glGetUniformLocation(drawProg, "inscatterSampler"), inscatterUnit);

	//deltaEUnit = 4;
	//deltaSRUnit = 5;
	//deltaSMUnit = 6;
	//deltaJUnit = 7;
	//glUniform1i(glGetUniformLocation(drawProg, "reflectanceSampler"), deltaEUnit);
	//glUniform1i(glGetUniformLocation(drawProg, "inscatterSampler"), deltaSRUnit);

    cout << "precomputations..." << endl;

	// we'll use up to 4 render targets if they're available
    // this is a setup stage for future render buffers
	/*
	glGetIntegerv(GL_MAX_DRAW_BUFFERS, &maxDrawBuffers);
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS_EXT, &maxColorAttachments);
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &maxTexUnits);
    maxDrawBuffers = (maxDrawBuffers > maxColorAttachments) ? maxColorAttachments : maxDrawBuffers;
    maxDrawBuffers = (maxDrawBuffers > (maxTexUnits-1)) ? (maxTexUnits-1) : maxDrawBuffers;
    maxDrawBuffers = (maxDrawBuffers > 4) ? 4 : maxDrawBuffers;

	// Set up some renderbuffer state
    glGenRenderbuffersEXT(2, &renderbufferID);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderbufferID);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32, fboWidth, fboHeight);

    glGenFramebuffersEXT(3, framebufferID);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebufferID[0]);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderbufferID);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, renderTextureID[0], 0);
	
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebufferID[1]);
    for (int i = 0; i < maxDrawBuffers; i++)
    {
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT + i, GL_TEXTURE_2D, renderTextureID[i+1], 0);
    }
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    glGenFramebuffersEXT(1, &fbo1);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo1);
	for (int i = 0; i < NUM_TEX_FBO; i++)
    {
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT + i, GL_TEXTURE_2D, renderTextureID[i], 0);
    }
	 GLenum fboStatus = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    if (fboStatus != GL_FRAMEBUFFER_COMPLETE_EXT)
    {
        fprintf(stderr, "FBO Error!\n");
    }
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	glGenRenderbuffersEXT(1, &renderbufferID);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderbufferID);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32, fboWidth, fboHeight);

	
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glDeleteFramebuffersEXT(1, &fbo1);
    glGenFramebuffersEXT(1, &fbo1);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo1);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderbufferID);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, testRenderTexture, 0);

	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
*/
	glDeleteFramebuffersEXT(1, &fbo);
    glGenFramebuffersEXT(1, &fbo);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	//glDrawBuffer(GL_NONE);
    //glReadBuffer(GL_NONE);

	

    // computes transmittance texture T (line 1 in algorithm 4.1)
    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, transmittanceTexture, 0);
	glViewport(0, 0, TRANSMITTANCE_W, TRANSMITTANCE_H);
    glUseProgram(transmittanceProg);
    drawQuad();
	//float* tmp = getTextureData((GLenum)GL_COLOR_ATTACHMENT0_EXT, TRANSMITTANCE_W, TRANSMITTANCE_H, 3);
	//disp->store_2Dvar(tmp,TRANSMITTANCE_W, TRANSMITTANCE_H, 3, disp->_transmittance, disp->_transWidth, disp->_transHeight);
	//delete [] tmp;
	//saveTextureToXYZ((GLenum)GL_COLOR_ATTACHMENT0_EXT, TRANSMITTANCE_W, TRANSMITTANCE_H, 3, "c:\\temp\\transmittance.XYZ", true);
	
#ifdef TEX2OPTIX
	//saveTextureToXYZ((GLenum)GL_COLOR_ATTACHMENT0_EXT, GL_RGBA, TRANSMITTANCE_H, TRANSMITTANCE_W, 4, "c:\\SVN\\transm_gpu.xyz");
	float* tmp = getTextureData((GLenum)GL_COLOR_ATTACHMENT0_EXT, transmittanceTexture, GL_RGB, TRANSMITTANCE_W, TRANSMITTANCE_H, 3);
	disp->_scene.create2DOptixTexFromBuffer("trans_texture", tmp, TRANSMITTANCE_W, TRANSMITTANCE_H, 3, true);
	//cutSavePGMf( "c:\\SVN\\transm_gpu.ppm", tmp, TRANSMITTANCE_W, TRANSMITTANCE_H);
	delete [] tmp;
	//disp->_scene.create2DOptixTex("trans_texture", transmittanceTexture,  disp->_scene.textrans_texture);
#endif	

	// calculating mapping from [0,1) to non-uniform distribution of the phase functions: step 1
    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, SRTexture, 0);
    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, SMTexture, 0);
    unsigned int bufs[2] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
    glDrawBuffers(2, bufs);
    glViewport(0, 0, PHASE_DISTR_RES, 1);
    glUseProgram(phaseFuncProg);
    //glUniform1i(glGetUniformLocation(inscatter1Prog, "transmittanceSampler"), transmittanceUnit);
	drawQuad();
	float minray = 0.0, maxray = 0.0, minmie = 0.0, maxmie = 0.0;
	getMinMax((GLenum)GL_COLOR_ATTACHMENT0_EXT, PHASE_DISTR_RES, 1, 3, minray, maxray);
	getMinMax((GLenum)GL_COLOR_ATTACHMENT1_EXT, PHASE_DISTR_RES, 1, 3, minmie, maxmie);
	
	// store texture into disp structure that connects GLSL with Optix
	tmp = getTextureData((GLenum)GL_COLOR_ATTACHMENT0_EXT, SRTexture, GL_RGB, PHASE_DISTR_RES, 1, 3);
	disp->_scene.create2DOptixTexFromBuffer("SRPhaseFuncSampler", tmp, PHASE_DISTR_RES, 1, 3);
	//disp->store_var(tmp,PHASE_DISTR_RES*3,disp->_sr_phase_func);
	delete [] tmp;

	tmp = getTextureData((GLenum)GL_COLOR_ATTACHMENT1_EXT, SMTexture, GL_RGB, PHASE_DISTR_RES, 1, 3);
	disp->_scene.create2DOptixTexFromBuffer("SMPhaseFuncSampler", tmp, PHASE_DISTR_RES, 1, 3);
	//disp->store_var(tmp,PHASE_DISTR_RES*3,disp->_sm_phase_func);
	delete [] tmp;

	//saveTextureToXYZ((GLenum)GL_COLOR_ATTACHMENT0_EXT, PHASE_DISTR_RES, 1, 3, "c:\\temp\\SRTexture.XYZ", false);

	// calculating mapping from [0,1) to non-uniform distribution of the phase functions: step 2
    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, SRDensTexture, 0);
    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, SMDensTexture, 0);
    bufs[0] = GL_COLOR_ATTACHMENT0_EXT;
	bufs[1] = GL_COLOR_ATTACHMENT1_EXT;
    glDrawBuffers(2, bufs);
    glViewport(0, 0, PHASE_DISTR_RES, 1);
    glUseProgram(phaseDensFuncProg);
	glUniform1i(glGetUniformLocation(phaseDensFuncProg, "SRSampler"), SRUnit);
    glUniform1i(glGetUniformLocation(phaseDensFuncProg, "SMSampler"), SMUnit);
	glUniform1f(glGetUniformLocation(phaseDensFuncProg, "minray"), minray);
	glUniform1f(glGetUniformLocation(phaseDensFuncProg, "maxray"), maxray);
	glUniform1f(glGetUniformLocation(phaseDensFuncProg, "minmie"), minmie);
	glUniform1f(glGetUniformLocation(phaseDensFuncProg, "maxmie"), maxmie);
	drawQuad();

	// store texture into disp structure that connects GLSL with Optix
	tmp = getTextureData((GLenum)GL_COLOR_ATTACHMENT0_EXT, SRDensTexture, GL_RGB, PHASE_DISTR_RES, 1, 3);
	disp->_scene.create2DOptixTexFromBuffer("SRPhaseFuncIndSampler", tmp, PHASE_DISTR_RES, 1, 3);
	//disp->store_var(tmp,PHASE_DISTR_RES*3,disp->_sr_phase_func_ind);
	delete [] tmp;

	tmp = getTextureData((GLenum)GL_COLOR_ATTACHMENT1_EXT, SMDensTexture, GL_RGB, PHASE_DISTR_RES, 1, 3);
	disp->_scene.create2DOptixTexFromBuffer("SMPhaseFuncIndSampler", tmp, PHASE_DISTR_RES, 1, 3);
	//disp->store_var(tmp,PHASE_DISTR_RES*3,disp->_sm_phase_func_ind);
	delete [] tmp;

	//saveTextureToXYZ((GLenum)GL_COLOR_ATTACHMENT0_EXT, PHASE_DISTR_RES, 1, 3, "c:\\temp\\SRDensTexture.XYZ", false);


    // computes irradiance texture deltaE (line 2 in algorithm 4.1)
    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, deltaETexture, 0);
    glViewport(0, 0, SKY_W, SKY_H);
    glUseProgram(irradiance1Prog);
    glUniform1i(glGetUniformLocation(irradiance1Prog, "transmittanceSampler"), transmittanceUnit);
    drawQuad();

    // computes single scattering texture deltaS (line 3 in algorithm 4.1)
    // Rayleigh and Mie separated in deltaSR + deltaSM
    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, deltaSRTexture, 0);
    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, deltaSMTexture, 0);
    bufs[0] = GL_COLOR_ATTACHMENT0_EXT;
	bufs[1] = GL_COLOR_ATTACHMENT1_EXT;
    glDrawBuffers(2, bufs);
    glViewport(0, 0, RES_MU_S * RES_NU, RES_MU);
    glUseProgram(inscatter1Prog);
    glUniform1i(glGetUniformLocation(inscatter1Prog, "transmittanceSampler"), transmittanceUnit);
    for (int layer = 0; layer < RES_R; ++layer) {
        setLayer(inscatter1Prog, layer);
        drawQuad();
    }
	
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, 0, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

#ifdef TEX2OPTIX
	tmp = getTextureData3D(GL_COLOR_ATTACHMENT0_EXT, deltaSRTexture, RES_MU_S * RES_NU, RES_MU, RES_R, 3, GL_RGB);
	disp->_scene.create3DOptixTexFromBuffer("raySampler", tmp, RES_MU_S * RES_NU, RES_MU, RES_R, 3);
	delete [] tmp;

	tmp = getTextureData3D(GL_COLOR_ATTACHMENT1_EXT, deltaSMTexture, RES_MU_S * RES_NU, RES_MU, RES_R, 3, GL_RGB);
	disp->_scene.create3DOptixTexFromBuffer("mieSampler", tmp, RES_MU_S * RES_NU, RES_MU, RES_R, 3);
	delete [] tmp;
#endif

/*
	float* pTexBuffer = new float[RES_MU_S * RES_NU * RES_MU* RES_R*4];
	glBindTexture(GL_TEXTURE_3D, deltaSRTexture);
	glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, pTexBuffer);
	//glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, RES_MU_S * RES_NU, RES_MU, RES_R, 0, GL_RGBA, GL_UNSIGNED_BYTE, mTexData);
	
	glDeleteTextures(1, &deltaSRTexture);
	glActiveTexture(GL_TEXTURE0 + deltaSRUnit);
    glGenTextures(1, &deltaSRTexture);
    glBindTexture(GL_TEXTURE_3D, deltaSRTexture);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, RES_MU_S * RES_NU, RES_MU, RES_R, 0, GL_RGBA, GL_FLOAT, pTexBuffer);
	//disp->_scene.create3DOptixTex("raySampler", deltaSRTexture , disp->_scene.texraySampler);
	delete [] pTexBuffer;
	//disp->_scene.create3DOptixTex("mieSampler", deltaSMTexture , disp->_scene.texmieSampler);
*/
    

    // copies deltaE into irradiance texture E (line 4 in algorithm 4.1)
    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, irradianceTexture, 0);
    glViewport(0, 0, SKY_W, SKY_H);
    glUseProgram(copyIrradianceProg);
    glUniform1f(glGetUniformLocation(copyIrradianceProg, "k"), 0.0);
    glUniform1i(glGetUniformLocation(copyIrradianceProg, "deltaESampler"), deltaEUnit);
    drawQuad();

    // copies deltaS into inscatter texture S (line 5 in algorithm 4.1)

    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, inscatterTexture, 0);
    glViewport(0, 0, RES_MU_S * RES_NU, RES_MU);
    glUseProgram(copyInscatter1Prog);
    glUniform1i(glGetUniformLocation(copyInscatter1Prog, "deltaSRSampler"), deltaSRUnit);
    glUniform1i(glGetUniformLocation(copyInscatter1Prog, "deltaSMSampler"), deltaSMUnit);
    char tempfn[MAX_PATH];
	for (int layer = 0; layer < RES_R; ++layer) {
        setLayer(copyInscatter1Prog, layer);
		drawQuad();
		if(layer == 0) 
			sprintf(tempfn,"c:\\temp\\transmittance_%d.XYZ",layer);
		//saveTextureToXYZ((GLenum)GL_COLOR_ATTACHMENT0_EXT, RES_MU_S * RES_NU, RES_MU, 3, tempfn, true);
		//glFramebufferTexture3DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, inscatterTexture, 0, 0);
#ifdef DEBUGGING
		//convert textures to 2D arrays
		if(layer == 0){
			float* pTexBuffer = new float[RES_MU_S * RES_NU *RES_MU*4];
			glReadBuffer((GLenum)GL_COLOR_ATTACHMENT0_EXT);
			glReadPixels(0, 0, RES_MU_S * RES_NU, RES_MU, GL_RGBA, GL_FLOAT, pTexBuffer);
			//glCopyTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, 0, 0, RES_MU_S * RES_NU, RES_MU, 0 );
			//glGetTexImage( GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pTexBuffer ); 
			
			////for(int i=0; i<RES_MU_S * RES_NU; i++){
			////	for(int j=0; j<RES_MU; j++){
			////		for(int k=0; k<4; k++){
			////			printf("[%d][%d]=%f\n",i,j,pTexBuffer[i*RES_MU*4 + j*4+k]);
			////		}
			////	}
			////}
			delete [] pTexBuffer;

	        // Texture format
			//gltWriteTGA("test.tga");
			//pBytes = gltLoadTGA("test.tga", &iWidth, &iHeight, &iComponents, &eFormat);
		}
#endif //DEBUGGING
    }
 

	//saveTextureToXYZ3D(GL_COLOR_ATTACHMENT0_EXT, inscatterTexture, RES_MU_S * RES_NU, RES_MU, RES_R, 4, "C:\\Temp\\pas\\pas", true);

	
#ifdef OLD_PHOTON_MAPPING
	for (int layer = 0; layer < MAX_PATH_LENGTH; ++layer) {
		if(layer == 0){
			InitializePPMData();
		}
		GLenum PhotonRayTraceBuffers[] = {GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT};
		glGenFramebuffersEXT(1, &PhotonRayTraceSurface);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, PhotonRayTraceSurface);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, photonPositionTexture, 0);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, photonFluxTexture, 0);
		//glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_TEXTURE_2D, randomTexture, 0);
		glDrawBuffers(2, PhotonRayTraceBuffers);
		//glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, PhotonRayTraceSurface);
		glViewport(0, 0, RES_PHOTONS_X, RES_PHOTONS_Y);
		glUseProgram(photonProg);

		//glUniform1i(glGetUniformLocation(photonProg, "transmittanceSampler"), transmittanceUnit);
		//glUniform1i(glGetUniformLocation(photonProg, "UseEyeRays"), 0);
		//glUniform4fv(glGetUniformLocation(photonProg, "BufInfo"), 1, BufInfo);
		glUniform1i(glGetUniformLocation(photonProg, "RandomTexture"), 9);
		//glUniform1i(glGetUniformLocation(photonProg, "PathLength"), 0);
		glUniform1i(glGetUniformLocation(photonProg, "PathLength"), layer);//layer % MAX_PATH_LENGTH);
		glUniform1i(glGetUniformLocation(photonProg, "Length"), RES_PHOTONS_Y);
		glUniform1i(glGetUniformLocation(photonProg, "Width"), RES_PHOTONS_X);
		glUniform1f(glGetUniformLocation(photonProg, "MaxPathLength"), MAX_PATH_LENGTH);
		glUniform3f(glGetUniformLocation(drawProg, "s"), s.x, s.y, s.z);

		drawQuad();

		if(layer == 0){
			//saveTextureToXYZ((GLenum)GL_COLOR_ATTACHMENT0_EXT, RES_PHOTONS_X, RES_PHOTONS_Y, 3, "c:\\temp\\random.XYZ");
			saveTextureToXYZ((GLenum)GL_COLOR_ATTACHMENT0_EXT, RES_PHOTONS_X, RES_PHOTONS_Y, 3, "c:\\temp\\position.XYZ");
			saveTextureToXYZ((GLenum)GL_COLOR_ATTACHMENT1_EXT, RES_PHOTONS_X, RES_PHOTONS_Y, 3, "c:\\temp\\direction.XYZ");
			SavePhotonMap((GLenum)GL_COLOR_ATTACHMENT0_EXT, (GLenum)GL_COLOR_ATTACHMENT1_EXT, RES_PHOTONS_X, RES_PHOTONS_Y, 3, "c:\\temp\\posdir.bin");
		}
	}
#endif //OLD_PHOTON_MAPPING
	
#ifdef NEYRET_MULT_SCATTERING
    // loop for each scattering order (line 6 in algorithm 4.1)
    for (int order = 2; order <= 6; ++order) {
		
        // computes deltaJ (line 7 in algorithm 4.1)
        glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, deltaJTexture, 0);
        glViewport(0, 0, RES_MU_S * RES_NU, RES_MU);
        
		glUseProgram(jProg);
        
		glUniform1f(glGetUniformLocation(jProg, "first"), order == 2 ? 1.0 : 0.0);
        glUniform1i(glGetUniformLocation(jProg, "transmittanceSampler"), transmittanceUnit);
        glUniform1i(glGetUniformLocation(jProg, "deltaESampler"), deltaEUnit);
        glUniform1i(glGetUniformLocation(jProg, "deltaSRSampler"), deltaSRUnit);
        glUniform1i(glGetUniformLocation(jProg, "deltaSMSampler"), deltaSMUnit);
        
		for (int layer = 0; layer < RES_R; ++layer) {
            setLayer(jProg, layer);
            drawQuad();

			if(layer == 1){
				//saveTextureToXYZ((GLenum)GL_COLOR_ATTACHMENT0_EXT, RES_MU_S * RES_NU, RES_MU, 3, "c:\\temp\\texbuf.XYZ");
			}
        }
	    
        // computes deltaE (line 8 in algorithm 4.1)
        glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, deltaETexture, 0);
        glViewport(0, 0, SKY_W, SKY_H);
        glUseProgram(irradianceNProg);
        glUniform1f(glGetUniformLocation(irradianceNProg, "first"), order == 2 ? 1.0 : 0.0);
        glUniform1i(glGetUniformLocation(irradianceNProg, "transmittanceSampler"), transmittanceUnit);
        glUniform1i(glGetUniformLocation(irradianceNProg, "deltaSRSampler"), deltaSRUnit);
        glUniform1i(glGetUniformLocation(irradianceNProg, "deltaSMSampler"), deltaSMUnit);
        drawQuad();

		
        // computes deltaS (line 9 in algorithm 4.1)
        glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, deltaSRTexture, 0);
        glViewport(0, 0, RES_MU_S * RES_NU, RES_MU);
        glUseProgram(inscatterNProg);
        glUniform1f(glGetUniformLocation(inscatterNProg, "first"), order == 2 ? 1.0 : 0.0);
        glUniform1i(glGetUniformLocation(inscatterNProg, "transmittanceSampler"), transmittanceUnit);
        glUniform1i(glGetUniformLocation(inscatterNProg, "deltaJSampler"), deltaJUnit);
        for (int layer = 0; layer < RES_R; ++layer) {
            setLayer(inscatterNProg, layer);
            drawQuad();
        }

        glEnable(GL_BLEND);
        glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);

        // adds deltaE into irradiance texture E (line 10 in algorithm 4.1)
        glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, irradianceTexture, 0);
        glViewport(0, 0, SKY_W, SKY_H);
        glUseProgram(copyIrradianceProg);
        glUniform1f(glGetUniformLocation(copyIrradianceProg, "k"), 1.0);
        glUniform1i(glGetUniformLocation(copyIrradianceProg, "deltaESampler"), deltaEUnit);
        drawQuad();

        // adds deltaS into inscatter texture S (line 11 in algorithm 4.1)
        glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, inscatterTexture, 0);
        glViewport(0, 0, RES_MU_S * RES_NU, RES_MU);
        glUseProgram(copyInscatterNProg);
        glUniform1i(glGetUniformLocation(copyInscatterNProg, "deltaSSampler"), deltaSRUnit);

        for (int layer = 0; layer < RES_R; ++layer) {
            setLayer(copyInscatterNProg, layer);
            drawQuad();
        }
		
        glDisable(GL_BLEND);		
    }

#endif //NEYRET_MULT_SCATTERING
    saveTextureToXYZ3D(GL_COLOR_ATTACHMENT0_EXT, inscatterTexture, RES_MU_S * RES_NU, RES_MU, RES_R, 4, "C:\\Temp\\pas\\pas", true);
#ifdef TEX2OPTIX
	tmp = getTextureData3D(GL_COLOR_ATTACHMENT0_EXT, inscatterTexture, RES_MU_S * RES_NU, RES_MU, RES_R, 4, GL_RGBA);
	disp->_scene.create3DOptixTexFromBuffer("inscatterSampler", tmp, RES_MU_S * RES_NU, RES_MU, RES_R, 4);
	disp->_scene.create3DOptixTexFromBuffer("inscatterPhotonSampler", tmp, RES_MU_S * RES_NU, RES_MU, RES_R, 4);
	delete [] tmp;
#endif
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glFinish();
//#ifdef FBO_ENABLED
//	test_renderer();
//#endif

	//glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    //glFinish();
	
    cout << "ready." << endl;
    glUseProgram(drawProg);
}

void recompute()
{
    glDeleteTextures(1, &transmittanceTexture);
    glDeleteTextures(1, &irradianceTexture);
    glDeleteTextures(1, &inscatterTexture);
    glDeleteTextures(1, &deltaETexture);
    glDeleteTextures(1, &deltaSRTexture);
    glDeleteTextures(1, &deltaSMTexture);
    glDeleteTextures(1, &deltaJTexture);
	//glDeleteTextures(1, &photonPositionTexture);
	//glDeleteTextures(1, &randomTexture);
	//glDeleteTextures(1, &photonFluxTexture);

	glDeleteTextures(1, &SRTexture);
	glDeleteTextures(1, &SMTexture);
	glDeleteTextures(1, &SRDensTexture);
	glDeleteTextures(1, &SMDensTexture);
    
	glDeleteProgram(transmittanceProg);
    glDeleteProgram(irradiance1Prog);
    glDeleteProgram(inscatter1Prog);
    glDeleteProgram(copyIrradianceProg);
    glDeleteProgram(copyInscatter1Prog);
    glDeleteProgram(jProg);
    glDeleteProgram(irradianceNProg);
    glDeleteProgram(inscatterNProg);
    glDeleteProgram(copyInscatterNProg);
    glDeleteProgram(drawProg);
	glDeleteProgram(photonProg);
	glDeleteProgram(phaseFuncProg);
	glDeleteProgram(phaseDensFuncProg);
	
	//glDeleteFramebuffersEXT(1, &PhotonRayTraceSurface);
    glDeleteFramebuffersEXT(1, &fbo);
    precompute();
}

// ----------------------------------------------------------------------------
// RENDERING
// ----------------------------------------------------------------------------

int width, height;
int oldx, oldy;
int oldsunx, oldsuny;
int move;

//vec3f s(0.0, -1.0, 0.0); // sun position

double lon = 0.0;
double lat = 0.0;
double theta = 0.0;
double phi = 0.0;
double d = Rg;
double cz_y = 1.0;
vec3d position;
mat4d view;

float test_camera = 0.0;

double exposure = 0.4;
bool bUpdatedMousePos = false;

void init_pas()
{
	lon = 0.0;
	lat = 0.0;
	phi = 0.0;
	theta = 0.0;
    //theta = max(0.0, min(9.0*M_PI/10.0, theta));
	theta = max(0.0, min(M_PI, theta));
    updateView();
	disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
	bUpdateRendering = true;
}

void updateView()
{
	double co = cos(lon);
	double so = sin(lon);
	double ca = cos(lat);
	double sa = sin(lat);
	//vec3d po = vec3d(co*sa, so*sa, ca) * Rg;
	vec3d po = vec3d(co*ca, so*ca, sa) * (Rg);
	vec3d px = vec3d(1.0,0.0,0.0);//vec3d(-so, co, 0);
    vec3d py = vec3d(0.0,1.0,0.0);//vec3d(-co*sa, -so*sa, ca);
    vec3d pz = vec3d(0.0,0.0,1.0);//vec3d(co*ca, so*ca, sa);
	px = vec3d(-so, co, 0);
    py = vec3d(-co*sa, -so*sa, ca);
    pz = vec3d(co*ca, so*ca, sa);

    double ct = cos(theta);
    double st = sin(theta);
    double cp = cos(phi);
    double sp = sin(phi);
    vec3d cx = px * cp + py * sp;
    vec3d cy = -px * sp*ct + py * cp*ct + pz * st;
    vec3d cz = px * sp*st - py * cp*st + pz * ct;
	
	cz.y *= cz_y;
	cz.z *= cz_y;
	/*
	if(test_camera > 0.0)
	{
		float c_rot = cos(test_camera);
		float s_rot = sin(test_camera);
		vec3d cz_new;
		cz_new.x = c_rot * cz.x - s_rot * cz.y;
		cz_new.y = s_rot * cz.x +c_rot * cz.y;
		cz_new.z = cz.z;
		cz = cz_new;
		position = po + cz*d;

	}
	else
	*/
	position = po + cz * d;

    if (position.length() < Rg + 0.01) {
    	position.normalize(Rg + 0.01);
    }

	//cx = px * cp + py * sp;
    //cy = -px * sp*ct + py * cp*ct + pz * st;
    //cz = px * sp*st - py * cp*st + pz * ct;
    
	view = mat4d(cx.x, cx.y, cx.z, 0,
            cy.x, cy.y, cy.z, 0,
            cz.x, cz.y, cz.z, 0,
            0, 0, 0, 1);
    view = view * mat4d::translate(-position);

	//vec3<double> norm_vec = cz;
	//norm_vec.dotproduct(position);
	//view = view * mat4d::rotate(M_PI/4,norm_vec);

	

	bUpdatedMousePos = true;

	//cout << "lat: " << lat << ", lon: " << lon << ", theta: " << theta << ", phi: " << phi << endl;
	bUpdateRendering = true;
}

void InitializePPMData()
{
	// emission & local photon count
	glActiveTexture(GL_TEXTURE8);
	for (int j = 0; j < BufferSize; j++)
	{
		for (int i = 0; i < BufferSize; i++)
		{
			TempData[(i + j * BufferSize) * 4 + 0] = 0.0;
			TempData[(i + j * BufferSize) * 4 + 1] = 0.0;
			TempData[(i + j * BufferSize) * 4 + 2] = 0.0;
			TempData[(i + j * BufferSize) * 4 + 3] = 0.0;
		}
	}
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, BufferSize, BufferSize, GL_RGBA, GL_FLOAT, TempData);

	// accumulated (unnormalized) flux & radius
	glActiveTexture(GL_TEXTURE10);
	for (int j = 0; j < BufferSize; j++)
	{
		for (int i = 0; i < BufferSize; i++)
		{
			TempData[(i + j * BufferSize) * 4 + 0] = 0.0;
			TempData[(i + j * BufferSize) * 4 + 1] = 0.0;
			TempData[(i + j * BufferSize) * 4 + 2] = 0.0;
			TempData[(i + j * BufferSize) * 4 + 3] = InitialRadius;
		}
	}
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, BufferSize, BufferSize, GL_RGBA, GL_FLOAT, TempData);

	glActiveTexture(GL_TEXTURE9);
	// generate random numbers
	for (int j = 0; j < BufferSize; j++)
	{
		for (int i = 0; i < BufferSize; i++)
		{
			TempData[(i + j * BufferSize) * 4 + 0] = XORShift::frand()* 4194304.0;
			TempData[(i + j * BufferSize) * 4 + 1] = XORShift::frand()* 4194304.0;
			TempData[(i + j * BufferSize) * 4 + 2] = XORShift::frand()* 4194304.0;
			TempData[(i + j * BufferSize) * 4 + 3] = XORShift::frand()* 4194304.0;
/*
			if(j == BufferSize -1 && i == BufferSize -1)
				printf("%f,%f,%f\n",TempData[(i + j * BufferSize) * 4 + 0],
								TempData[(i + j * BufferSize) * 4 + 1],
								TempData[(i + j * BufferSize) * 4 + 2]);
*/
		}
	}
	
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, BufferSize, BufferSize, GL_RGBA, GL_FLOAT, TempData);

	TotalPhotonNum = 0.0;
}


#define  EARTH_SPHERE

//GLfloat cameraPos[] = { 50.0f, 50.0f, 150.0f, 1.0f};
float diag = nv::length(isg->modelBBMax - isg->modelBBMin);
//GLfloat cameraPos[] = {  0.0, -diag, 0.0};
//GLfloat cameraPos[] = { 200.0f, -850.0f, -100.0, 1.0f};
//GLfloat cameraPos[] = { 200.0f, 400.0f, -300.0, 1.0f};
GLfloat cameraPos[] = { 0.0f, 400.0f, 0.0, 1.0f};
GLdouble cameraZoom = 1.0;//0.4;
GLfloat lightPos[] = { 140.0f, 250.0f, 140.0f, 1.0f};
GLfloat lightRotation = 30.0f;
GLint lightPosLoc;
GLint tess = 75;


// Called to draw scene objects
void DrawModels(GLboolean drawBasePlane)
{
    if (drawBasePlane)
    {
        // Draw plane that the objects rest on
        glColor3f(0.0f, 0.0f, 0.90f); // Blue
        glNormal3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_QUADS);
            glVertex3f(-100.0f, -25.0f, -100.0f);
            glVertex3f(-100.0f, -25.0f, 100.0f);
            glVertex3f(100.0f,  -25.0f, 100.0f);
            glVertex3f(100.0f,  -25.0f, -100.0f);
        glEnd();
    }

    // Draw red cube
    glColor3f(1.0f, 0.0f, 0.0f);
    glutSolidCube(48.0f);

    // Draw green sphere
    glColor3f(0.0f, 1.0f, 0.0f);
    glPushMatrix();
    glTranslatef(-60.0f, 0.0f, 0.0f);
    glutSolidSphere(25.0f, 50, 50);
    glPopMatrix();

    // Draw yellow cone
    glColor3f(1.0f, 1.0f, 0.0f);
    glPushMatrix();
    glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
    glTranslatef(60.0f, 0.0f, -24.0f);
    glutSolidCone(25.0f, 50.0f, 50, 50);
    glPopMatrix();

    // Draw magenta torus
    glColor3f(1.0f, 0.0f, 1.0f);
    glPushMatrix();
    glTranslatef(0.0f, 0.0f, 60.0f);
    glutSolidTorus(8.0f, 16.0f, 50, 50);
    glPopMatrix();

    // Draw cyan octahedron
    glColor3f(0.0f, 1.0f, 1.0f);
    glPushMatrix();
    glTranslatef(0.0f, 0.0f, -60.0f);
    glScalef(25.0f, 25.0f, 25.0f);
    glutSolidOctahedron();
    glPopMatrix();
	CHECK_ERRORS();
}

#define EARTH_SPHERE_TEST

void draw_model_scene()
{
	float h = position.length() - Rg;
    float vfov = 2.0 * atan(float(height) / float(width) * tan(80.0 / 180 * M_PI / 2.0)) / M_PI * 180;
    mat4f proj = mat4f::perspectiveProjection(vfov, float(width) / float(height), 0.1 * h, 1e5 * h);

    mat4f iproj = proj.inverse();
    mat4d iview = view.inverse();
    vec3d c = iview * vec3d(0.0, 0.0, 0.0); //view position

    mat4f iviewf = mat4f(iview[0][0], iview[0][1], iview[0][2], iview[0][3],
        iview[1][0], iview[1][1], iview[1][2], iview[1][3],
        iview[2][0], iview[2][1], iview[2][2], iview[2][3],
        iview[3][0], iview[3][1], iview[3][2], iview[3][3]);
	mat4f modelf = mat4f(	1.0, 0.0, 0.0, -isg->modelBBCenter.x,
							0.0, 1.0, 0.0, -isg->modelBBCenter.y+Rg,
							0.0, 0.0, 1.0, -isg->modelBBCenter.z,
							0.0, 0.0, 0.0, 1.0);
	mat4f imodelf = modelf.inverse();

#ifdef EARTH_SPHERE_TEST
	glEnable(GL_DEPTH_TEST);
	// Track camera angle
	//glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	//CHECK_ERRORS();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (width > height)
    {
        GLdouble ar = (GLdouble)width / (GLdouble)height;
        //glFrustum(-ar * cameraZoom, ar * cameraZoom, -cameraZoom, cameraZoom, 1.0, 1000.0);
    }
    else
    {
        GLdouble ar = (GLdouble)height / (GLdouble)width;
        //glFrustum(-cameraZoom, cameraZoom, -ar * cameraZoom, ar * cameraZoom, 1.0, 1000.0);
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //gluLookAt(cameraPos[0], cameraPos[1], cameraPos[2], 
    //          0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	//gluLookAt(c.x, c.y, c.z, 
    //          0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	// draw final colors
    
//#ifdef  EARTH_SPHERE
	glViewport(0,0, fboWidth, fboHeight);
//#else	
//	glViewport(0,0, width, height);
//#endif
	glUseProgram(drawGeometryProg);
	float ftest = 1.0;
	glUniform1f(glGetUniformLocation(drawGeometryProg, "flickerFactor"), ftest);
	glUniform3f(glGetUniformLocation(drawProg, "c"), c.x, c.y, c.z);
    glUniform3f(glGetUniformLocation(drawProg, "s"), s.x, s.y, s.z);
    glUniformMatrix4fv(glGetUniformLocation(drawProg, "projInverse"), 1, true, iproj.coefficients());
    glUniformMatrix4fv(glGetUniformLocation(drawProg, "viewInverse"), 1, true, iviewf.coefficients());
	glUniformMatrix4fv(glGetUniformLocation(drawProg, "modelInverse"), 1, true, imodelf.coefficients());
	glColor3f(1.0,0.0,1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    

//#if !defined(EARTH_SPHERE)
	//glPushMatrix();
	//glTranslatef( -isg->modelBBCenter.x,
    //              -isg->modelBBCenter.y,
    //              -isg->modelBBCenter.z );
    //glUniform1f(starIntensityLoc, abs(angleIncrement));
//#endif
    // Draw objects in the scene
	glColor4f(0.0, 0.0, 1.0, 0.9);
	glutSolidSphere(Rg, 40, 40);
	//isg->draw_model(emptyPlaceholder, true);

    //DrawModels(true);
//#if !defined(EARTH_SPHERE)
	//glPopMatrix();
//#endif
    glDisable(GL_DEPTH_TEST);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	//glDrawBuffer(GL_BACK);
	//isg->draw_model(emptyPlaceholder, true);
	glUseProgram(0);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	disp->_scene.optix_display(make_float3(c.x, c.y,c.z), make_float3(s.x,s.y,s.z), iproj.coefficients(), iviewf.coefficients());
#else
	
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glUseProgram(0);
	

	disp->_scene.optix_display(make_float3(c.x, c.y,c.z), make_float3(s.x,s.y,s.z), iproj.coefficients(), iviewf.coefficients());
#endif
}

#define FBO_ENABLED

void renderModel()
{
	#ifdef EARTH_SPHERE_TEST
	//bUpdateRendering = false;
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, geomRenderTexture, 0);
	#endif
	//glViewport(0,0,fboWidth, fboHeight);
	//glUseProgram(drawGeometryProg);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	draw_model_scene();
	glUseProgram(0);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}


void renderEarth()
{
	bUpdateRendering = false;
	// computes transmittance texture T (line 1 in algorithm 4.1)
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
    glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, testRenderTexture, 0);
	float h = position.length() - Rg;
    float vfov = 2.0 * atan(float(height) / float(width) * tan(80.0 / 180 * M_PI / 2.0)) / M_PI * 180;
    mat4f proj = mat4f::perspectiveProjection(vfov, float(width) / float(height), 0.1 * h, 1e5 * h);

    mat4f iproj = proj.inverse();
    mat4d iview = view.inverse();
    vec3d c = iview * vec3d(0.0, 0.0, 0.0); //view position

    mat4f iviewf = mat4f(iview[0][0], iview[0][1], iview[0][2], iview[0][3],
        iview[1][0], iview[1][1], iview[1][2], iview[1][3],
        iview[2][0], iview[2][1], iview[2][2], iview[2][3],
        iview[3][0], iview[3][1], iview[3][2], iview[3][3]);

	if(bUpdatedMousePos){
		/*
		std::cerr << "vfov = " << vfov << std::endl;
		std::cerr << "view: " << std::endl;
		view.printMatrix();
		std::cerr << "proj: " << std::endl;
		proj.printMatrix();
		*/
		bUpdatedMousePos = false;
	}

    glViewport(0,0,fboWidth, fboHeight);
    glUseProgram(drawProg);
    glUniform3f(glGetUniformLocation(drawProg, "c"), c.x, c.y, c.z);
    glUniform3f(glGetUniformLocation(drawProg, "s"), s.x, s.y, s.z);
    glUniformMatrix4fv(glGetUniformLocation(drawProg, "projInverse"), 1, true, iproj.coefficients());
    glUniformMatrix4fv(glGetUniformLocation(drawProg, "viewInverse"), 1, true, iviewf.coefficients());
    glUniform1f(glGetUniformLocation(drawProg, "exposure"), exposure);
	glUniform1i(glGetUniformLocation(drawProg, "photonFluxSampler"), photonFluxUnit);

	glUniform1i(glGetUniformLocation(drawProg, "SRDensSampler"), SRDensUnit);
    glUniform1i(glGetUniformLocation(drawProg, "SMDensSampler"), SMDensUnit);
	glUniform1i(glGetUniformLocation(drawProg, "SRSampler"), SRUnit);
    glUniform1i(glGetUniformLocation(drawProg, "SMSampler"), SMUnit);
    drawQuad();
	glUseProgram(0);
}



void redisplayFunc_pas()
{
#ifdef EARTH_SPHERE
	//glEnable(GL_DEPTH_TEST);
	//renderModel();
	//optix_display();
	//glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo1);
	

	//glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	//test_renderer();
	
	
#ifdef FBO_ENABLED	
	
	//glDisable(GL_DEPTH_TEST);
	/*
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	*/
	if(bUpdateRendering) {
		//glEnable(GL_BLEND);
#if !defined(DEMO_MODE)
		renderEarth();
#endif
		renderModel();
		
		//glDisable(GL_BLEND);
	}

/*	
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
*/

	//glEnable(GL_TEXTURE_2D);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glViewport(0, 0, width, height);
	glUseProgram(combineProg);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUniform1i(glGetUniformLocation(combineProg, "sampler0"), testRenderTextureUnit);
#ifdef EARTH_SPHERE_TEST
	//glUniform1i(glGetUniformLocation(combineProg, "sampler1"), geomRenderTextureUnit);
//#else
	glUniform1i(glGetUniformLocation(combineProg, "sampler1"), disp->_scene.getTexUnit());
#endif	
	//glBindTexture(GL_TEXTURE_2D, testRenderTexture);
    //glUniform1i(glGetUniformLocation(combineProg, "sampler1"), renderTextureIDUnit+1);

	drawQuad();
	glUseProgram(0);
	//glBindTexture(GL_TEXTURE_2D, 0);
#else
	//foo();

   
	test_renderer();
#endif
	
	
#else
	//glEnable(GL_DEPTH_TEST);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glUseProgram(0);
	//glUseProgram(drawGeometryProg);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//drawQuad();
	//draw_model_scene();
    //SimpleTest();
	DrawModels(true);
	//disp->_scene.optix_display(make_float3(c.x, c.y,c.z), make_float3(s.x,s.y,s.z), iproj.coefficients(), iviewf.coefficients());
#endif
	glutSwapBuffers();
	FrameCount++;
}


// ----------------------------------------------------------------------------
// USER INTERFACE
// ----------------------------------------------------------------------------

void reshapeFunc_pas(int x, int y)
{
#ifdef EARTH_SPHERE
    width = x;
    height = y;
	GLint origWidth = fboWidth;
    GLint origHeight = fboHeight;
	fboWidth = x;
    fboHeight = y;
    glViewport(0, 0, x, y);
/*
	if (fboWidth > maxTexSize)
    {
        fboWidth = maxTexSize;
    }

    if (fboHeight > maxTexSize)
    {
        fboHeight = maxTexSize;
    }
*/
    if ((origWidth != fboWidth) || (origHeight != fboHeight))
    {
        //glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32, fboWidth, fboHeight);

        glBindTexture(GL_TEXTURE_2D, testRenderTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, fboWidth, fboHeight, 0, GL_RGB, GL_FLOAT, 0);

		glBindTexture(GL_TEXTURE_2D, geomRenderTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, fboWidth, fboHeight, 0, GL_RGB, GL_FLOAT, 0);
	}

    glutPostRedisplay();
#else
	
	// We don't want to allocate 0 memory for the PBOs
	width = x == 0 ? 1 : x; 
	height = y == 0 ? 1 : y; 

	isg->rasterWidth = width;
	isg->rasterHeight = height;

	isg->traceWidth = isg->rasterWidth >> isg->logShadowSamplingRate;
	isg->traceHeight = isg->rasterHeight >> isg->logShadowSamplingRate;

	isg->manipulator.reshape(width,height);

	/*
	float aspect = (float)width/(float)height;
	float diag = nv::length(isg->modelBBMax - isg->modelBBMin);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, aspect, diag*.01, diag*Rg);//100);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	try {
	// resize PBOs
	isg->rtShadowBuffer->unregisterGLBuffer();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, isg->shadowMapPBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, isg->traceWidth*isg->traceHeight*sizeof(float), 0, GL_STREAM_READ);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	isg->rtShadowBuffer->registerGLBuffer();

	// resize FBOs
	isg->rtWorldSpaceTexture->unregisterGLTexture();
	glBindTexture(GL_TEXTURE_2D, isg->worldSpaceFBO->GetAttachedId( GL_COLOR_ATTACHMENT0_EXT ) );
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, isg->traceWidth, isg->traceHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	isg->rtWorldSpaceTexture->registerGLTexture();
	} catch ( optix::Exception& e ) {
	sutilReportError( e.getErrorString().c_str() );
	exit(-1);
	}

	glBindTexture(GL_TEXTURE_2D, isg->worldSpaceFBO->GetAttachedId( GL_DEPTH_ATTACHMENT_EXT ) );
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8_EXT, isg->traceWidth, isg->traceHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);

	// resize shadow map texture
	glBindTexture(GL_TEXTURE_2D, isg->shadowMapTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, isg->traceWidth, isg->traceHeight, 0, GL_LUMINANCE, GL_FLOAT, 0);

	glBindTexture(GL_TEXTURE_2D, 0);

	// resize rt buffers
	isg->rtShadowBuffer->setSize(isg->traceWidth,isg->traceHeight);

	glutPostRedisplay();

	*/
	/*

	width = x;
    height = y;
    GLfloat fAspect;

    // Prevent a divide by zero
    if(height == 0)
        height = 1;

    // Set Viewport to window dimensions
    glViewport(0, 0, width, height);

    // Reset coordinate system
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    fAspect = (GLfloat)width / (GLfloat)height;
    gluPerspective(35.0f, fAspect, 1.0f, 1000.0f);
     
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -400.0f);
	*/
#endif   
	GLfloat fAspect;
	// Prevent a divide by zero
    if(height == 0)
        height = 1;

    // Set Viewport to window dimensions
    glViewport(0, 0, width, height);

    // Reset coordinate system
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    //fAspect = (GLfloat)width / (GLfloat)height;
    //gluPerspective(35.0f, fAspect, 1.0f, 1000.0f);
     
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //glTranslatef(0.0f, 0.0f, -400.0f);
}

void mouseClickFunc_pas(int button, int state, int x, int y)
{
    oldx = x;
    oldy = y;
    int modifiers = glutGetModifiers();
    bool ctrl = (modifiers & GLUT_ACTIVE_CTRL) != 0;
    bool shift = (modifiers & GLUT_ACTIVE_SHIFT) != 0;
    if (ctrl) {
    	move = 0;
    } else if (shift) {
        move = 1;
    } else {
		oldsunx = x;
		oldsuny = y;
    	move = 2;
    }
	glutPostRedisplay();
}

void mouseMotionFunc_pas(int x, int y)
{
    if (move == 0) {
		double factor = position.length() - Rg;
		factor = (disp->_scene.isCtrlFactorEnabled()) ? factor / Rg : 1.0;
		if(disp->_scene.isThetaOnly()){
			theta += (oldy - y) / 1500.0*factor;
		}
		else{
			phi += (oldx - x) / 500.0;
    		theta += (oldy - y) / 500.0*factor;
		}
        //theta = max(0.0, min(9.0*M_PI/10.0, theta));
		theta = max(0.0, min(M_PI, theta));
        updateView();
        oldx = x;
        oldy = y;
		//disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		disp->_scene.signalCameraChanged();
		bUpdateRendering = true;
    } else if (move == 1) {
    	double factor = position.length() - Rg;
    	factor = factor / Rg;
    	lon += (oldx - x) / 400.0 * factor;
    	lat -= (oldy - y) / 400.0 * factor;
        lat = max(-M_PI / 2.0, min(M_PI / 2.0, lat));
        updateView();
        oldx = x;
        oldy = y;
		//disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		disp->_scene.signalCameraChanged();
		bUpdateRendering = true;
    } else if (move == 2) {
		if(disp->_scene.isSunControlActivated()){
			// converting cartesian coord in latitude and longitude 
			float vangle = 0.0;
			if(disp->_scene.isSingle()){
				vangle = asin(s.z);	// latitude  (horizontal lines)	---> spheres with changing radii
				vangle += (oldy - y) / 180.0 * M_PI / 4;
			}

			float hangle = 0.0;//atan2(s.x, s.y); // longitude   (vertical lines) ---> identical radii
			//if(disp->_scene.isSingle()){
				hangle = atan2(s.y, s.x); // longitude   (vertical lines) ---> identical radii
				hangle += (oldx - x) / 180.0 * M_PI / 4;
			//}
    		// convertin latitude and longitude into cartesian coord
			s.x = cos(vangle) * cos(hangle);
    		s.y = cos(vangle) * sin(hangle);
    		s.z = sin(vangle);
			//s.x = sin(vangle) * cos(hangle);
    		//s.y = sin(vangle) * sin(hangle);
    		//s.z = cos(vangle);
#if !defined(DEMO_MODE)
			cout << "s.x: " << s.x << ", s.y: " << s.y << ", s.z: " << s.z << endl;
#endif
			disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
			oldx = x;
			oldy = y;
			bUpdateRendering = true;

		}
    }
	glutPostRedisplay();
}

void specialKeyFunc_pas(int c, int x, int y)
{
    switch (c) {
    case GLUT_KEY_PAGE_UP:
    	//d = d * 1.005;
        d += 2;
		updateView();
        break;
    case GLUT_KEY_PAGE_DOWN:
    	d -= 2;
		//d = d / 1.005;
        updateView();
        break;
    case GLUT_KEY_F5:
	    recompute();
        glViewport(0, 0, width, height);
        break;
	}
}

void keyboardFunc_pas(unsigned char c, int x, int y)
{
    if (c == 27) {
        ::exit(0);
		} else if (c == 'K') {
		//lon = 0.0;
		//lat = 0.0;
		//phi = 0.0;
    	//theta = 0.0;
		cout << "Enter  s.x: ";
		cin >> s.x;
		cout << "Enter  s.y ";
		cin >> s.y;
		cout << "Enter  s.z ";
		cin >> s.z;
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == 'k') {
		//lon = 0.0;
		//lat = 0.0;
		//phi = 0.0;
    	//theta = 0.0;
		cout << "Enter  lon: ";
		cin >> lon;
		cout << "Enter  lat (-M_PI / 2.0 ... M_PI / 2.0): ";
		cin >> lat;
		cout << "Enter  phi: ";
		cin >> phi;
		cout << "Enter  theta (0.0 ... M_PI): ";
		cin >> theta;
		theta = max(0.0, min(M_PI, theta));
		lat = max(-M_PI / 2.0, min(M_PI / 2.0, lat));
		//lat: 0.966582, lon: 1.89074, theta: 2.084, phi: -0.426
        //theta = max(0.0, min(9.0*M_PI/10.0, theta));
        updateView();
        oldx = x;
        oldy = y;
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
    } else if (c == '+') {
		exposure *= 1.1;
	} else if (c == '-') {
		exposure /= 1.1;
	} else if (c == 's') {
		disp->_scene.switchSunControlState();
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	//} else if (c == 't') {
	//	disp->_scene.switchThetaOnly();
	} else if (c == 'c') {
		disp->_scene.switchCtrlFactor();
	} else if (c == '1') {
		lon = 0.0;
		lat = 0.0;
		phi = 0.0;
    	theta = 0.0;
        //theta = max(0.0, min(9.0*M_PI/10.0, theta));
		theta = max(0.0, min(M_PI, theta));
        updateView();
        oldx = x;
        oldy = y;
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == '2') {
		lon = 1.90129;
		lat = 0.996678;
		phi = -0.39;
    	theta = 2.08957;
        //theta = max(0.0, min(9.0*M_PI/10.0, theta));
		theta = max(0.0, min(M_PI, theta));
        updateView();
        oldx = x;
        oldy = y;
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == '3') {
		//lat: 1.00158, lon: 1.90343, theta: 2.0915, phi: -0.39   (cityx10)
		//lat: 1.00158, lon: 1.90343, theta: 2.97624, phi: -0.39  (no city)
		lon = 1.90319;
		lat = 1.00927;
		phi = -0.39;
    	theta = 2.09362;
        //theta = max(0.0, min(9.0*M_PI/10.0, theta));
		theta = max(0.0, min(M_PI, theta));
        updateView();
        oldx = x;
        oldy = y;
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == 'z') {
		lon = 0.0;
		lat = 0.0;
		theta = 3.14;
		phi = 2.48;
		updateView();
	} else if (c == '5') { //example: using center(x,y,z) for non-scaled Textured\atext_0.obj
		lat = 0.976607;
		lon = 1.92981;
		theta = 2.09144;
		phi =-0.426;
		theta = max(0.0, min(M_PI, theta));
        updateView();
        oldx = x;
        oldy = y;
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == '6') {
		lat = 0.869138;
		lon = 1.88738;
		theta = 2.05733;
		phi = -0.426;
		theta = max(0.0, min(M_PI, theta));
        updateView();
        oldx = x;
        oldy = y;
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == '7') {
		lat = 0.896603;
		lon = 1.88845;
		theta = 2.084;
		phi = -0.426;
		theta = max(0.0, min(M_PI, theta));
        updateView();
        oldx = x;
        oldy = y;
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == '8') {
		lon = 1.89539;
		lat = 0.98997;
		phi = -0.39;
    	theta = 2.09027;
        //theta = max(0.0, min(9.0*M_PI/10.0, theta));
		theta = max(0.0, min(M_PI, theta));
        updateView();
        oldx = x;
        oldy = y;
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == '9') {
		//lat: 0.966582, lon: 1.89074, theta: 2.084, phi: -0.426
		lon = 1.89074;
		lat = 0.966582;
		phi = -0.426;
    	theta = 2.084;
        //theta = max(0.0, min(9.0*M_PI/10.0, theta));
		theta = max(0.0, min(M_PI, theta));
        updateView();
        oldx = x;
        oldy = y;
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == 'n') {

		float vangle = 0.0;
		float hangle = 0.0;
		disp->_scene.getHVAngles(vangle, hangle);
		s.x = cos(vangle) * cos(hangle);
		s.y = cos(vangle) * sin(hangle);
		s.z = sin(vangle);
		
		cout << "vangle: " << vangle << ", hangle" << hangle << ", s.x: " << s.x << ", s.y: " << s.y << ", s.z: " << s.z << endl;
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
		glutPostRedisplay();
	} else if (c == 'h' || c == 'H'){
		cout << "\nKeyboard commands: " << endl;
		cout << "\t<Esc>: exit" << endl;
		cout << "\t<k>: print lon lat theta phi" << endl;
		cout << "\t<+/->: increase/decrease exposure" << endl;
		cout << "\t<s>: enable/disable sun movement, enabled by default" << endl;
		cout << "\t<t>: enable/disable RT Pass testing" << endl;
		//cout << "\t<t>: enable/disable theta angle only manipulation" << endl;
		cout << "\t<m>: switch between single scattering view and photon bin counts atmospheres" << endl; 
		cout << "\t<c>: change the multiplication factor used to calculate changing theta angle as the camera gets closer to planet surface" << endl;
		cout << "\t<2,3,8,9>: position camera at the city location" << endl;
		cout << "\t<f> scale factor in rt_pass for rendering final image" << endl;
		cout << "\t<1>: position camera in space" << endl;
	} else if(c== 'f'){
		float sfactor;
		cout << "Enter  scaling factor(float: ";
		cin >> sfactor;
		disp->_scene.signalScaleFactorChange(sfactor);
		bUpdateRendering = true;
	} else if(c=='t'){
		disp->_scene.signalTestRtPassChange();
		bUpdateRendering = true;
	} else if (c == 'm' || c == 'M'){
		disp->_scene.switchAtmosphereMode();
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == 'a' || c == 'A'){
		disp->_scene.switchAtmosphereMode1();
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == 'b'){
		cz_y += 0.0001;
		updateView();
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c == 'B'){
		cz_y -= 0.0001;
		cz_y = (double)fmaxf(0.0,cz_y);
		updateView();
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c =='l'){
		test_camera  += 0.1;
		updateView();
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	} else if (c =='L'){
		test_camera  -= 0.1;
		updateView();
		disp->_scene.signalLightChanged(make_float3(s.x,s.y,s.z));
		bUpdateRendering = true;
	}


}

void idleFunc_pas()
{
    glutPostRedisplay();
}

#ifdef TEST_PAS
int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(1024, 768);
	//glutInitWindowSize(TRANSMITTANCE_W, TRANSMITTANCE_H);
    glutCreateWindow("Precomputed Atmospheric Scattering");
    glutCreateMenu(NULL);
    glutDisplayFunc(redisplayFunc_pas);
    glutReshapeFunc(reshapeFunc_pas);
    glutMouseFunc(mouseClickFunc_pas);
    glutMotionFunc(mouseMotionFunc_pas);
    glutSpecialFunc(specialKeyFunc_pas);
    glutKeyboardFunc(keyboardFunc_pas);
    glutIdleFunc(idleFunc_pas);
    //glewInit();

	BufInfo[0] = float(BufferSize);
	BufInfo[1] = float(BufferSize);
	BufInfo[2] = 1.0f / float(BufferSize);
	BufInfo[3] = 1.0f / float(BufferSize);

    loadData();
    precompute();
    updateView();
    glutMainLoop();
}
#endif


//////////////////////////////////////////////////////////////////////////////////////////////

