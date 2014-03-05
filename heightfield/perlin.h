#pragma once

class PerlinNoise
{
public:

  // Constructor
    PerlinNoise();
    PerlinNoise(double _persistence, double _frequency, double _amplitude, int _octaves, int _randomseed);

  // Get Height
    double GetHeight(double x, double y) const;

  // Get
  double Persistence() const { return persistence; }
  double Frequency()   const { return frequency;   }
  double Amplitude()   const { return amplitude;   }
  int    Octaves()     const { return octaves;     }
  int    RandomSeed()  const { return randomseed;  }

  // Set
  void Set(double _persistence, double _frequency, double _amplitude, int _octaves, int _randomseed);

  void SetPersistence(double _persistence) { persistence = _persistence; }
  void SetFrequency(  double _frequency)   { frequency = _frequency;     }
  void SetAmplitude(  double _amplitude)   { amplitude = _amplitude;     }
  void SetOctaves(    int    _octaves)     { octaves = _octaves;         }
  void SetRandomSeed( int    _randomseed)  { randomseed = _randomseed;   }

private:

    double Total(double i, double j) const;
    double GetValue(double x, double y) const;
    double Interpolate(double x, double y, double a) const;
    double Noise(int x, int y) const;

    double persistence, frequency, amplitude;
    int octaves, randomseed;
};


PerlinNoise::PerlinNoise()
{
  persistence = 0;
  frequency = 0;
  amplitude  = 0;
  octaves = 0;
  randomseed = 0;
}

PerlinNoise::PerlinNoise(double _persistence, double _frequency, double _amplitude, int _octaves, int _randomseed)
{
  persistence = _persistence;
  frequency = _frequency;
  amplitude  = _amplitude;
  octaves = _octaves;
  randomseed = 2 + _randomseed * _randomseed;
}

void PerlinNoise::Set(double _persistence, double _frequency, double _amplitude, int _octaves, int _randomseed)
{
  persistence = _persistence;
  frequency = _frequency;
  amplitude  = _amplitude;
  octaves = _octaves;
  randomseed = 2 + _randomseed * _randomseed;
}

double PerlinNoise::GetHeight(double x, double y) const
{
  return amplitude * Total(x, y);
}

double PerlinNoise::Total(double i, double j) const
{
    //properties of one octave (changing each loop)
    double t = 0.0f;
    double _amplitude = 1;
    double freq = frequency;

    for(int k = 0; k < octaves; k++) 
    {
        t += GetValue(j * freq + randomseed, i * freq + randomseed) * _amplitude;
        _amplitude *= persistence;
        freq *= 2;
    }

    return t;
}

double PerlinNoise::GetValue(double x, double y) const
{
    int Xint = (int)x;
    int Yint = (int)y;
    double Xfrac = x - Xint;
    double Yfrac = y - Yint;

  //noise values
  double n01 = Noise(Xint-1, Yint-1);
  double n02 = Noise(Xint+1, Yint-1);
  double n03 = Noise(Xint-1, Yint+1);
  double n04 = Noise(Xint+1, Yint+1);
  double n05 = Noise(Xint-1, Yint);
  double n06 = Noise(Xint+1, Yint);
  double n07 = Noise(Xint, Yint-1);
  double n08 = Noise(Xint, Yint+1);
  double n09 = Noise(Xint, Yint);

  double n12 = Noise(Xint+2, Yint-1);
  double n14 = Noise(Xint+2, Yint+1);
  double n16 = Noise(Xint+2, Yint);

  double n23 = Noise(Xint-1, Yint+2);
  double n24 = Noise(Xint+1, Yint+2);
  double n28 = Noise(Xint, Yint+2);

  double n34 = Noise(Xint+2, Yint+2);

    //find the noise values of the four corners
    double x0y0 = 0.0625*(n01+n02+n03+n04) + 0.125*(n05+n06+n07+n08) + 0.25*(n09);  
    double x1y0 = 0.0625*(n07+n12+n08+n14) + 0.125*(n09+n16+n02+n04) + 0.25*(n06);  
    double x0y1 = 0.0625*(n05+n06+n23+n24) + 0.125*(n03+n04+n09+n28) + 0.25*(n08);  
    double x1y1 = 0.0625*(n09+n16+n28+n34) + 0.125*(n08+n14+n06+n24) + 0.25*(n04);  

    //interpolate between those values according to the x and y fractions
    double v1 = Interpolate(x0y0, x1y0, Xfrac); //interpolate in x direction (y)
    double v2 = Interpolate(x0y1, x1y1, Xfrac); //interpolate in x direction (y+1)
    double fin = Interpolate(v1, v2, Yfrac);  //interpolate in y direction

    return fin;
}

double PerlinNoise::Interpolate(double x, double y, double a) const
{
    double negA = 1.0 - a;
  double negASqr = negA * negA;
    double fac1 = 3.0 * (negASqr) - 2.0 * (negASqr * negA);
  double aSqr = a * a;
    double fac2 = 3.0 * aSqr - 2.0 * (aSqr * a);

    return x * fac1 + y * fac2; //add the weighted factors
}

double PerlinNoise::Noise(int x, int y) const
{
    int n = x + y * 57;
    n = (n << 13) ^ n;
    int t = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;
    return 1.0 - double(t) * 0.931322574615478515625e-9;/// 1073741824.0);
}


class PerlinNoiseGPU
{
public:
	PerlinNoiseGPU();
	void init();

	static int perm[256];
	static int grad3[16][3];
};

PerlinNoiseGPU::PerlinNoiseGPU()
{
	init();
}

void PerlinNoiseGPU::init()
{
}

int PerlinNoiseGPU::perm[256] = {151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
  129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180};

int PerlinNoiseGPU::grad3[16][3] = {{0,1,1},{0,1,-1},{0,-1,1},{0,-1,-1},
                   {1,0,1},{1,0,-1},{-1,0,1},{-1,0,-1},
                   {1,1,0},{1,-1,0},{-1,1,0},{-1,-1,0}, // 12 cube edges
                   {1,0,-1},{-1,0,-1},{0,-1,1},{0,1,1}}; // 4 more to make 16
/*
GLuint permTextureID;

void initPermTexture(GLuint *texID)
{
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
}
*/