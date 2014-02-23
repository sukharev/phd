//uniform sampler1D InputTexture;
uniform sampler2D RandomTexture;
uniform int PathLength;
uniform int Length;
uniform int Width;
uniform float MaxPathLength;
uniform vec3 s;

//uniform int UseEyeRays;
//uniform vec4 BufInfo;


//uniform float Time;
//uniform float Wavelength;

const bool FullSpectrum = true; // modify progressive.fs as well
const bool DOF = true;
const bool MotionBlur = true;
const bool Glossy = true;

const float FocalLength = 210.0;
const float ApertureSize = 7.0;
const float MotionSize = 10.0;
const float Glossiness = 0.5;



// WCMYRGBXYZ using Gaussians fitting
const float White = 1.0;

const vec4 Cyan0 = vec4(0.0, 0.424537460743542, 0.0866503554583976, 0.560757618949125);
const vec4 Cyan1 = vec4(0.0, 0.246400896854156, 0.0795161416808855, 0.216116362841135);
const vec4 Cyan2 = vec4(1.0, 0.067666394964209, 0.2698588575757230, 0.890716186803857);

const vec4 Magenta0 = vec4(0.0, 0.092393363155047, -0.030670840714796, 0.425200104381996);
const vec4 Magenta1 = vec4(0.0, 0.174734179228986, 0.0690508593874629, 0.983929883263911);
const vec4 Magenta2 = vec4(1.0, 0.613995338323662, 0.0794711389383399, 1.003105061865860);

const vec4 Yellow0 = vec4(0.0, 0.369673263739623, -0.071355497310236, 0.503666150930812);
const vec4 Yellow1 = vec4(0.0, 0.558410218684172,  0.151858057162275, 0.878349029651678);
const vec4 Yellow2 = vec4(1.0, 0.587945864428471,  0.101005427723483, 0.109960421083442);

const vec4 Red0 = vec4(0.0, 0.574803873802654,  0.0349961565910619, 0.670478585641923);
const vec4 Red1 = vec4(0.0, 0.042753652345675, -0.076576978780864,  0.070884754752968);
const vec4 Red2 = vec4(1.0, 0.669048230499984,  0.0587027396330119, 0.957999219817480);

const vec4 Green0 = vec4(0.0, 0.305242141596798,  0.0337596436768638, 0.424248514020785);
const vec4 Green1 = vec4(0.0, 0.476992126451749, -0.0541085157876399, 0.815789194891182);
const vec4 Green2 = vec4(1.0, 0.365833471799225, -0.0583175076362409, 0.792406519710127);

const vec4 Blue0 = vec4(0.0, 0.144760614900738, 0.0848347582999023, 0.993361426917213);
const vec4 Blue1 = vec4(0.0, 0.600421286424602, -0.060880809655396, 0.0744873773945442);
const vec4 Blue2 = vec4(1.0, 0.231505955455338, -0.029894351908322, 0.339396172335299);


#ifdef _VERTEX_

void main() {
    gl_Position = gl_Vertex;
}

#endif

#ifdef _FRAGMENT_

float Gaussian(const float x0, const float s, const float w, const float x)
{
  return w * exp( -(x - x0) * (x - x0) / (2.0 * s * s + 1.0e-20) );
}



float GaussianMixture(const float lambda, const vec4 Data0, const vec4 Data1, const vec4 Data2)
{
	float t = (lambda - 0.380) / (0.780 - 0.380);
	float g0 = Gaussian(Data0.y, Data0.z, Data0.w, t);
	float g1 = Gaussian(Data1.y, Data1.z, Data1.w, t);
	float g2 = Gaussian(Data2.y, Data2.z, Data2.w, t);

	return min(max(g0 + g1 + g2 + Data0.x, Data1.x), Data2.x);
}



float RGB2Spectrum(const vec3 rgb, const float lambda)
{
	float r2g = rgb.r - rgb.g;
	float g2b = rgb.g - rgb.b;
	float b2r = rgb.b - rgb.r;

	if ((rgb.r <= rgb.g) && (rgb.r <= rgb.b)) 
	{
		if (rgb.g <= rgb.b)
		{
			return rgb.r * White - (r2g * GaussianMixture(lambda, Cyan0, Cyan1, Cyan2) + g2b * GaussianMixture(lambda, Blue0, Blue1, Blue2)); 
		}
		else
		{
			return rgb.r * White + (b2r * GaussianMixture(lambda, Cyan0, Cyan1, Cyan2) + g2b * GaussianMixture(lambda, Green0, Green1, Green2));
		}
	}
	else if ((rgb.g <= rgb.r) && (rgb.g <= rgb.b)) 
	{
		if (rgb.b <= rgb.r)
		{
			return rgb.g * White - (g2b * GaussianMixture(lambda, Magenta0, Magenta1, Magenta2) + b2r * GaussianMixture(lambda, Red0, Red1, Red2)); 
		}
		else
		{
			return rgb.g * White + (r2g * GaussianMixture(lambda, Magenta0, Magenta1, Magenta2) + b2r * GaussianMixture(lambda, Blue0, Blue1, Blue2));
		}
	}
	else 
	{
		if (rgb.r <= rgb.g)
		{
			return rgb.b * White - (b2r * GaussianMixture(lambda, Yellow0, Yellow1, Yellow2) + r2g * GaussianMixture(lambda, Green0, Green1, Green2));
		}
		else
		{
			return rgb.b * White + (g2b * GaussianMixture(lambda, Yellow0, Yellow1, Yellow2) + r2g *GaussianMixture(lambda, Red0, Red1, Red2));
		}
	}
}





float GPURnd(inout vec4 n)
{
	// from http://gpgpu.org/forums/viewtopic.php?t=2591&sid=17051481b9f78fb49fba5b98a5e0f1f3
	const vec4 q = vec4(   1225.0,    1585.0,    2457.0,    2098.0);
	const vec4 r = vec4(   1112.0,     367.0,      92.0,     265.0);
	const vec4 a = vec4(   3423.0,    2646.0,    1707.0,    1999.0);
	const vec4 m = vec4(4194287.0, 4194277.0, 4194191.0, 4194167.0);

	vec4 beta = floor(n / q);
	vec4 p = a * (n - beta * q) - beta * r;
	beta = (sign(-p) + vec4(1.0)) * vec4(0.5) * m;

	n = (p + beta);

	return fract(dot(n / m, vec4(1.0, -1.0, 1.0, -1.0))); 
}



struct Ray
{
	vec3 org;
	vec3 dir;
};
struct Sphere
{
	vec3 c;
	float r;
	vec3 col;
	int f;
	bool part_medium;
};
struct Intersection
{
	float t;
	Sphere sphere;
};



vec3 tangent(const vec3 n)
{
	vec3 t = n;
	vec3 a = abs(n);

	if ((a.x < a.y) && (a.x < a.z))
	{
		t.x = 1.0;
	}
	else if (a.y < a.z)
	{
		t.y = 1.0;
	}
	else
	{
		t.z = 1.0;
	}

	return normalize(cross(t, n));
}



vec3 along(const vec3 v, const vec3 n)
{
	vec3 t = tangent(n);
	vec3 s = cross(t, n);

	return (v.x * t + v.y * n + v.z * s);
}
/*

// use vec3 transmittance(float r, float mu, float d)
// to get transmittance between x and x0 => d, mu, r
void main()
{
	float mu, muS, nu;
	float r = 0.0;
	vec4 dhdH = vec4(0.0,0.0,0.0,0.0);
	
	vec2 PixelIndex = gl_FragCoord.xy -0.5;// * BufInfo.zw;
	vec4 rnd = texture2D(RandomTexture, vec2(PixelIndex.x/Width,PixelIndex.y/Length));//PixelIndex);
	vec4 rndv = rnd;

	if(PathLength == 0){
		r = sqrt(Rg * Rg + (Rt * Rt - Rg * Rg)) -0.001;
	}
	float cthetamin = -sqrt(1.0 - (Rg / r) * (Rg / r));
	float phi = 0.0;
	float theta = 0.0;
	
	Ray ray;
	ray.org.x = 0.0;
	ray.org.y = 0.0;
	ray.org.z = 0.0;
	
	ray.dir.x = 0.0;
	ray.dir.y = 0.0;
	ray.dir.z = 0.0;
	// photon rays
	// point light source
	if(PathLength == 0){
		float muS = (Width * PixelIndex.y + PixelIndex.x)/(Length* Width);
		//float r = Rt-Rg;
		
		// generate random direction
		phi = GPURnd(rndv); //phi
		theta = GPURnd(rndv); //theta
		// rnd.x -> theta, rnd.y -> phi
		phi = M_PI * phi;
		theta = M_PI * theta;//acos(sqrt(1.0 - rnd.y));
		
		//r = sqrt(Rg * Rg + (Rt * Rt - Rg * Rg)) -0.001;
		//ray.org.x = sin(rnd.x);
		//ray.org.y = cos(rnd.x);
		//ray.org.z = 2.0;
		
		
		//vec3 w = vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
		
		//convert position at ray origin from spherical coordinates to cartesian coordinates
		ray.org.x = r*sin(phi) * sin(theta);
		ray.org.y = r*cos(theta);
		ray.org.z = r*cos(phi) * sin(theta);
		
		//calculate direction at the ray origin
		ray.dir.x = 0.0;//sin(phi) * sin(theta);
		ray.dir.y = 0.0;-cos(theta);
		ray.dir.z = -cos(phi) * sin(theta);	
	}

	float ctheta = cos(theta);
	vec3 dir = ray.dir;
	vec3 pos = ray.org;
	vec3 new_pos = vec3(0.0);
	vec3 new_dir = vec3(0.0);
	float dist = 0.0; // dist traveled to the next scattering event
	
    float greflectance = 0.0;
    float dground = 0.0;
    vec3 gtransp = vec3(0.0);
    float E = 0.0;
    vec3 L = vec3(0.0);
    
    // compute transparency gtransp between x and ground
    //greflectance = AVERAGE_GROUND_REFLECTANCE / M_PI;
    //dground = -r * ctheta - sqrt(r * r * (ctheta * ctheta - 1.0) + Rg * Rg);
    //gtransp = transmittance(Rg, -(r * ctheta + dground) / Rg, dground);
    
    //distance calculation
    E = GPURnd(rndv);
    //dist = -log(E)/(betaMSca+betaR);
    //(betaR * exp(-(r - Rg) / HR) * pr2 + betaMSca * exp(-(r - Rg) / HM) * pm2)
    float RND = GPURnd(rndv);
    dist = RND*200.0;
    
    //scattering
    E = GPURnd(rndv);
    L = (betaMSca)/(betaMEx);
    if(E < L.x){
		//scatter
		
		//calculate new position
		new_pos = pos + dist*dir;
		float len = sqrt(dot(new_pos,new_pos));
		if(len > Rt || len < Rg)
			new_pos = vec3(0.0);//pos;
			
		//calculate new direction
		float u1 = GPURnd(rndv);
		float u2 = GPURnd(rndv);
		vec3 prev_dir = dir;
		new_dir = UniformSampleSphere(u1,u2);
		
		//check out what the phase function thinks of your new direction
		float3 ref = Phase(prev_dir,new_dir, new_pos);
		if(Black(ref)){
			new_dir = vec3(0.0);
		}
    }
    else{
		//absorb
		new_pos = vec3(0.0);//vec3(0.0);
		dir = vec3(1.0);
    }

	


	
	// photon tracing
	//col *= vec3(5000.0) * (4.0 * 3.141592) * MaxPathLength;
	//nrm = ray.dir;

	gl_FragData[0] = vec4(new_pos, 1.0);
	gl_FragData[1] = vec4(dir, 1.0);
	//gl_FragData[2] = rndv;
}
*/


void main()
{
	bool bEvent = false;
	bool bOutOfBoundary = false;
	float new_dir = vec3(0.0);
	float new_pos = vec3(0.0);
	vec2 PixelIndex = gl_FragCoord.xy -0.5;// * BufInfo.zw;
	vec4 rndv = texture2D(RandomTexture, vec2(PixelIndex.x/Width,PixelIndex.y/Length));//PixelIndex);
	
	Ray ray;
	ray.org = vec3(0.0);
	ray.dir = vec3(0.0);

	if(PathLength == 0){
		//float muS = (Width * PixelIndex.y + PixelIndex.x)/(Length* Width);
		//float r = Rt-Rg;
		
		// generate random direction
		phi = M_PI * GPURnd(rndv); //phi
		theta = M_PI * GPURnd(rndv); //theta

		ray.dir = s;
		
		//convert position at ray origin from spherical coordinates to cartesian coordinates
		ray.org.x = r*sin(phi) * sin(theta);
		ray.org.y = r*cos(theta);
		ray.org.z = r*cos(phi) * sin(theta);
		
		//calculate direction at the ray origin
		ray.dir.x = 0.0;//sin(phi) * sin(theta);
		ray.dir.y = 0.0;-cos(theta);
		ray.dir.z = -cos(phi) * sin(theta);	
	}
	// determine if event happened
	// 1) find a random direction on a unisphere
	// for the first event determine the direction towards the sun
	float u1 = GPURnd(rndv);
	float u2 = GPURnd(rndv);
	vec3 w = UniformSampleSphere(u1,u2);
	
	// 2) check with phase function if it agrees with the direction
	//v is an incoming direction and w is an outgoing direction
	v = ray.dir;
	float nu = dot(v, w);
    float pr = phaseFunctionR(nu);
    float pm = phaseFunctionM(nu);
	
    // raymie += raymie1 * (betaR * exp(-(r - Rg) / HR) * pr + betaMSca * exp(-(r - Rg) / HM) * pm) * dw;
    float r = dot(ray.org, ray.org);
	new_dir = betaR * exp(-(r - Rg) / HR) * pr + betaMSca * exp(-(r - Rg) / HM) * pm;
	new_pos = ray.org;
	
	int i = 0;
	
	while(!bEvent && !bOutOfBoundary){
		//cases:
		//1) we are inside the atmosphere and ray intersects the atmosphere
		//2) we are inside the atmosphere and ray intersects ground
		float choice = GPURnd(rndv);
		vec3 testProb = vec3(choice);
		float t = ptStep * (float)i;
		new_pos = new_pos + t * new_dir;
		float len = dot(new_pos, new_pos);
		if(len > Rt || len <Rg){
			//TODO need to set event to the boundary of the atmosphere or to the surface of the planet
			bOutOfBoundary = true;
			break;
		}
		vec3 ti = transmittance(r, nu, t);
		vec3 probEvent = 1 - ti; 
		if(testProb.x < probEvent.x){
			bEvent = true;
		}
		if(bEvent){
			// determine the type of event: absorbtion or scattering
			vec3 albedo = betaMSca/betaMEx;
			choice = GPURnd(rndv);
			testProb = vec3(choice);
			if(albedo.x < testProb.x){
				//scatter
				break;	
			}
			else{
				//absorb
				new_pos = vec3(0.0);
				break;
			}		
			
		}
		i=i+1;
	}
	if(!bEvent)
		new_pos = vec3(0.0);

	gl_FragData[0] = vec4(new_pos, 1.0);
	gl_FragData[1] = vec4(new_dir, 1.0);
}

#endif