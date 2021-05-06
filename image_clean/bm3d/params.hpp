#ifndef _PARAMS_HPP_
#define _PARAMS_HPP_

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short ushort;

//Index handling
#define idx2(x,y,dim_x) ( (x) + ((y)*(dim_x)) )
#define idx3(x,y,z,dim_x,dim_y) ( (x) + ((y)*(dim_x)) + ((z)*(dim_x)*(dim_y)) )

struct Params {
	/*
	   RESTRICTIONS:
	   k must be divisible by p
	 */
	uint n;				// Half of area (in each dim) in which the similar blocks are searched
	uint k;				// width and height of a patch
	uint N;				// Maximal number of similar blocks in stack (without reference block)
	uint T;				// Distance treshold under which two blocks are assumed simialr //DEV: NOT NECESSARY
	uint Tn;			// Distance treshold under which two blocks are assumed simialr (with normalization facotr)
	uint p;				// Step between reference patches
	float L3D;			// Treshold in colaborative filtering under which coefficients are replaced by zeros.

   Params(uint n = 32, uint k = 8, uint N = 8, uint T = 2500, uint p = 3, float L3D = 2.7f):n(n), k(k), N(N - 1), T(T), Tn(T * k * k), p(p),
		L3D(L3D)
	{
}};

#endif
