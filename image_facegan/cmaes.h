/* --------------------------------------------------------- */
/* --- File: cmaes.h ----------- Author: Nikolaus Hansen --- */
/* ---------------------- last modified: IX 2010         --- */
/* --------------------------------- by: Nikolaus Hansen --- */
/* --------------------------------------------------------- */
/*   
     CMA-ES for non-linear function minimization. 

     Copyright (C) 1996, 2003-2010  Nikolaus Hansen. 
     e-mail: nikolaus.hansen (you know what) inria.fr
      
     License: see file cmaes.c
   
*/
#ifndef NH_cmaes_h				/* only include ones */
#define NH_cmaes_h

#include <time.h>

#define CheckPoint(fmt, arg...) printf("# CheckPoint: %d(%s): " fmt "\n", (int)__LINE__, __FILE__, ##arg)

typedef struct
/* cmaes_random_t 
 * sets up a pseudo random number generator instance 
 */
{
	/* Variables for Uniform() */
	long int startseed;
	long int aktseed;
	long int aktrand;

	/* Variables for Gauss() */
	short flgstored;
	double hold;
} cmaes_random_t;

typedef struct
/* cmaes_timings_t 
 * time measurement, used to time eigendecomposition 
 */
{
	/* for outside use */
	double totaltime;			/* zeroed by calling re-calling cmaes_timings_start */
	double totaltotaltime;
	double tictoctime;
	double lasttictoctime;

	/* local fields */
	clock_t lastclock;
	time_t lasttime;
	clock_t ticclock;
	time_t tictime;
	short istic;
	short isstarted;

	double lastdiff;
	double tictoczwischensumme;
} cmaes_timings_t;

typedef struct
/* cmaes_readpara_t
 * collects all parameters, in particular those that are read from 
 * a file before to start. This should split in future? 
 */
{
	char *filename;				/* keep record of the file that was taken to read parameters */
	short flgsupplemented;

	/* input parameters */
	int N;						/* problem dimension, must stay constant, should be unsigned or long? */
	unsigned int seed;
	double *xstart;
	double *typicalX;
	int typicalXcase;
	double *rgInitialStds;
	double *rgDiffMinChange;

	/* termination parameters */
	double stopMaxFunEvals;
	double facmaxeval;
	double stopMaxIter;
	struct {
		int flg;
		double val;
	} stopFitness;
	double stopTolFun;
	double stopTolFunHist;
	double stopTolX;
	double stopTolUpXFactor;

	/* internal evolution strategy parameters */
	int lambda;					/* -> mu, <- N */
	int mu;						/* -> weights, (lambda) */
	double mucov, mueff;		/* <- weights */
	double *weights;			/* <- mu, -> mueff, mucov, ccov */
	double damps;				/* <- cs, maxeval, lambda */
	double cs;					/* -> damps, <- N */
	double ccumcov;				/* <- N */
	double ccov;				/* <- mucov, <- N */
	double diagonalCov;			/* number of initial iterations */
	struct {
		int flgalways;
		double modulo;
		double maxtime;
	} updateCmode;
	double facupdateCmode;

	/* supplementary variables */

	char *weigkey;
	char resumefile[99];
	const char **rgsformat;
	void **rgpadr;
	const char **rgskeyar;
	double ***rgp2adr;
	int n1para, n1outpara;
	int n2para;
} cmaes_readpara_t;

typedef struct
/* cmaes_t 
 * CMA-ES "object" 
 */
{
	const char *version;
	/* char *signalsFilename; */
	cmaes_readpara_t sp;
	cmaes_random_t rand;		/* random number generator */

	double sigma;				/* step size */

	double *rgxmean;			/* mean x vector, "parent" */
	double *rgxbestever;
	double **rgrgx;				/* range of x-vectors, lambda offspring */
	int *index;					/* sorting index of sample pop. */
	double *arFuncValueHist;

	short flgIniphase;			/* not really in use anymore */
	short flgStop;

	double chiN;
	double **C;					/* lower triangular matrix: i>=j for C[i][j] */
	double **B;					/* matrix with normalize eigenvectors in columns */
	double *rgD;				/* axis lengths */

	double *rgpc;
	double *rgps;
	double *rgxold;
	double *rgout;
	double *rgBDz;				/* for B*D*z */
	double *rgdTmp;				/* temporary (random) vector used in different places */
	double *rgFuncValue;
	double *publicFitness;		/* returned by cmaes_init() */

	double gen;					/* Generation number */
	double countevals;
	double state;				/* 1 == sampled, 2 == not in use anymore, 3 == updated */

	double maxdiagC;			/* repeatedly used for output */
	double mindiagC;
	double maxEW;
	double minEW;

	char sOutString[330];		/* 4x80 */

	short flgEigensysIsUptodate;
	short flgCheckEigen;		/* control via cmaes_signals.par */
	double genOfEigensysUpdate;
	cmaes_timings_t eigenTimings;

	double dMaxSignifKond;
	double dLastMinEWgroesserNull;

	short flgresumedone;

	time_t printtime;
	time_t writetime;			/* ideally should keep track for each output file */
	time_t firstwritetime;
	time_t firstprinttime;

} cmaes_t;

/* --- initialization, constructors, destructors --- */

	double *cmaes_init(cmaes_t *, int dimension, double *xstart,
					   double *stddev, long seed, int lambda, const char *input_parameter_filename);
	void cmaes_init_para(cmaes_t *, int dimension, double *xstart,
						 double *stddev, long seed, int lambda, const char *input_parameter_filename);
	double *cmaes_init_final(cmaes_t *);
	void cmaes_resume_distribution(cmaes_t * evo_ptr, char *filename);
	void cmaes_exit(cmaes_t *);

/* --- core functions --- */
	double *const *cmaes_SamplePopulation(cmaes_t *, double wsamples[][512]);
	double *cmaes_UpdateDistribution(cmaes_t *, const double *rgFitnessValues);
	const char *cmaes_TestForTermination(cmaes_t *);

/* --- additional functions --- */
	double *const *cmaes_ReSampleSingle(cmaes_t * t, int index);
	double *cmaes_SampleSingleInto(cmaes_t * t, double *rgx);
	void cmaes_UpdateEigensystem(cmaes_t *, int flgforce);

/* --- getter functions --- */
	double cmaes_Get(cmaes_t *, char const *keyword);
	const double *cmaes_GetPtr(cmaes_t *, char const *keyword);	/* e.g. "xbestever" */
	double *cmaes_GetNew(cmaes_t * t, char const *keyword);	/* user is responsible to free */
	double *cmaes_GetInto(cmaes_t * t, char const *keyword, double *mem);	/* allocs if mem==NULL, user is responsible to free */

/* --- online control and output --- */
	void cmaes_ReadSignals(cmaes_t *, char const *filename);
	void cmaes_WriteToFile(cmaes_t *, const char *szKeyWord, const char *output_filename);
	char *cmaes_SayHello(cmaes_t *);
/* --- misc --- */
	double *cmaes_NewDouble(int n);	/* user is responsible to free */
	void cmaes_FATAL(char const *s1, char const *s2, char const *s3, char const *s4);
#endif
