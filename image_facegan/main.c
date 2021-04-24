/************************************************************************************
***
***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2020-11-22 13:18:11
***
************************************************************************************/

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <syslog.h>
#include <pthread.h>

#include <nimage/image.h>
#include <nimage/nnmsg.h>

#include "engine.h"
#include "cmaes.h"

#define LAMBDA 22		// CMA population size
#define W_SPACE_DIM 512

#define DEBUG 1

OrtEngine *decoder_engine = NULL;
OrtEngine *transformer_engine = NULL;
OrtEngine *loss_engine = NULL;

TENSOR *standard_tensor = NULL;		// Search Reference tensor !!!

typedef struct {				/* Used as argument to fit_start() */
	pthread_t id;				/* ID returned by pthread_create() */
	int no;						/* Sub thread */
	double x[W_SPACE_DIM], y;
} THREAD_INFOS;

float random_gauss()
{
	// Marsaglia and Bray 1964
    static double V1, V2, S;
    static int phase = 0;
    double X;
     
    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
             
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
         
        X = V1 * sqrt(-2 * log(S) / S);
    } else {
        X = V2 * sqrt(-2 * log(S) / S);
    }
         
    phase = 1 - phase;
 
    return (float)X;
}

void output_image(TENSOR *tensor, const char *prefix, int index)
{
	char filename[256];
	snprintf(filename, sizeof(filename), "output/%s-%04d.png", prefix, index);
	SaveTensorAsImage(tensor, filename);
}

TENSOR *random_zcode()
{
	int i;
	TENSOR *zcode_tensor;

	// Create zcode tensor
	zcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(zcode_tensor);
	for (i = 0; i < W_SPACE_DIM; i++)
		zcode_tensor->data[i] = random_gauss();

	return zcode_tensor;
}

TENSOR *random_wcode()
{
	TENSOR *zcode_tensor, *wcode_tensor;

	CheckEngine(transformer_engine);

	// Create zcode tensor
	zcode_tensor = random_zcode();
	CHECK_TENSOR(zcode_tensor);

	// Compute wcode
	wcode_tensor = TensorForward(transformer_engine, zcode_tensor);
	CHECK_TENSOR(wcode_tensor);

	tensor_destroy(zcode_tensor);

	return wcode_tensor;
}

TENSOR *create_wtensor(double *data)
{
	int i;
	TENSOR *wcode_tensor;

	wcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(wcode_tensor);
	for (i = 0; i < W_SPACE_DIM; i++)
		wcode_tensor->data[i] = (float)data[i];

	return wcode_tensor;
}

float wcode_loss(TENSOR *wcode_tensor)
{
	static int face_index = 1;
	int i, n;
	float loss = 11.0, p_loss;
	TENSOR *output_image_tensor, *output_tensor, *loss_input_tensor, *loss_tensor;

	CheckEngine(decoder_engine);

	output_image_tensor = TensorForward(decoder_engine, wcode_tensor);

	if (! tensor_valid(output_image_tensor)) {
		syslog_error("Generate image.");
		return loss;
	}

	output_tensor = tensor_zoom(output_image_tensor, 256, 256);
	if (! tensor_valid(output_tensor)) {
		syslog_error("Zoom image tensor to (256x256)");
		return loss;
	}

	#if 1 // DEBUG
	output_image(output_image_tensor, "face-", (face_index++) % LAMBDA);
	#endif

	// Compute loss, maybe we need more complex method !!!
	loss = 0.0;
	for (i = 0; i < 256*256*3; i++) {
		loss += ABS(standard_tensor->data[i] - output_tensor->data[i]);
	}
	loss /= (256*256*3);

	// Get perception loss
	loss_input_tensor = tensor_create(2, 3, 256, 256);
	if (! tensor_valid(loss_input_tensor)) {
		return 11.0;
	}

	n = 3*256*256;
	memcpy(loss_input_tensor->data, standard_tensor->data, n * sizeof(float));
	memcpy(&loss_input_tensor->data[n], output_tensor->data, n * sizeof(float));

	loss_tensor = TensorForward(loss_engine, loss_input_tensor);
	if (! tensor_valid(loss_tensor)) {
		return 11.0;
	}

	p_loss = 0;
	n = loss_tensor->batch * loss_tensor->chan * loss_tensor->height * loss_tensor->width;
	for (i = 0; i < n; i++)
		p_loss += loss_tensor->data[i];
	p_loss /= n;

	loss += 10.0 * p_loss;		// perception loss's important is 10.0

	tensor_destroy(loss_tensor);
	tensor_destroy(loss_input_tensor);

	// Release tensors
	tensor_destroy(output_image_tensor);
	tensor_destroy(output_tensor);

	return loss;
}

static void *fit_start(void *arg)
{
	THREAD_INFOS *info = (THREAD_INFOS *)arg;

	CheckEngine(decoder_engine);

	// syslog_info("Running thread %d ...", info->no);
	// Create input tensor
	TENSOR *wcode_tensor;
	wcode_tensor = create_wtensor(info->x);
	info->y = wcode_loss(wcode_tensor);
	// Release tensors
	tensor_destroy(wcode_tensor);
	// syslog_info("Thread %d result = %.4f", info->no, info->y);

	return NULL;
}

// The following code borrow from cmaes_SamplePopulation for wcode samples
double *const *cmaes_SamplePopulationFromWSpace(cmaes_t * t)
{
	int iNk, i, j, N = t->sp.N;
	int flgdiag = ((t->sp.diagonalCov == 1) || (t->sp.diagonalCov >= t->gen));
	double sum;
	double const *xmean = t->rgxmean;
	TENSOR *wcode_tensor;

	/* cmaes_SetMean(t, xmean); * xmean could be changed at this point */

	/* calculate eigensystem  */
	if (!t->flgEigensysIsUptodate) {
		if (!flgdiag)
			cmaes_UpdateEigensystem(t, 0);
		else {
			for (i = 0; i < N; ++i)
				t->rgD[i] = sqrt(t->C[i][i]);
			t->minEW = douSquare(rgdouMin(t->rgD, N));
			t->maxEW = douSquare(rgdouMax(t->rgD, N));
			t->flgEigensysIsUptodate = 1;
			cmaes_timings_start(&t->eigenTimings);
		}
	}

	/* treat minimal standard deviations and numeric problems */
	TestMinStdDevs(t);

	// CheckPoint("flgdiag = %d", flgdiag); flgdiag == 0

	for (iNk = 0; iNk < t->sp.lambda; ++iNk) {	/* generate scaled cmaes_random vector (D * z)    */
		// Sample randn(N)
		wcode_tensor = random_wcode();
		if (! tensor_valid(wcode_tensor)) {
			syslog_error("Create wcode tensor !!!");
			continue;
		}
		for (i = 0; i < N; ++i) {
			if (flgdiag)
				t->rgrgx[iNk][i] = xmean[i] + t->sigma * t->rgD[i] * (double)wcode_tensor->data[i];
			else
				t->rgdTmp[i] = t->rgD[i] * (double)wcode_tensor->data[i];
		}
		if (!flgdiag)
			/* add mutation (sigma * B * (D*z)) */
			for (i = 0; i < N; ++i) {
				for (j = 0, sum = 0.; j < N; ++j)
					sum += t->B[i][j] * t->rgdTmp[j];
				t->rgrgx[iNk][i] = xmean[i] + t->sigma * sum;
			}

		tensor_destroy(wcode_tensor);
	}
	if (t->state == 3 || t->gen == 0)
		++t->gen;
	t->state = 1;

	return (t->rgrgx);
}								/* SamplePopulation() */

double *cmaes_search(int epochs)
{
	cmaes_t evo;
	int i, j, lambda, ret;
	double *cost_values, *xfinal, *const *pop;
	THREAD_INFOS *info;
	pthread_attr_t attr;
	double xstart[W_SPACE_DIM] = {0};
	double stddev[W_SPACE_DIM] = {1.0};

	for (i = 0; i < W_SPACE_DIM; i++) {
		xstart[i] = 0.0;
		stddev[i] = 1.0;
	}

	// cost_values = cmaes_init(&evo, W_SPACE_DIM dimmesion, xstart, stddev, 0, / * lambda */, "none");
	cmaes_init_para(&evo, W_SPACE_DIM /*dimmesion*/, xstart, stddev, 0, LAMBDA  /* lambda */, "none");

	evo.sp.stopMaxIter = epochs;	// stop after given number of iterations (generations)
	evo.sp.stopFitness.flg = 1;
	evo.sp.stopFitness.val = 1e-2;	// stop if function value is smaller than stopFitness
	evo.sp.stopTolFun = 1e-3;		// stop if function value differences are small
	cost_values =  cmaes_init_final(&evo);

	// dimension = (unsigned int) cmaes_Get(&evo, "dimension");
	syslog_info("%s", cmaes_SayHello(&evo));

	lambda = cmaes_Get(&evo, "lambda");

	info = (THREAD_INFOS *)calloc(lambda, sizeof(THREAD_INFOS));
	if (info == NULL)
		syslog_error("Allocte memory.");

	while (!cmaes_TestForTermination(&evo)) {
		/* generate lambda new search points, sample population */
		pop = cmaes_SamplePopulationFromWSpace(&evo);	/* do not change content of pop */
		/* Initialize thread creation attributes */
		ret = pthread_attr_init(&attr);
		if (ret != 0)
			syslog_error("Init thread attr.");

		// for (i = 0; i < lambda; ++i) {
		// 	cost_values[i] = fitfun(pop[i], dimension);	/* evaluate */
		// }
		for (i = 0; i < lambda; i++) {
			info[i].no = i + 1;
			for (j = 0; j < W_SPACE_DIM; j++)
				info[i].x[j] = pop[i][j];
			info[i].y = 0.0;

			ret = pthread_create(&info[i].id, &attr, &fit_start, &info[i]);
			if (ret != 0)
				syslog_error("Create thread.");
		}
		ret = pthread_attr_destroy(&attr);
		if (ret != 0)
			syslog_error("Destropy thread attr.");

		/* Now join with each thread */
		for (i = 0; i < lambda; i++) {
			ret = pthread_join(info[i].id, NULL);
			if (ret != 0)
				syslog_error("Join thread.");
		}

		// Save cost
		for (i = 0; i < lambda; i++) {
			cost_values[i] = info[i].y;
			// syslog_info("Cost %d = %.4f", i, cost_values[i]);
		}

		cmaes_UpdateDistribution(&evo, cost_values);	/* assumes that pop[i] has not been modified */

		// if ((int)evo.gen % 5 == 0) {
		// 	syslog_info("Progress %6.2f %%, loss = %.2lf ...", 
		// 		(float)(100.0 * evo.gen/epochs),  evo.rgFuncValue[evo.index[0]]);
		// 	// fflush(stdout);
		// }
		syslog_info("Progress %6.2f %%, loss = %.4lf ...", 
			(float)(100.0 * evo.gen/epochs),  evo.rgFuncValue[evo.index[0]]);
#if DEBUG
		// Xbest ...
		xfinal = cmaes_GetNew(&evo, "xbest");
		TENSOR *wcode_tensor = create_wtensor(xfinal);
		TENSOR *output_tensor = TensorForward(decoder_engine, wcode_tensor);
		output_image(output_tensor, "best-", evo.gen);
		tensor_destroy(output_tensor);
		tensor_destroy(wcode_tensor);
		free(xfinal);
#endif

	}
	free(info);

	syslog_info("Stop Condition:%s", cmaes_TestForTermination(&evo));	/* print termination reason */

	cmaes_WriteToFile(&evo, "all", "output/cmaes_results.txt");	/* write final results */

	/* get best estimator for the optimum, xmean */
	xfinal = cmaes_GetNew(&evo, "xmean");

	cmaes_exit(&evo);			/* release memory */

	return xfinal;
}

TENSOR *wcode_search(TENSOR *reference_tensor)
{
	int i;
	double *best;
	TENSOR *wcode_tensor;

	CHECK_TENSOR(reference_tensor);

	standard_tensor = reference_tensor;
	best = cmaes_search(100);

	// Save best to wcode
	wcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(wcode_tensor);
	for (i = 0; i < W_SPACE_DIM; i++)
		wcode_tensor->data[i] = (float)best[i];

	free(best);

	return wcode_tensor;
}

int FaceGanService(char *endpoint, int use_gpu)
{
	int socket, lambda, msgcode;
	TENSOR *input_tensor, *output_tensor, *wcode_tensor;

	srand(time(NULL));

	InitEngineRunningTime(); // Avoid compiler complaint

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	decoder_engine = CreateEngine((char *)"image_gandecoder.onnx", 0 /*not use_gpu for model bug !!!*/);
	CheckEngine(decoder_engine);

	transformer_engine = CreateEngine((char *)"image_gantransformer.onnx", use_gpu /*use_gpu*/);
	CheckEngine(transformer_engine);

	loss_engine = CreateEngine((char *)"image_ganloss.onnx", use_gpu /*use_gpu*/);
	CheckEngine(loss_engine);

	lambda = 0;
	for (;;) {
		syslog_info("Service %d times", lambda);

		input_tensor = service_request(socket, &msgcode);
		if (!tensor_valid(input_tensor))
			continue;

		// msgcode = IMAGE_FACEGAN_SERVICE;
		// Real service ...
		time_reset();
		wcode_tensor = wcode_search(input_tensor); check_tensor(wcode_tensor);

		output_tensor = TensorForward(decoder_engine, wcode_tensor);
		time_spend((char *)"Infer");

		if (tensor_valid(output_tensor)) {
#if 0	// For debug			
			IMAGE *image = image_from_tensor(output_tensor, 0);
			image_save(image, "output/last.png");
			image_destroy(image);
#endif
			service_response(socket, IMAGE_FACEGAN_SERVICE, output_tensor);
			tensor_destroy(output_tensor);
		}

		tensor_destroy(wcode_tensor);

		tensor_destroy(input_tensor);

		lambda++;
	}
	DestroyEngine(loss_engine);
	DestroyEngine(transformer_engine);
	DestroyEngine(decoder_engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

int server(char *endpoint, int use_gpu)
{
	return FaceGanService(endpoint, use_gpu);
}

int facegan(int socket, char *input_file)
{
	IMAGE *send_image, *resize_send_image;
	TENSOR *send_tensor, *recv_tensor;

	printf("Face Searching %s ...\n", input_file);

	send_image = image_load(input_file); check_image(send_image);
	resize_send_image = image_zoom(send_image, 256, 256, 1); check_image(resize_send_image);

	if (image_valid(resize_send_image)) {
		send_tensor = tensor_from_image(resize_send_image, 0);
		check_tensor(send_tensor);

		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_FACEGAN_SERVICE);
		if (tensor_valid(recv_tensor)) {
			SaveTensorAsImage(recv_tensor, input_file);
			tensor_destroy(recv_tensor);
		}

		tensor_destroy(send_tensor);
		image_destroy(resize_send_image);
	}
	image_destroy(send_image);

	return RET_OK;
}

int test()
{
	TENSOR *wcode_tensor, *output_tensor;
	IMAGE *image;

	srand(time(NULL));

	transformer_engine = CreateEngine((char *)"image_gantransformer.onnx", 0);
	CheckEngine(transformer_engine);

	decoder_engine = CreateEngine((char *)"image_gandecoder.onnx", 0);
	CheckEngine(decoder_engine);

	time_reset();

	wcode_tensor = random_wcode();
	check_tensor(wcode_tensor);

	output_tensor = TensorForward(decoder_engine, wcode_tensor);
	time_spend((char *)"Infer");

	image = image_from_tensor(output_tensor, 0);
	image_save(image, "test.png");
	image_destroy(image);
	
	tensor_destroy(output_tensor);
	tensor_destroy(wcode_tensor);

	// Release ...
	DestroyEngine(decoder_engine);
	DestroyEngine(transformer_engine);

	return 0;
}


void help(char *cmd)
{
	printf("Usage: %s [option] <image files>\n", cmd);
	printf("    h, --help                   Display this help.\n");
	printf("    e, --endpoint               Set endpoint.\n");
	printf("    s, --server <0 | 1>         Start server (use gpu).\n");
	printf("    t, --test                   Test.\n");

	exit(1);
}

int main(int argc, char **argv)
{
	int i, optc;
	int use_gpu = 1;
	int running_server = 0;
	int socket;

	int option_index = 0;
	char *endpoint = (char *) IMAGE_FACEGAN_URL;

	struct option long_opts[] = {
		{"help", 0, 0, 'h'},
		{"endpoint", 1, 0, 'e'},
		{"server", 1, 0, 's'},
		{"test", 0, 0, 't'},

		{0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);

	while ((optc = getopt_long(argc, argv, "h e: s: t", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'e':
			endpoint = optarg;
			break;
		case 's':
			running_server = 1;
			use_gpu = atoi(optarg);
			break;
		case 't':
			return test();
			break;

		case 'h':				// help
		default:
			help(argv[0]);
			break;
		}
	}

	if (running_server)
		return server(endpoint, use_gpu);
	else if (argc > 1) {
		if ((socket = client_open(endpoint)) < 0)
			return RET_ERROR;

		for (i = 1; i < argc; i++)
			facegan(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	return RET_ERROR;
}
