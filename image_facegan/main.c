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

#define IMAGE_FACEGAN_REQCODE 0x0108
// #define IMAGE_FACEGAN_URL "ipc:///tmp/image_facegan.ipc"
#define IMAGE_FACEGAN_URL "tcp://127.0.0.1:9108"

#define W_SPACE_DIM 512

OrtEngine *decoder_engine = NULL;
OrtEngine *transformer_engine = NULL;
TENSOR *stand_tensor = NULL;

typedef struct {				/* Used as argument to fit_start() */
	pthread_t id;				/* ID returned by pthread_create() */
	int no;						/* Sub thread */
	double *x, y;
} THREAD_INFOS;


// Transform zcode to wcode
int sample_t(double *x, int n)
{
	int i;
	TENSOR *zcode_tensor, *wcode_tensor;

	CheckEngine(transformer_engine);

	// Create zcode tensor
	zcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	if (! tensor_valid(zcode_tensor)) {
		syslog_error("Create zcode tensor.");
	}
	for (i = 0; i < n; i++)
		zcode_tensor->data[i] = (float)x[i];

	// Compute wcode
	wcode_tensor = TensorForward(transformer_engine, zcode_tensor);
	if (! tensor_valid(wcode_tensor)) {
		syslog_error("Compute wcode from zcode.");
	}

	// Debug ...
	#if 0
	TENSOR *output_tensor = TensorForward(decoder_engine, wcode_tensor);check_tensor(output_tensor);
	IMAGE *image = image_from_tensor(output_tensor, 0); check_image(image);
	image_save(image, "/tmp/facegan.png");
	image_destroy(image);
	#endif

	// Save wcode to x
	for (i = 0; i < n; i++)
		x[i] = (double)wcode_tensor->data[i];

	tensor_destroy(wcode_tensor);
	tensor_destroy(zcode_tensor);

	return 0;
}

static void *fit_start(void *arg)
{
	int i;
	TENSOR *wcode_tensor, *output_image_tensor, *output_tensor;
	THREAD_INFOS *info = (THREAD_INFOS *)arg;

	CheckEngine(decoder_engine);

	// syslog_info("Running thread %d ...", info->no);
	// Create wcode tensor
	wcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	if (! tensor_valid(wcode_tensor)) {
		syslog_error("Create wcode tensor");
	}
	for (i = 0; i < W_SPACE_DIM; i++)
		wcode_tensor->data[i] = (float)info->x[i];

	// Generator image from wcode
	output_image_tensor = TensorForward(decoder_engine, wcode_tensor);
	if (! tensor_valid(output_image_tensor)) {
		syslog_error("Generate image.");
	}
	output_tensor = tensor_zoom(output_image_tensor, 256, 256);
	if (! tensor_valid(output_tensor)) {
		syslog_error("Zoom image tensor to (256x256)");
	}

	// Compute loss, maybe we need more complex method !!!
	info->y = 0.0;
	for (i = 0; i < W_SPACE_DIM; i++) {
		info->y += (double)(stand_tensor->data[i] - output_tensor->data[i]) * 
				(stand_tensor->data[i] - output_tensor->data[i]);
	}
	info->y /= W_SPACE_DIM;
	// syslog_info("Thread %d result = %.4f", info->no, sum/W_SPACE_DIM);

	// Release tensors
	tensor_destroy(output_image_tensor);
	tensor_destroy(output_tensor);
	tensor_destroy(wcode_tensor);

	return NULL;
}


double *cmaes_search(int epochs)
{
	cmaes_t evo;
	int i, lambda, ret;
	double *cost_values, *xfinal, *const *pop;
	THREAD_INFOS *info;
	pthread_attr_t attr;

	// cost_values = cmaes_init(&evo, W_SPACE_DIM dimmesion, xstart, stddev, 0, / * lambda */, "none");
	cmaes_init_para(&evo, W_SPACE_DIM /*dimmesion*/, NULL, NULL, 0, 0 /* lambda */, "none");

	evo.sp.stopMaxIter = epochs;	// stop after given number of iterations (generations)
	evo.sp.stopFitness.flg = 1;
	evo.sp.stopFitness.val = 1e-2;	// stop if function value is smaller than stopFitness
	evo.sp.stopTolFun = 1e-3;		// stop if function value differences are small
	cost_values =  cmaes_init_final(&evo);

	// dimension = (unsigned int) cmaes_Get(&evo, "dimension");
	syslog_info("%s", cmaes_SayHello(&evo));

	lambda = cmaes_Get(&evo, "lambda");

	while (!cmaes_TestForTermination(&evo)) {
		/* generate lambda new search points, sample population */
		pop = cmaes_SamplePopulation(&evo, sample_t);	/* do not change content of pop */

		/* Initialize thread creation attributes */
		ret = pthread_attr_init(&attr);
		if (ret != 0)
			syslog_error("pthread_attr_init");

		info = (THREAD_INFOS *)calloc(lambda, sizeof(THREAD_INFOS));
		if (info == NULL)
			syslog_error("Allocte memory.");

		// for (i = 0; i < lambda; ++i) {
		// 	cost_values[i] = fitfun(pop[i], dimension);	/* evaluate */
		// }
		for (i = 0; i < lambda; i++) {
			info[i].no = i + 1;
			info[i].x = (double *)&pop[i];

			ret = pthread_create(&info[i].id, &attr, &fit_start, &info[i]);
			if (ret != 0)
				syslog_error("pthread_create");
		}
		ret = pthread_attr_destroy(&attr);
		if (ret != 0)
			syslog_error("pthread_attr_destroy");

		/* Now join with each thread */
		for (i = 0; i < lambda; i++) {
			ret = pthread_join(info[i].id, NULL);
			if (ret != 0)
				syslog_error("pthread_join");
		}

		// Save cost
		for (i = 0; i < lambda; i++) {
			cost_values[i] = info[i].y;
		}

		free(info);

		cmaes_UpdateDistribution(&evo, cost_values);	/* assumes that pop[i] has not been modified */

		// if ((int)evo.gen % 5 == 0) {
		// 	syslog_info("Progress %6.2f %%, loss = %.2lf ...", 
		// 		(float)(100.0 * evo.gen/epochs),  evo.rgFuncValue[evo.index[0]]);
		// 	// fflush(stdout);
		// }
		syslog_info("Progress %6.2f %%, loss = %.4lf ...", 
			(float)(100.0 * evo.gen/epochs),  evo.rgFuncValue[evo.index[0]]);
	}
	syslog_info("Stop Condition:%s", cmaes_TestForTermination(&evo));	/* print termination reason */

	cmaes_WriteToFile(&evo, "all", "/tmp/cmaes_results.txt");	/* write final results */

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

	wcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(wcode_tensor);

	stand_tensor = reference_tensor;
	best = cmaes_search(10);

	// Save best to wcode
	for (i = 0; i < W_SPACE_DIM; i++)
		wcode_tensor->data[i] = (float)best[i];

	free(best);

	return wcode_tensor;
}

int FaceGanService(char *endpoint, int use_gpu)
{
	int socket, reqcode, lambda, rescode;
	TENSOR *input_tensor, *output_tensor, *wcode_tensor;

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	decoder_engine = CreateEngine("image_gandecoder.onnx", 0 /*not use_gpu for model bug !!!*/);
	CheckEngine(decoder_engine);

	transformer_engine = CreateEngine("image_gantransformer.onnx", use_gpu);
	CheckEngine(transformer_engine);

	lambda = 0;
	for (;;) {
		syslog_info("Service %d times", lambda);

		input_tensor = request_recv(socket, &reqcode);

		if (!tensor_valid(input_tensor)) {
			syslog_error("Request recv bad tensor ...");
			continue;
		}
		syslog_info("Request Code = %d", reqcode);

		// Real service ...
		time_reset();
		wcode_tensor = wcode_search(input_tensor); check_tensor(wcode_tensor);

		output_tensor = TensorForward(decoder_engine, wcode_tensor);
		time_spend((char *)"Infer");

		if (tensor_valid(output_tensor)) {
			rescode = reqcode;
#if 1	// For debug			
			IMAGE *image = image_from_tensor(output_tensor, 0);
			image_save(image, "/tmp/decoder.png");
			image_destroy(image);
#endif
			response_send(socket, output_tensor, rescode);
			tensor_destroy(output_tensor);
		}

		tensor_destroy(wcode_tensor);

		tensor_destroy(input_tensor);

		lambda++;
	}
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
	int rescode;
	IMAGE *send_image, *resize_send_image;
	TENSOR *send_tensor, *recv_tensor;

	printf("Face Searching %s ...\n", input_file);

	send_image = image_load(input_file); check_image(send_image);
	resize_send_image = image_zoom(send_image, 256, 256, 1); check_image(resize_send_image);

	if (image_valid(resize_send_image)) {
		send_tensor = tensor_from_image(resize_send_image, 0);
		check_tensor(send_tensor);

		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_FACEGAN_REQCODE, &rescode);
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
	int i;
	TENSOR *wcode_tensor, *output_tensor;
	IMAGE *image;

	decoder_engine = CreateEngine("image_gandecoder.onnx", 0);
	CheckEngine(decoder_engine);

	time_reset();
	wcode_tensor = tensor_create(1, 1, 1, 512);
	for (i = 0; i < 512; i++)
		wcode_tensor->data[0] = 0;
	output_tensor = TensorForward(decoder_engine, wcode_tensor);
	time_spend((char *)"Infer");

	image = image_from_tensor(output_tensor, 0);
	image_save(image, "zero.png");
	image_destroy(image);


	tensor_destroy(output_tensor);
	tensor_destroy(wcode_tensor);

	DestroyEngine(decoder_engine);

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

	help(argv[0]);

	return RET_ERROR;
}
