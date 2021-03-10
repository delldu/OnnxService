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

// Transform zcode to wcode
int sample_t(double *x, int n)
{
	int i;
	float *f;
	TENSOR *zcode_tensor, *wcode_tensor;

	zcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);

	wcode_tensor = TensorForward(transformer_engine, zcode_tensor);

	// Update x
	f = wcode_tensor->data;
	for (i = 0; i < n; i++)
		*x++ = (double)*f++;

	tensor_destroy(wcode_tensor);
	tensor_destroy(zcode_tensor);

	return 0;
}

double fitfun(double const *x, unsigned long N)
{
	int i;
	double sum;
	float *f1, *f2;
	TENSOR *wcode_tensor, *output_image_tensor, *output_tensor;

	// Create wcode tensor
	wcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	f1 = wcode_tensor->data;
	for (i = 0; i < W_SPACE_DIM; i++) {
		*f1 = (float)(*x);
		f1++; x++;
	}

	output_image_tensor = TensorForward(decoder_engine, wcode_tensor);
	output_tensor = tensor_zoom(output_image_tensor, 256, 256);

	// Compute loss
	sum = 0;
	f1 = stand_tensor->data;
	f2 = output_tensor->data;
	for (i = 0; i < (int)N; i++) {
		sum += (double)(*f1 - *f2) * (*f1 - *f2);
		f1++; f2++;
	}

	// Release tensors
	tensor_destroy(output_image_tensor);
	tensor_destroy(output_tensor);
	tensor_destroy(wcode_tensor);

	return sum;
}

double *cmaes_search(int epochs)
{
	int i, dimension;
	cmaes_t evo;
	double *cost_values, *xfinal, *const *pop;

	// cost_values = cmaes_init(&evo, W_SPACE_DIM dimmesion, xstart, stddev, 0, / * lambda */, "none");
	cmaes_init_para(&evo, W_SPACE_DIM /*dimmesion*/, NULL, NULL, 0, 0 /* lambda */, "none");

	evo.sp.stopMaxIter = epochs;	// stop after given number of iterations (generations)
	evo.sp.stopFitness.flg = 1;
	evo.sp.stopFitness.val = 1e-3;	// stop if function value is smaller than stopFitness
	evo.sp.stopTolFun = 1e-4;		// stop if function value differences are small
	cost_values =  cmaes_init_final(&evo);


	dimension = (unsigned int) cmaes_Get(&evo, "dimension");
	printf("%s\n", cmaes_SayHello(&evo));

	while (!cmaes_TestForTermination(&evo)) {
		/* generate lambda new search points, sample population */
		pop = cmaes_SamplePopulation(&evo, sample_t);	/* do not change content of pop */

		for (i = 0; i < cmaes_Get(&evo, "lambda"); ++i) {
			cost_values[i] = fitfun(pop[i], dimension);	/* evaluate */
		}

		cmaes_UpdateDistribution(&evo, cost_values);	/* assumes that pop[i] has not been modified */

		if ((int)evo.gen % 10 == 0) {
			printf("Progress %6.2f %% ...\n", (float)(100.0 * evo.gen/epochs));
			fflush(stdout);
		}
	}
	printf("Stop Condition:\n%s\n", cmaes_TestForTermination(&evo));	/* print termination reason */

	cmaes_WriteToFile(&evo, "all", "/tmp/cmaes_results.txt");	/* write final results */

	/* get best estimator for the optimum, xmean */
	xfinal = cmaes_GetNew(&evo, "xmean");

	cmaes_exit(&evo);			/* release memory */

	return xfinal;
}

TENSOR *wcode_search(TENSOR *reference_tensor)
{
	CHECK_TENSOR(reference_tensor);

	return NULL;
}

int FaceGanService(char *endpoint, int use_gpu)
{
	int socket, reqcode, count, rescode;
	TENSOR *input_tensor, *output_tensor, *wcode_tensor;

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	decoder_engine = CreateEngine("image_gandecoder.onnx", use_gpu);
	CheckEngine(decoder_engine);

	transformer_engine = CreateEngine("image_gantransformer.onnx", use_gpu);
	CheckEngine(transformer_engine);


	count = 0;
	for (;;) {
		syslog_info("Service %d times", count);

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
			response_send(socket, output_tensor, rescode);
			tensor_destroy(output_tensor);
		}

		tensor_destroy(wcode_tensor);

		tensor_destroy(input_tensor);

		count++;
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

void help(char *cmd)
{
	printf("Usage: %s [option] <image files>\n", cmd);
	printf("    h, --help                   Display this help.\n");
	printf("    e, --endpoint               Set endpoint.\n");
	printf("    s, --server <0 | 1>         Start server (use gpu).\n");

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
		{0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);

	while ((optc = getopt_long(argc, argv, "h e: s:", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'e':
			endpoint = optarg;
			break;
		case 's':
			running_server = 1;
			use_gpu = atoi(optarg);
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
