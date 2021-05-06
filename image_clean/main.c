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

#include "bm3d/bm3d.h"

#include "engine.h"

TENSOR *guide_filter(TENSOR *input_tensor, int parameters)
{
	int radius;
	IMAGE *image;
	TENSOR *output_tensor;

	CHECK_TENSOR(input_tensor);

	radius = (int)(0.10 * parameters); 	// max-radius is 10 !!!

	image = image_from_tensor(input_tensor, 0);	// batch 0
	CHECK_IMAGE(image);

	// int image_guided_filter(IMAGE * img, IMAGE * guidance, int radius, float eps, int scale, int debug)
	// scale > 1 --> fast filter ...
	image_guided_filter(image, NULL, radius, 0.000001, 2 /*scale*/, 0 /*debug*/);

	if (input_tensor->chan == 4 || input_tensor->chan == 2)
		output_tensor = tensor_from_image(image, 1);	// with alpha
	else
		output_tensor = tensor_from_image(image, 0);	// without alpha
	image_destroy(image);

	return output_tensor;
}

TENSOR *bm3d_filter(TENSOR *input_tensor, int parameters)
{
	int sigma;
	IMAGE *input_image, *output_image;
	TENSOR *output_tensor;

	CHECK_TENSOR(input_tensor);

	sigma = parameters; 	// max-sigma is 100 !!!

	input_image = image_from_tensor(input_tensor, 0);	// batch 0
	CHECK_IMAGE(input_image);

	output_image = image_copy(input_image);
	CHECK_IMAGE(output_image);
 
    // int bm3d(unsigned char *imgdata, int channels, int height, int width, int sigma, unsigned char *outdata, int debug);
    bm3d((unsigned char *)input_image->base, 3, input_image->height, input_image->width, sigma, 
    	(unsigned char *)output_image->base, 0);

	if (input_tensor->chan == 4 || input_tensor->chan == 2)
		output_tensor = tensor_from_image(output_image, 1);	// with alpha
	else
		output_tensor = tensor_from_image(output_image, 0);	// without alpha

	image_destroy(output_image);
	image_destroy(input_image);

	return output_tensor;
}

TENSOR *haze_filter(TENSOR *input_tensor, int parameters)
{
	int radius;
	IMAGE *image;
	TENSOR *output_tensor;

	CHECK_TENSOR(input_tensor);

	radius = (int)(0.10 * parameters); 	// max-radius is 10 !!!

	image = image_from_tensor(input_tensor, 0);	// batch 0
	CHECK_IMAGE(image);

	// int image_dehaze_filter(IMAGE * img, int radius, int debug)
	image_dehaze_filter(image, radius, 0 /*debug*/);

	if (input_tensor->chan == 4 || input_tensor->chan == 2)
		output_tensor = tensor_from_image(image, 1);	// with alpha
	else
		output_tensor = tensor_from_image(image, 0);	// without alpha
	image_destroy(image);

	return output_tensor;
}

int CleanOnnxService(char *endpoint, char *onnx_file, int use_gpu, CustomSevice custom_service_function)
{
	int socket, count, msgcode;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *engine = NULL;

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	if (!custom_service_function)
		custom_service_function = service_response;

	count = 0;
	for (;;) {
		if (EngineIsIdle())
			StopEngine(engine);

		if (!socket_readable(socket, 1000))	// timeout 1 s
			continue;

		input_tensor = service_request(socket, &msgcode);
		if (!tensor_valid(input_tensor))
			continue;

		syslog_info("Cleaning service %d times", count);
		if (SERVICE_CODE(msgcode) == IMAGE_CLEAN_SERVICE) {
			StartEngine(engine, onnx_file, use_gpu);

			// Real service ...
			time_reset();
			output_tensor = TensorForward(engine, input_tensor);
			time_spend((char *) "Deep cleaning");

			service_response(socket, IMAGE_CLEAN_SERVICE, output_tensor);
			tensor_destroy(output_tensor);
		} else if (SERVICE_CODE(msgcode) == IMAGE_CLEAN_SERVICE_WITH_GUIDED_FILTER) {
			// Real service ...
			time_reset();
			output_tensor = guide_filter(input_tensor, SERVICE_ARGUMENT(msgcode));
			time_spend((char *) "Guided cleaning");

			service_response(socket, IMAGE_CLEAN_SERVICE_WITH_GUIDED_FILTER, output_tensor);
			tensor_destroy(output_tensor);
		} else if (SERVICE_CODE(msgcode) == IMAGE_CLEAN_SERVICE_WITH_BM3D) {
			// Real service ...
			time_reset();
			output_tensor = bm3d_filter(input_tensor, SERVICE_ARGUMENT(msgcode));
			time_spend((char *) "BM3D cleaning");

			service_response(socket, IMAGE_CLEAN_SERVICE_WITH_BM3D, output_tensor);
			tensor_destroy(output_tensor);
		} else if (SERVICE_CODE(msgcode) == IMAGE_CLEAN_SERVICE_WITH_DEHAZE) {
			// Real service ...
			time_reset();
			output_tensor = haze_filter(input_tensor, SERVICE_ARGUMENT(msgcode));
			time_spend((char *) "Haze cleaning");

			service_response(socket, IMAGE_CLEAN_SERVICE_WITH_DEHAZE, output_tensor);
			tensor_destroy(output_tensor);
		} else {
			// service_response(socket, servicecode, input_tensor)
			custom_service_function(socket, OUTOF_SERVICE, NULL);
		}
		count++;

		tensor_destroy(input_tensor);
	}
	StopEngine(engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

// Image clean model input: 1x3x(-1)x(-1), output 1x3x(-1)x(-1)
int server(char *endpoint, int use_gpu)
{
	InitEngineRunningTime();	// aviod compiler compaint

	return CleanOnnxService(endpoint, (char *)"image_clean.onnx", use_gpu, NULL);
}

int image_clean(int socket, char *input_file)
{
	IMAGE *send_image;
	TENSOR *send_tensor, *recv_tensor;

	printf("Cleaning %s ...\n", input_file);

	send_image = image_load(input_file); check_image(send_image);

	if (image_valid(send_image)) {
		send_tensor = tensor_from_image(send_image, 0);
		check_tensor(send_tensor);

		// Clean server limited: only accept 4 times tensor !!!
		recv_tensor = ResizeOnnxRPC(socket, send_tensor, IMAGE_CLEAN_SERVICE, 4);
		if (tensor_valid(recv_tensor)) {
			SaveTensorAsImage(recv_tensor, input_file);
			tensor_destroy(recv_tensor);
		}

		tensor_destroy(send_tensor);
		image_destroy(send_image);
	}

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
	char *endpoint = (char *) IMAGE_CLEAN_URL;

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

	if (! CudaAvailable() && use_gpu) {
		syslog_info("Cuda is not aviable, so running on CPU.");
		use_gpu = 0;
	}

	if (running_server) {
		if (IsRunning(argv[0])) {
			syslog_error("Service is running ...");
			exit(-1);
		}
		return server(endpoint, use_gpu);
	}
	else if (argc > 1) {
		if ((socket = client_open(endpoint)) < 0)
			return RET_ERROR;

		for (i = optind; i < argc; i++)
			image_clean(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	help(argv[0]);

	return RET_ERROR;
}
