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

// For scale == 4:
// input_tensor  -- [2, 2, 512, 960]
// output_tensor -- [1, 3, 512, 960]
TENSOR *rife_do(OrtEngine *fc, TENSOR *input_tensor)
{
	CheckEngine(fc);
	CHECK_TENSOR(input_tensor);
	TENSOR *output_tensor;

	// Suppose_X: input is 1x6xHxW input !!!
	if (input_tensor->batch != 2 || input_tensor->chan != 3) {
		syslog_error("Now only support 2x3xHxW input tensor.");
		return NULL;
	}

    output_tensor = TensorForward(fc, input_tensor);
    CHECK_TENSOR(output_tensor);

	return output_tensor;
}

// Video rife model
//
// video_rife.onnx
// 		input: 2 x 3 x -1 x -1
// 		output: 1 x 3 x -1 x -1
int RifeService(char *endpoint, int use_gpu)
{
	int socket, count, msgcode;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *rife_engine = NULL;

	srand(time(NULL));

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	count = 0;
	for (;;) {
		if (EngineIsIdle()) {
			StopEngine(rife_engine);
		}

		if (! socket_readable(socket, 1000))	// timeout 1 s
			continue;

		input_tensor = service_request(socket, &msgcode);
		if (!tensor_valid(input_tensor))
			continue;

		if (msgcode == VIDEO_RIFE_SERVICE) {
			syslog_info("Service %d times", count);

			StartEngine(rife_engine, (char *)"video_rife.onnx", use_gpu);

			// Real service ...
			time_reset();
			output_tensor = rife_do(rife_engine, input_tensor);
			time_spend((char *)"Video rife");

			service_response(socket, VIDEO_RIFE_SERVICE, output_tensor);
			tensor_destroy(output_tensor);

			tensor_destroy(input_tensor);

			count++;
		} else {
			// ...
		}
	}
	StopEngine(rife_engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

int server(char *endpoint, int use_gpu)
{
	return RifeService(endpoint, use_gpu);
}

TENSOR *rife_load(char *input_file1, char *input_file2)
{
	int n;
	float *to;

	IMAGE *image1, *image2;
	TENSOR *tensor1, *tensor2, *output = NULL;

	image1 = image_load(input_file1); CHECK_IMAGE(image1);
	image2 = image_load(input_file2); CHECK_IMAGE(image2);

	if (image1->height == image2->height && image1->width == image2->width) {
		tensor1 = tensor_from_image(image1, 0 /* without channel A */);	CHECK_TENSOR(tensor1);
		tensor2 = tensor_from_image(image2, 0 /* without channel A */);	CHECK_TENSOR(tensor2);

		output = tensor_create(2, 3, image1->height, image1->width);
		CHECK_TENSOR(output);
		n = output->height * output->width;
		to = output->data;
		memcpy(to, tensor1->data, 3 * n * sizeof(float));
		to = &output->data[3 * n];
		memcpy(to, tensor2->data, 3 * n * sizeof(float));

		tensor_destroy(tensor2);
		tensor_destroy(tensor1);
	} else {
		syslog_error("Image size is not same,");
	}

	image_destroy(image2);
	image_destroy(image1);

	return output;
}

int rife_save(TENSOR *tensor)
{
	int b;
	IMAGE *image;
	char filename[256];

	check_tensor(tensor);
	for (b = 0; b < tensor->batch; b++) {
		image = image_from_tensor(tensor, b);
		snprintf(filename, sizeof(filename), "output/%06d.png", b + 1);
		image_save(image, filename);
		image_destroy(image);
	}

	return RET_OK;
}

int rife(int socket, char *input_file1, char *input_file2)
{
	TENSOR *send_tensor, *recv_tensor;

	printf("Video rife between %s and %s ...\n", input_file1, input_file2);

	send_tensor = rife_load(input_file1, input_file2);
	check_tensor(send_tensor);

	// Server limited: only accept 8 times tensor !!!
	recv_tensor = ZeropadOnnxRPC(socket, send_tensor, VIDEO_RIFE_SERVICE, 8);
	if (tensor_valid(recv_tensor)) {
		rife_save(recv_tensor);
		tensor_destroy(recv_tensor);
	}

	tensor_destroy(send_tensor);

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
	char *endpoint = (char *) VIDEO_RIFE_URL;

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

		for (i = optind; i + 1 < argc; i++)
			rife(socket, argv[i], argv[i + 1]);

		client_close(socket);
		return RET_OK;
	}

	return RET_ERROR;
}
