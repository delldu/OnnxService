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

#define VIDEO_COLOR_REQCODE 0x0202
// #define VIDEO_COLOR_URL "ipc:///tmp/video_color.ipc"
#define VIDEO_COLOR_URL "tcp://127.0.0.1:9202"


TENSOR *color_do(OrtEngine *vgg19, OrtEngine *warp, OrtEngine *color, TENSOR *input_tensor)
{
	TENSOR *output_tensor = NULL;

	return output_tensor;
}

int ColorService(char *endpoint, int use_gpu)
{
	int socket, reqcode, lambda, rescode;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *color_engine = NULL;
	OrtEngine *vgg19_engine = NULL;
	OrtEngine *warp_engine = NULL;

	srand(time(NULL));

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	vgg19_engine = CreateEngine("video_vgg19.onnx", use_gpu /*use_gpu*/);
	CheckEngine(vgg19_engine);

	warp_engine = CreateEngine("video_warp.onnx", use_gpu /*use_gpu*/);
	CheckEngine(vgg19_engine);

	color_engine = CreateEngine("video_color.onnx", use_gpu /*use_gpu*/);
	CheckEngine(color_engine);

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
		output_tensor = color_do(vgg19_engine, warp_engine, color_engine, input_tensor);
		time_spend((char *)"Infer");

		if (tensor_valid(output_tensor)) {
			rescode = reqcode;
			response_send(socket, output_tensor, rescode);
			tensor_destroy(output_tensor);
		}

		tensor_destroy(input_tensor);

		lambda++;
	}

	DestroyEngine(color_engine);
	DestroyEngine(warp_engine);
	DestroyEngine(vgg19_engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

int server(char *endpoint, int use_gpu)
{
	return ColorService(endpoint, use_gpu);
}

TENSOR *color_load(char *input_file1, char *input_file2)
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

		output = tensor_create(1, 6, image1->height, image1->width);
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

int color_save(TENSOR *tensor)
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


TENSOR *color_onnxrpc(int socket, TENSOR *send_tensor)
{
	int nh, nw, rescode;
	TENSOR *resize_send, *resize_recv, *recv_tensor;

	CHECK_TENSOR(send_tensor);

	// server limited: only accept 4 times tensor !!!
	nh = (send_tensor->height + 7)/8; nh *= 8;
	nw = (send_tensor->width + 7)/8; nw *= 8;

	if (send_tensor->height == nh && send_tensor->width == nw) {
		// Normal onnx RPC
		recv_tensor = OnnxRPC(socket, send_tensor, VIDEO_COLOR_REQCODE, &rescode);
	} else {
		resize_send = tensor_zoom(send_tensor, nh, nw); CHECK_TENSOR(resize_send);
		resize_recv = OnnxRPC(socket, resize_send, VIDEO_COLOR_REQCODE, &rescode);
		recv_tensor = tensor_zoom(resize_recv, send_tensor->height, send_tensor->width);
		tensor_destroy(resize_recv);
		tensor_destroy(resize_send);
	}

	return recv_tensor;
}


int color(int socket, char *input_file1, char *input_file2)
{
	TENSOR *send_tensor, *recv_tensor;

	printf("Video coloring between %s and %s ...\n", input_file1, input_file2);

	send_tensor = color_load(input_file1, input_file2);
	check_tensor(send_tensor);

	recv_tensor = color_onnxrpc(socket, send_tensor);
	if (tensor_valid(recv_tensor)) {
		color_save(recv_tensor);
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
	char *endpoint = (char *) VIDEO_COLOR_URL;

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
			color(socket, argv[i], argv[i + 1]);

		client_close(socket);
		return RET_OK;
	}

	return RET_ERROR;
}
