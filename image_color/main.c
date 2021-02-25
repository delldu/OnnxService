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

#define IMAGE_COLOR_REQCODE 0x0102
// #define IMAGE_COLOR_URL "ipc:///tmp/image_color.ipc"
#define IMAGE_COLOR_URL "tcp://127.0.0.1:9102"

TENSOR *color_onnxrpc(int socket, TENSOR *send_tensor)
{
	int nh, nw, rescode;
	TENSOR *resize_send, *resize_recv, *recv_tensor;

	CHECK_TENSOR(send_tensor);

	// Color server limited: max 512, only accept 8 times !!!
	resize(send_tensor->height, send_tensor->width, 512, 8, &nh, &nw);
	if (send_tensor->height == nh && send_tensor->width == nw) {
		// Normal onnx RPC
		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_COLOR_REQCODE, &rescode);
	} else {
		resize_send = tensor_zoom(send_tensor, nh, nw); CHECK_TENSOR(resize_send);
		resize_recv = OnnxRPC(socket, resize_send, IMAGE_COLOR_REQCODE, &rescode);
		recv_tensor = tensor_zoom(resize_recv, send_tensor->height, send_tensor->width);
		tensor_destroy(resize_recv);
		tensor_destroy(resize_send);
	}

	return recv_tensor;
}

TENSOR *color_normlab(IMAGE * image)
{
	TENSOR *tensor;

	CHECK_IMAGE(image);
	tensor = tensor_rgb2lab(image); CHECK_TENSOR(tensor);
	tensor_setmask(tensor, 1.0);

	return tensor;
}

int blend_fake(TENSOR *source, TENSOR *fake_ab)
{
	int i, j;
	float *source_a, *source_b;
	float *fake_a, *fake_b;

	check_tensor(source);	// 1x4xHxW
	check_tensor(fake_ab);	// 1x2xHxW

	if (source->batch != 1 || source->chan != 4 || fake_ab->batch != 1 || fake_ab->chan != 2) {
		syslog_error("Bad source or fake_ab tensor.");
		return RET_ERROR;
	}
	if (source->height != fake_ab->height || source->width != fake_ab->width) {
		syslog_error("Source tensor size is not same as fake_ab.");
		return RET_ERROR;
	}

	source_a = tensor_start_chan(source, 0 /*batch*/, 1 /*channel*/);
	source_b = tensor_start_chan(source, 0 /*batch*/, 2 /*channel*/);

	fake_a = tensor_start_chan(fake_ab, 0 /*batch*/, 0 /*channel*/);
	fake_b = tensor_start_chan(fake_ab, 0 /*batch*/, 1 /*channel*/);

	for (i = 0; i < source->height; i++) {
		for (j = 0; j < source->width; j++) {
			*source_a++ = *fake_a++;
			*source_b++ = *fake_b++;
		}
	}
	return RET_OK;
}

int server(char *endpoint, int use_gpu)
{
	return OnnxService(endpoint, (char *)"image_color.onnx", use_gpu);
}

void dump(TENSOR * recv_tensor, char *filename)
{
	char output_filename[256], *p;

	IMAGE *image = tensor_lab2rgb(recv_tensor, 0);

	if (image_valid(image)) {
		p = strrchr(filename, '/');
	 	p = (! p)? filename : p + 1;
		snprintf(output_filename, sizeof(output_filename) - 1, "/tmp/%s", p);
		image_save(image, output_filename);
		image_destroy(image);
	}
}

int color(int socket, char *input_file)
{
	IMAGE *send_image;
	TENSOR *send_tensor, *recv_tensor;

	printf("Coloring %s ...\n", input_file);

	send_image = image_load(input_file); check_image(send_image);
	// color_togray(send_image);

	if (image_valid(send_image)) {
		send_tensor = color_normlab(send_image);
		check_tensor(send_tensor);

		recv_tensor = color_onnxrpc(socket, send_tensor);
		if (tensor_valid(recv_tensor)) {
			blend_fake(send_tensor, recv_tensor);
			dump(send_tensor, input_file);
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
	char *endpoint = (char *) IMAGE_COLOR_URL;

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
			color(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	help(argv[0]);

	return RET_ERROR;
}
