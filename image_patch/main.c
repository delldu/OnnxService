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

#define IMAGE_PATCH_REQCODE 0x0104
// #define IMAGE_PATCH_URL "ipc:///tmp/image_patch.ipc"
#define IMAGE_PATCH_URL "tcp://127.0.0.1:9104"

TENSOR *resize_onnxrpc(int socket, TENSOR *send_tensor)
{
	int nh, nw, rescode;
	TENSOR *resize_send, *resize_recv, *recv_tensor;

	CHECK_TENSOR(send_tensor);

	// Color server limited: max 512, only accept 8 times !!!
	// resize(send_tensor->height, send_tensor->width, 512, 1, &nh, &nw);
	nh = send_tensor->height;
	nw = send_tensor->width;
	if (send_tensor->height == nh && send_tensor->width == nw) {
		// Normal onnx RPC
		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_PATCH_REQCODE, &rescode);
	} else {
		resize_send = tensor_zoom(send_tensor, nh, nw); CHECK_TENSOR(resize_send);
		resize_recv = OnnxRPC(socket, resize_send, IMAGE_PATCH_REQCODE, &rescode);
		recv_tensor = tensor_zoom(resize_recv, send_tensor->height, send_tensor->width);
		tensor_destroy(resize_recv);
		tensor_destroy(resize_send);
	}

	return recv_tensor;
}


int server(char *endpoint, int use_gpu)
{
	return OnnxService(endpoint, (char *)"image_patch.onnx", use_gpu);
}

void dump(TENSOR * recv_tensor, char *filename)
{
	// int i, j;
	char output_filename[256], *p;
	IMAGE *image = image_from_tensor(recv_tensor, 0);

	if (image_valid(image)) {
		p = strrchr(filename, '/');
	 	p = (! p)? filename : p + 1;
		snprintf(output_filename, sizeof(output_filename) - 1, "/tmp/%s", p);
		image_save(image, output_filename);
		image_destroy(image);
	}
}

int patch(int socket, char *input_file)
{
	IMAGE *send_image;
	TENSOR *send_tensor, *recv_tensor;

	send_image = image_load(input_file); check_image(send_image);

	if (image_valid(send_image)) {
		send_tensor = tensor_from_image(send_image, 1);	// with alpha
		check_tensor(send_tensor);

		recv_tensor = resize_onnxrpc(socket, send_tensor);
		if (tensor_valid(recv_tensor)) {
			dump(recv_tensor, input_file);
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
	char *endpoint = (char *) IMAGE_PATCH_URL;

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
			patch(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	help(argv[0]);

	return RET_ERROR;
}
