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


int server(char *endpoint, int use_gpu)
{
	// Nima model input: 1 x 3 x 224 x 224, output 1 x 10
	InitEngineRunningTime();
	return OnnxService(endpoint, (char *)"image_nima.onnx", IMAGE_NIMA_SERVICE, use_gpu, NULL);
}

void dump(TENSOR * recv_tensor, char *filename)
{
	// dump scores ...
	int i;
	float *f, mean;
	f = recv_tensor->data;
	mean = 0.0;
	for (i = 0; i < 10; i++) {
		mean += (*f++) * (i + 1.0);
	}
	printf("%6.4f %s\n", mean, filename);
}

int nima(int socket, char *input_file)
{
	IMAGE *image, *send_image;
	TENSOR *send_tensor, *recv_tensor;

	image = image_load(input_file); check_image(image);
	send_image = image_zoom(image, 224, 224, 1); check_image(send_image);
	image_destroy(image);

	if (image_valid(send_image)) {
		send_tensor = tensor_from_image(send_image, 0);	// 1x3x244x244
		check_tensor(send_tensor);

		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_NIMA_SERVICE);
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
	char *endpoint = (char *) IMAGE_NIMA_URL;

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

		for (i = optind; i < argc; i++)
			nima(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	help(argv[0]);

	return RET_ERROR;
}
