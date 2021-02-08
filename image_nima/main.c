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

#define URL "ipc:///tmp/image_nima.ipc"

int server(char *endpoint)
{
	return OnnxService(endpoint, (char *)"image_nima.onnx", 0 /*use_gpu*/);
}

void dump(TENSOR * recv_tensor, int rescode)
{
	// dump scores ...
	(void)rescode;

	IMAGE *image = image_from_tensor(recv_tensor, 0);
	if (image_valid(image)) {
		image_save(image, "output.png");
		image_destroy(image);
	}
}

int client(char *endpoint, char *input_file)
{
	int rescode, socket;
	IMAGE *send_image;
	TENSOR *send_tensor, *recv_tensor;

	if ((socket = client_open(endpoint)) < 0)
		return RET_ERROR;

	send_image = image_load(input_file);

	if (image_valid(send_image)) {
		send_tensor = tensor_from_image(send_image);
		check_tensor(send_tensor);

		recv_tensor = OnnxRPC(socket, send_tensor, 6789, 3.14f, &rescode);
		if (tensor_valid(recv_tensor)) {
			// Process recv tensor ...
			dump(recv_tensor, rescode);
			tensor_destroy(recv_tensor);
		}

		tensor_destroy(send_tensor);
		image_destroy(send_image);
	}

	client_close(socket);

	return RET_OK;
}

void help(char *cmd)
{
	printf("Usage: %s [option]\n", cmd);
	printf("    h, --help                   Display this help.\n");
	printf("    e, --endpoint               Set endpoint.\n");
	printf("    s, --server                 Start server.\n");
	printf("    c, --client <file>          Call service.\n");

	exit(1);
}

int main(int argc, char **argv)
{
	int optc;
	int running_server = 0;

	int option_index = 0;
	char *endpoint = (char *) URL;
	char *client_file = NULL;

	struct option long_opts[] = {
		{"help", 0, 0, 'h'},
		{"endpoint", 1, 0, 'e'},
		{"server", 0, 0, 's'},
		{"client", 1, 0, 'c'},
		{0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);

	while ((optc = getopt_long(argc, argv, "h e: s c:", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'e':
			endpoint = optarg;
			break;
		case 's':
			running_server = 1;
			break;
		case 'c':				// Client
			client_file = optarg;
			break;
		case 'h':				// help
		default:
			help(argv[0]);
			break;
		}
	}

	if (running_server)
		return server(endpoint);
	else if (client_file) {
		return client(endpoint, client_file);
	}

	help(argv[0]);

	return RET_ERROR;
}
