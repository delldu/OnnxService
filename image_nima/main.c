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
const char *model_onnx = "image_nima.onnx";

void dump(TENSOR *recv_tensor, int rescode)
{
	// dump scores ...
	IMAGE *image = image_from_tensor(recv_tensor, 0);
	if (image_valid(image)) {
		image_save(image, "output.png");
		image_destroy(image);
	}
}

int server(char *endpoint)
{
	int socket, reqcode, count, rescode;
	float option;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *engine;

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	engine = CreateEngine(model_onnx); CheckEngine(engine);

	count = 0;
	for (;;) {
		syslog_info("Service %d times", count);

		input_tensor = request_recv(socket, &reqcode, &option);

		if (!tensor_valid(input_tensor)) {
			syslog_error("Request recv bad tensor ...");
			continue;
		}
		syslog_info("Request Code = %d, Option = %f", reqcode, option);

		// Real service ...
		output_tensor = TensorForward(engine, input_tensor);

		if (tensor_valid(output_tensor)) {
			rescode = reqcode;
			response_send(socket, output_tensor, rescode);
			tensor_destroy(output_tensor);
		}
		
		tensor_destroy(input_tensor);

		count++;
	}
	DestroyEngine(engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

int client(char *endpoint, char *input_file)
{
	int ret, rescode, socket;
	IMAGE *send_image;
	TENSOR *send_tensor, *recv_tensor;

	if ((socket = client_open(endpoint)) < 0)
		return RET_ERROR;

	ret = RET_ERROR;
	send_image = image_load(input_file);
	if (!image_valid(send_image))
		goto finish;

	send_tensor = tensor_from_image(send_image);
	check_tensor(send_tensor);

	if (tensor_valid(send_tensor)) {
		// Send
		ret = request_send(socket, 6789, send_tensor, 3.14f);

		if (ret == RET_OK) {
			// Recv
			recv_tensor = response_recv(socket, &rescode);
			if (tensor_valid(recv_tensor)) {
				// Process recv tensor ...
				dump(recv_tensor, rescode);
				tensor_destroy(recv_tensor);
			}
		}

		tensor_destroy(send_tensor);
	}
	image_destroy(send_image);

finish:
	client_close(socket);

	return ret;
}

void help(char *cmd)
{
	printf("Usage: %s [option]\n", cmd);
	printf("    h, --help                   Display this help.\n");
	printf("    e, --endpoint               Set endpoint.\n");
	printf("    s, --server                 Start server.\n");
	printf("    c, --client <file>          Client image.\n");

	exit(1);
}

int main(int argc, char **argv)
{
	int optc;
	int running_server = 0;

	int option_index = 0;
	char *client_file = NULL;
	char *endpoint = (char *)URL;

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
