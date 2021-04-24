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

#define VIDEO_CLEAN_SERVICE 0x0201
// #define VIDEO_CLEAN_URL "ipc:///tmp/video_clean.ipc"
#define VIDEO_CLEAN_URL "tcp://127.0.0.1:9201"

// Patch model input: 1 x 16 x (-1) x (-1), 1 x 3 x (-1) x (-1)
int ColorService(char *endpoint, int use_gpu)
{
	int socket, count;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *clean_engine = NULL;

	srand(time(NULL));

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	count = 0;
	for (;;) {
		if (EngineIsIdle())
			StopEngine(clean_engine);

		if (! socket_readable(socket, 1000))	// timeout 1 s
			continue;

		input_tensor = service_request(socket, VIDEO_CLEAN_SERVICE);
		if (!tensor_valid(input_tensor))
			continue;

		syslog_info("Service %d times", count);
		StartEngine(clean_engine, (char *)"video_clean.onnx", use_gpu);

		// Real service ...
		time_reset();
		output_tensor = TensorForward(clean_engine, input_tensor);
		time_spend((char *)"Video cleaning");

		service_response(socket, VIDEO_CLEAN_SERVICE, output_tensor);
		tensor_destroy(output_tensor);

		tensor_destroy(input_tensor);

		count++;
	}
	StopEngine(clean_engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

int server(char *endpoint, int use_gpu)
{
	return ColorService(endpoint, use_gpu);
}

TENSOR *clean_load(int n, char *filenames[])
{
	int i, len;
	IMAGE *image;
	TENSOR *tensors[6], *output;

	if (n < 5) {
		syslog_error("At least 5 frames at same time.");
		return NULL;
	}

	for (i = 0; i < 5; i++) {
		image = image_load(filenames[i]);
		CHECK_IMAGE(image);
		tensors[i] = tensor_from_image(image, 0 /* without alpha */);
		CHECK_TENSOR(tensors[i]);
	}

	// Make noise
	tensors[5] = tensor_create(1, 1, tensors[0]->height, tensors[0]->width);
	len = 1 * 1 * tensors[0]->height * tensors[0]->width;
	for (i = 0; i < len; i++) {
		tensors[5]->data[i] = 0.05;
	}

	output = tensor_stack_chan(6, tensors);

	for (i = 0; i < 6; i++)
		tensor_destroy(tensors[i]);

	return output;
}

int clean_save(TENSOR *tensor, int index)
{
	IMAGE *image;
	char filename[256];

	image = image_from_tensor(tensor, 0);
	snprintf(filename, sizeof(filename), "output/%06d.png", index);
	image_save(image, filename);
	image_destroy(image);

	return RET_OK;
}

void help(char *cmd)
{
	printf("Usage: %s [option] <images>\n", cmd);
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
	TENSOR *send_tensor, *recv_tensor;

	int option_index = 0;
	char *endpoint = (char *) VIDEO_CLEAN_URL;

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

		for (i = optind; i + 4 < argc; i++) {
			printf("Video cleaning file %s ...\n", argv[i]);

			send_tensor = clean_load(5, &argv[i]);
			if (tensor_valid(send_tensor)) {
				recv_tensor = OnnxRPC(socket, send_tensor, VIDEO_CLEAN_SERVICE);
				clean_save(recv_tensor, i - optind + 1);
				tensor_destroy(recv_tensor);
				tensor_destroy(send_tensor);
			}
		}

		client_close(socket);
		return RET_OK;
	}

	return RET_ERROR;
}
