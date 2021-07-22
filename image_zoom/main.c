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

// Patch model input: 1 x 3 x (-1) x (-1), 1 x 3 x (-1) x (-1)
int server(char *endpoint, int use_gpu)
{
	int socket, count, msgcode;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *engine = NULL;
	OrtEngine *pan_engine = NULL;

	InitEngineRunningTime();	// aviod compiler compaint

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	count = 0;
	for (;;) {
		if (EngineIsIdle()) {
			StopEngine(engine);
			StopEngine(pan_engine);
		}

		if (! socket_readable(socket, 1000))	// timeout 1 s
			continue;

		input_tensor = service_request(socket, &msgcode);
		if (! tensor_valid(input_tensor))
			continue;

		if (msgcode == IMAGE_ZOOM_SERVICE) {
			syslog_info("Service %d times", count);
			StartEngine(engine, (char *)"image_zoom.onnx", use_gpu);

			// Real service ...
			time_reset();
			output_tensor = TensorForward(engine, input_tensor);
			time_spend((char *)"Image zoom4x");

			service_response(socket, IMAGE_ZOOM_SERVICE, output_tensor);
			tensor_destroy(output_tensor);

			count++;
		} else if (msgcode == IMAGE_ZOOM_SERVICE_WITH_PAN) {
			syslog_info("Service %d times", count);
			StartEngine(pan_engine, (char *)"image_zooms.onnx", use_gpu);

			// Real service ...
			time_reset();
			output_tensor = TensorForward(pan_engine, input_tensor);
			time_spend((char *)"Image zoom4x with PAN");

			service_response(socket, IMAGE_ZOOM_SERVICE_WITH_PAN, output_tensor);
			tensor_destroy(output_tensor);

			count++;
		} else {
			service_response(socket, OUTOF_SERVICE_MESSAGE, NULL);
		}

		tensor_destroy(input_tensor);
	}
	StopEngine(engine);
	StopEngine(pan_engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

int zoom(int socket, char *input_file)
{
	IMAGE *send_image;
	TENSOR *send_tensor, *recv_tensor;

	printf("Zooming %s ...\n", input_file);

	send_image = image_load(input_file); check_image(send_image);

	if (image_valid(send_image)) {
		send_tensor = tensor_from_image(send_image, 0);
		check_tensor(send_tensor);

		// Default with Zoom PAN for memory saving
		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_ZOOM_SERVICE_WITH_PAN);
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
	printf("    s, --server <1 | 0>         Start server (use gpu).\n");

	exit(1);
}

int main(int argc, char **argv)
{
	int i, optc;
	int use_gpu = 1;
	int running_server = 0;
	int socket;

	int option_index = 0;
	char *endpoint = (char *) IMAGE_ZOOM_URL;

	struct option long_opts[] = {
		{"help", 0, 0, 'h'},
		{"endpoint", 1, 0, 'e'},
		{"server", 1, 0, 's'},
		{0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);

	while ((optc = getopt_long(argc, argv, "h e: s:" , long_opts, &option_index)) != EOF) {
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

	if (running_server) {
		if (IsRunning(endpoint))
			exit(-1);
		return server(endpoint, use_gpu);
	}
	else if (argc > 1) {
		if ((socket = client_open(endpoint)) < 0)
			return RET_ERROR;

		for (i = optind; i < argc; i++)
			zoom(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	help(argv[0]);

	return RET_ERROR;
}
