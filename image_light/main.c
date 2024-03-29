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
#include "model.h"

int clahe(int socket, int service_code, TENSOR *input_tensor)
{
	IMAGE *image;
	TENSOR *output_tensor;

	if (service_code == IMAGE_LIGHT_SERVICE_WITH_CLAHE) {
		// input_tensor is OK
		image = image_from_tensor(input_tensor, 0); check_image(image);
		image_clahe(image, 4, 4, 4.0);
		output_tensor = tensor_from_image(image, (input_tensor->chan == 4 || input_tensor->chan == 2)?1 : 0);
		image_destroy(image);

		service_response(socket, IMAGE_LIGHT_SERVICE_WITH_CLAHE, output_tensor);
		tensor_destroy(output_tensor);
	} else {
		service_response(socket, OUTOF_SERVICE_MESSAGE, NULL);
	}
	return RET_OK;
}


int server(char *endpoint, int use_gpu)
{
	InitEngineRunningTime();	// aviod compiler compaint
	
	// Lighting model input: 1x3x(-1)x(-1), output 1x3x(-1)x(-1)
	return OnnxServiceFromArray(endpoint, image_light_onnx, image_light_onnx_len, IMAGE_LIGHT_SERVICE, use_gpu, clahe);
}

int light(int socket, char *input_file)
{
	IMAGE *send_image;
	TENSOR *send_tensor, *recv_tensor;

	printf("Lighting %s ...\n", input_file);

	send_image = image_load(input_file); check_image(send_image);

	if (image_valid(send_image)) {
		send_tensor = tensor_from_image(send_image, 0);
		check_tensor(send_tensor);

		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_LIGHT_SERVICE);
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
	char *endpoint = (char *) IMAGE_LIGHT_URL;

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

	if (running_server) {
		if (IsRunning(endpoint))
			exit(-1);
		return server(endpoint, use_gpu);
	}
	else if (argc > 1) {
		if ((socket = client_open(endpoint)) < 0)
			return RET_ERROR;

		for (i = optind; i < argc; i++)
			light(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	help(argv[0]);

	return RET_ERROR;
}
