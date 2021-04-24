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

#define IMAGE_MATTING_SERVICE 0x0107
// #define IMAGE_MATTING_URL "ipc:///tmp/image_matting.ipc"
#define IMAGE_MATTING_URL "tcp://127.0.0.1:9107"

int server(char *endpoint, int use_gpu)
{
	// Matting model input: 4x3x(-1)x(-1), output 1x1x(-1)x(-1)
	InitEngineRunningTime(); // Avoid compiler complaint
	return OnnxService(endpoint, (char *)"image_matting.onnx", IMAGE_MATTING_SERVICE, use_gpu);
}

int normal_input(TENSOR *tensor)
{
	int i, j;
	float *tensor_R, *tensor_G, *tensor_B;

	check_tensor(tensor);

	// Normal ...
	tensor_R = tensor_start_chan(tensor, 0 /*batch*/, 0 /*channel */);
	tensor_G = tensor_start_chan(tensor, 0 /*batch*/, 1 /*channel */);
	tensor_B = tensor_start_chan(tensor, 0 /*batch*/, 2 /*channel */);
	for (i = 0; i < tensor->height; i++) {
		for (j = 0; j < tensor->width; j++) {
			*tensor_R = (*tensor_R - 0.485f)/0.229f;
			*tensor_G = (*tensor_G - 0.456f)/0.224f;
			*tensor_B = (*tensor_B - 0.406f)/0.225f;

			tensor_R++; tensor_G++; tensor_B++;
		}
	}

	// Change RGB To BGR
	// tensor_R = tensor_start_chan(tensor, 0 /*batch*/, 0 /*channel */);
	// tensor_B = tensor_start_chan(tensor, 0 /*batch*/, 2 /*channel */);
	// for (i = 0; i < tensor->height; i++) {
	// 	for (j = 0; j < tensor->width; j++) {
	// 		d = *tensor_B;
	// 		*tensor_B = *tensor_R;
	// 		*tensor_R = d;
	// 		tensor_R++; tensor_B++;
	// 	}
	// }

	return RET_OK;
}

int normal_output(TENSOR *tensor)
{
	int i, size;
	float *data, min, max, d;

	check_tensor(tensor);

    // ma = torch.max(d)
    // mi = torch.min(d)
    // dn = (d-mi)/(ma-mi)
	// return dn
	size = tensor->chan * tensor->height * tensor->width;
	data = tensor->data;
	min = max = *data++;
	for (i = 1; i < size; i++, data++) {
		if (*data > max)
			max = *data;
		if (*data < min)
			min = *data;
	}

	data = tensor->data;
	max = max - min;
	if (max > 1e-3) {
		for (i = 0; i < size; i++, data++) {
			d = *data - min;
			d /= max;
			*data = d;
		}
	}

	return RET_OK;
}


TENSOR *matting_onnxrpc(int socket, TENSOR *send_tensor)
{
	int nh, nw;
	TENSOR *resize_send, *resize_recv, *recv_tensor;

	CHECK_TENSOR(send_tensor);

	resize_send = resize_recv = recv_tensor = NULL;	// avoid compile complaint

	// Matting server limited: only accept 4 times tensor !!!
	nh = (send_tensor->height + 3)/4; nh *= 4;
	nw = (send_tensor->width + 3)/4; nw *= 4;
	
	// Limited memory !!!
	while(nh > 512)
		nh /= 2;
	while (nw > 512)
		nw /= 2;

	if (send_tensor->height == nh && send_tensor->width == nw) {
		// Normal onnx RPC
		normal_input(send_tensor);
		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_MATTING_SERVICE);
		normal_output(recv_tensor);
	} else {
		resize_send = tensor_zoom(send_tensor, nh, nw); CHECK_TENSOR(resize_send);

		normal_input(resize_send);
		resize_recv = OnnxRPC(socket, resize_send, IMAGE_MATTING_SERVICE);
		normal_output(resize_recv);

		recv_tensor = tensor_zoom(resize_recv, send_tensor->height, send_tensor->width);
		tensor_destroy(resize_recv);
		tensor_destroy(resize_send);
	}

	return recv_tensor;
}


int blend_mask(IMAGE *source_image, TENSOR *mask_tensor)
{
	int i, j;
	float *mask_A, alpha;

	check_image(source_image);
	check_tensor(mask_tensor);

	mask_A = tensor_start_chan(mask_tensor, 0 /* batch */, 0 /* channel */);
	image_foreach(source_image, i, j) {
		alpha = *mask_A;
		source_image->ie[i][j].r = (BYTE)(alpha * source_image->ie[i][j].r + (1 - alpha)*0);
		source_image->ie[i][j].g = (BYTE)(alpha * source_image->ie[i][j].g + (1 - alpha)*255);
		source_image->ie[i][j].b = (BYTE)(alpha * source_image->ie[i][j].b + (1 - alpha)*0);
		source_image->ie[i][j].a = 255;
		mask_A++;
	}

	return RET_OK;
}


int matting(int socket, char *input_file)
{
	IMAGE *send_image;
	TENSOR *send_tensor, *recv_tensor;

	printf("Matting %s ...\n", input_file);

	send_image = image_load(input_file); check_image(send_image);

	if (image_valid(send_image)) {
		send_tensor = tensor_from_image(send_image, 0);
		check_tensor(send_tensor);

		recv_tensor = matting_onnxrpc(socket, send_tensor);
		if (tensor_valid(recv_tensor)) {
			blend_mask(send_image, recv_tensor);
			SaveOutputImage(send_image, input_file);
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
	char *endpoint = (char *) IMAGE_MATTING_URL;

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
			matting(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	help(argv[0]);

	return RET_ERROR;
}
