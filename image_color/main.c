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
#define IMAGE_COLOR_URL "ipc:///tmp/image_color.ipc"

TENSOR *color_normlab(IMAGE * image)
{
	int i, j;
	TENSOR *tensor;
	float *R, *G, *B, *A, L, a, b;

	CHECK_IMAGE(image);

	tensor = tensor_create(1, sizeof(RGBA_8888), image->height, image->width);
	CHECK_TENSOR(tensor);

	R = tensor->data;
	G = R + tensor->height * tensor->width;
	B = G + tensor->height * tensor->width;
	A = B + tensor->height * tensor->width;

	image_foreach(image, i, j) {
		color_rgb2lab(image->ie[i][j].r, image->ie[i][j].g, image->ie[i][j].b, &L, &a, &b);
		L = (L - 50.f)/100.f; a /= 110.f; b /= 110.f;
		CheckPoint("L=%.4f, a = %.4f, b = %.4f, R=%d, G=%d, B = %d", 
			L, a, b, image->ie[i][j].r, image->ie[i][j].g, image->ie[i][j].b);

		*R++ = L; 
		*G++ = 0.f; *B++ = 0.f; *A++ = -0.5f;
		// if (image->ie[i][j].a > 0) {
		// 	*G++ = a; *B++ = b; *A++ = 0.5f;
		// } else {
		// 	*G++ = 0.f; *B++ = 0.f; *A++ = -0.5f;
		// }
	}

	return tensor;
}

int blend_fake(TENSOR *source, TENSOR *fake_ab)
{
	int i, j;
	float *source_L, *source_a, *source_b, *source_m;
	float *fake_a, *fake_b;
	float L, a, b;
	BYTE R, G, B;

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

	for (i = 0; i < source->height; i++) {
		source_L = tensor_startrow(source, 0 /*batch*/, 0 /*channel*/, i);
		source_a = tensor_startrow(source, 0 /*batch*/, 1 /*channel*/, i);
		source_b = tensor_startrow(source, 0 /*batch*/, 2 /*channel*/, i);
		source_m = tensor_startrow(source, 0 /*batch*/, 3 /*channel*/, i);

		fake_a = tensor_startrow(fake_ab, 0 /*batch*/, 0 /*channel*/, i);
		fake_b = tensor_startrow(fake_ab, 0 /*batch*/, 1 /*channel*/, i);

		for (j = 0; j < source->width; j++) {
			L = *source_L; a = *fake_a; b = *fake_b;
			L = L*100.f + 50.f; a *= 110.f; b *= 110.f;
			color_lab2rgb(L, a, b, &R, &G, &B);
			// CheckPoint("L=%.4f, a = %.4f, b = %.4f, R=%d, G=%d, B = %d", L, a, b, R, G, B);

			L = (float)R/255.f;
			a = (float)G/255.f;
			b = (float)B/255.f;

			*source_L++ = L;
			*source_a++ = a;
			*source_b++ = b;
			*source_m++ = 0.f;
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
	char output_filename[256];
	IMAGE *image = image_from_tensor(recv_tensor, 0);
	if (image_valid(image)) {
		snprintf(output_filename, sizeof(output_filename) - 1, "/tmp/%s", filename);
		image_save(image, output_filename);
		image_destroy(image);
	}
}

int color(int socket, char *input_file)
{
	int rescode;
	IMAGE *send_image;
	TENSOR *send_tensor, *recv_tensor;

	send_image = image_load(input_file); check_image(send_image);

	if (image_valid(send_image)) {
		send_tensor = color_normlab(send_image);
		check_tensor(send_tensor);

		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_COLOR_REQCODE, &rescode);
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
