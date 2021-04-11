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

// input: rgb,  r, g, b in [0.0, 1.0]
// output: lab, L in [-.50, 0.5], ab in [-1.0, 1.0]
TENSOR *image_rgb2lab(TENSOR *rgb)
{
	int i, n, b;
	TENSOR *lab;
	float *RC, *GC, *BC, *LC, *aC, *bC;	// Channels
	BYTE R, G, B;
	float fL, fa, fb;

	CHECK_TENSOR(rgb);
	if (rgb->chan < 3) {
		syslog_error("Tensor channels < 3");
		return NULL;
	}

	lab = tensor_create(rgb->batch, 4, rgb->height, rgb->width);
	CHECK_TENSOR(lab);
	for (b = 0; b < rgb->batch; b++) {
		RC = tensor_start_chan(rgb, b, 0);
		GC = tensor_start_chan(rgb, b, 1);
		BC = tensor_start_chan(rgb, b, 2);

		LC = tensor_start_chan(lab, b, 0);
		aC = tensor_start_chan(lab, b, 1);
		bC = tensor_start_chan(lab, b, 2);

		n = rgb->height * rgb->width;
		for (i = 0; i < n; i++) {
			// void color_rgb2lab(BYTE R, BYTE G, BYTE B, float *L, float *a, float *b);
			R =(BYTE)(*RC * 255.0); G =(BYTE)(*GC * 255.0); B =(BYTE)(*BC * 255.0); 
			color_rgb2lab(R, G, B, &fL, &fa, &fb);
			fL -= 50;
			*LC = fL/100.0; *aC = fa/110.0; *bC = fb/110.0;

			RC++; GC++; BC++;
			LC++; aC++; bC++;
		}
	}

	return lab;	// L in [-50, 50], ab in [-110, 110]
}

TENSOR *color_normlab(IMAGE * image)
{
	TENSOR *tensor, *output_lab_tensor;

	CHECK_IMAGE(image);
	tensor = tensor_from_image(image, 1 /*with alpha */);
	CHECK_TENSOR(tensor);

	output_lab_tensor = image_rgb2lab(tensor);
	CHECK_TENSOR(output_lab_tensor);
	tensor_destroy(tensor);

	tensor_setmask(output_lab_tensor, 1.0);

	return output_lab_tensor;
}

// input: lab, L in [-0.5, 0.5], ab in [-1.0, 1.0]
// output: rgb,  r, g, b in [0.0, 1.0]
TENSOR *image_lab2rgb(TENSOR *lab)
{
	int i, n, b;
	TENSOR *rgb;
	float *RC, *GC, *BC, *LC, *aC, *bC;	// Channels
	BYTE R, G, B;
	float fL, fa, fb;

	CHECK_TENSOR(lab);
	if (lab->chan < 3) {
		syslog_error("Tensor channels < 3");
		return NULL;
	}

	rgb = tensor_create(lab->batch, 3, lab->height, lab->width);
	CHECK_TENSOR(rgb);
	for (b = 0; b < rgb->batch; b++) {
		LC = tensor_start_chan(lab, b, 0);
		aC = tensor_start_chan(lab, b, 1);
		bC = tensor_start_chan(lab, b, 2);

		RC = tensor_start_chan(rgb, b, 0);
		GC = tensor_start_chan(rgb, b, 1);
		BC = tensor_start_chan(rgb, b, 2);

		n = lab->height * lab->width;
		for (i = 0; i < n; i++) {
			fL = *LC; fa = *aC; fb = *bC;
			fL += 0.5; fL *= 100.0;
			fa *= 110.0; fb *= 110.0;
			color_lab2rgb(fL, fa, fb, &R, &G, &B);
			*RC = (float)R/255.0; *GC = (float)G/255.0; *BC = (float)B/255.0;
			LC++; aC++; bC++;
			RC++; GC++; BC++;
		}
	}

	return rgb;
}


TENSOR *blend_fake(TENSOR *source, TENSOR *fake_ab)
{
	int i, j;
	float *source_a, *source_b;
	float *fake_a, *fake_b;

	CHECK_TENSOR(source);	// 1x4xHxW
	CHECK_TENSOR(fake_ab);	// 1x2xHxW

	if (source->batch != 1 || source->chan != 4 || fake_ab->batch != 1 || fake_ab->chan != 2) {
		syslog_error("Bad source or fake_ab tensor.");
		return NULL;
	}
	if (source->height != fake_ab->height || source->width != fake_ab->width) {
		TENSOR *resize_fake_ab = tensor_zoom(fake_ab, source->height, source->width);
		CHECK_TENSOR(resize_fake_ab);
		tensor_destroy(fake_ab);
		fake_ab = resize_fake_ab;
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

	// Need transform lab to rgb for output
	return image_lab2rgb(source);
}

int server(char *endpoint, int use_gpu)
{
	// image color model:
	// input:  lab with mask, l in [-0.5, 0.5], ab in [-1.0, 1.0], mask in [0, 1.0], and 1.0 is valid
	// output: ab
	return OnnxService(endpoint, (char *)"image_color.onnx", use_gpu);
}

int color(int socket, char *input_file)
{
	IMAGE *send_image;
	TENSOR *send_tensor, *recv_tensor, *ouput_rgb_tensor;

	printf("Coloring %s ...\n", input_file);

	send_image = image_load(input_file); check_image(send_image);
	// color_togray(send_image);

	if (image_valid(send_image)) {
		send_tensor = color_normlab(send_image);
		check_tensor(send_tensor);

		recv_tensor = ZeropadOnnxRPC(socket, send_tensor, IMAGE_COLOR_REQCODE, 8);
		if (tensor_valid(recv_tensor)) {
			ouput_rgb_tensor = blend_fake(send_tensor, recv_tensor);
			SaveTensorAsImage(ouput_rgb_tensor, input_file);
			tensor_destroy(ouput_rgb_tensor);

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

		for (i = optind; i < argc; i++)
			color(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	help(argv[0]);

	return RET_ERROR;
}
