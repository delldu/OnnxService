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

#define VIDEO_COLOR_SERVICE 0x0202
#define VIDEO_REFERENCE_SERVICE 0x0212
// #define VIDEO_COLOR_URL "ipc:///tmp/video_color.ipc"
#define VIDEO_COLOR_URL "tcp://127.0.0.1:9202"
TENSOR *reference_rgb512_tensor = NULL;
TENSOR *reference_lab512_tensor = NULL;
TENSOR *last_lab512_tensor = NULL;

// input: rgb,  r, g, b in [0.0, 1.0]
// output: lab, L in [-50, 50], ab in [-110, 110]
TENSOR *rgb2lab(TENSOR *rgb)
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

	lab = tensor_create(rgb->batch, 3, rgb->height, rgb->width);
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
			*LC = fL; *aC = fa; *bC = fb;

			RC++; GC++; BC++;
			LC++; aC++; bC++;
		}
	}

	return lab;	// L in [-50, 50], ab in [-110, 110]
}

// input: lab, L in [-50, 50], ab in [-110, 110]
// output: rgb,  r, g, b in [0.0, 1.0]
TENSOR *lab2rgb(TENSOR *lab)
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
			fL = *LC; fa = *aC; fb = *bC; fL += 50.0;
			color_lab2rgb(fL, fa, fb, &R, &G, &B);
			*RC = (float)R/255.0; *GC = (float)G/255.0; *BC = (float)B/255.0;
			LC++; aC++; bC++;
			RC++; GC++; BC++;
		}
	}

	return rgb;
}

TENSOR *color_do(OrtEngine *align_engine, OrtEngine *color_engine, TENSOR *input_tensor)
{
	TENSOR *array[8];
	TENSOR *output_tensor, *input_lab512_tensor, *input_lab512_tensor_L;

	CHECK_TENSOR(input_tensor);
	CheckEngine(align_engine);
	CheckEngine(color_engine);

	TENSOR *input_rgb512_tensor = tensor_zoom(input_tensor, 512, 512);
	CHECK_TENSOR(input_rgb512_tensor);
	input_lab512_tensor = rgb2lab(input_rgb512_tensor);
	CHECK_TENSOR(input_lab512_tensor);
	input_lab512_tensor_L = tensor_slice_chan(input_lab512_tensor, 0, 1);
	CHECK_TENSOR(input_lab512_tensor_L);

    // align_input = torch.cat((B_lab, lab2rgb(A_lab),lab2rgb(B_lab)), dim=1)
    // align_output = align_model(align_input)
	array[0] = reference_lab512_tensor;
	array[1] = input_rgb512_tensor;
	array[2] = reference_rgb512_tensor;
	TENSOR *align_input = tensor_stack_chan(3, array);
	CHECK_TENSOR(align_input);
	tensor_destroy(input_rgb512_tensor);

	TENSOR *align_output = TensorForward(align_engine, align_input);
	CHECK_TENSOR(align_output);
	tensor_destroy(align_input);

    // color_input = torch.cat((A_lab[:, 0:1, :, :], global_lab[:, 1:3, :, :], similarity, A_last_lab), dim=1)
    // color_output = color_model(color_input)
	TENSOR *global_ab = tensor_slice_chan(align_output, 1, 3);	// 2 channels
	CHECK_TENSOR(global_ab);
	TENSOR *similarity = tensor_slice_chan(align_output, 3, 4);	// 1 channel
	CHECK_TENSOR(similarity);
	tensor_destroy(align_output);

	array[0] = input_lab512_tensor_L;
	array[1] = global_ab;
	array[2] = similarity;
	array[3] = last_lab512_tensor;
	TENSOR *color_input = tensor_stack_chan(4, array);
	CHECK_TENSOR(color_input);

	TENSOR *color_output = TensorForward(color_engine, color_input);

	CHECK_TENSOR(color_output);		// only ab channels
	tensor_destroy(global_ab);
	tensor_destroy(similarity);
	tensor_destroy(color_input);

    // current_ab_predict = double_size(color_output)
    // predict_rgb = lab2rgb(torch.cat((current_lab[:, 0:1, :, :], current_ab_predict), dim = 1))
	TENSOR *predict_ab = tensor_zoom(color_output, input_tensor->height, input_tensor->width);
	CHECK_TENSOR(predict_ab);
	TENSOR *current_lab = rgb2lab(input_tensor);
	CHECK_TENSOR(current_lab);

	TENSOR *current_lab_l = tensor_slice_chan(current_lab, 0, 1);
	CHECK_TENSOR(current_lab_l);
	array[0] = current_lab_l;
	array[1] = predict_ab;
	output_tensor = tensor_stack_chan(2, array);

	tensor_destroy(current_lab_l);
	tensor_destroy(current_lab);
	tensor_destroy(predict_ab);

    // last_lab512_tensor = torch.cat((A_lab[:, 0:1, :, :], color_output), dim=1)
	tensor_destroy(last_lab512_tensor);
	array[0] = input_lab512_tensor_L;
	array[1] = color_output;
	last_lab512_tensor = tensor_stack_chan(2, array);
	CHECK_TENSOR(last_lab512_tensor);

	tensor_destroy(color_output);
	tensor_destroy(input_lab512_tensor_L);
	tensor_destroy(input_rgb512_tensor);

	TENSOR *output_rgb_tensor = lab2rgb(output_tensor);
	tensor_destroy(output_tensor);

	return output_rgb_tensor;
}

// Video color model
// 
//	 video_align.onnx
//	 	input: 1 x 9 x 512 x 512
//	 	Output: 1 x 4 x 512 x 512
//	 video_color.onnx
//	 	Input: 1 x 7 x 512 x 512
//	 	Output: 1 x 2 x 512 x 512
//

int ColorService(char *endpoint, int use_gpu)
{
	int socket, reqcode, count;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *color_engine = NULL;
	OrtEngine *align_engine = NULL;

	srand(time(NULL));

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	count = 0;
	for (;;) {
		if (EngineIsIdle()) {
			StopEngine(align_engine);
			StopEngine(color_engine);
		}

		if (! socket_readable(socket, 1000))	// timeout 1 s
			continue;

		input_tensor = service_request_withcode(socket, &reqcode);
		if (!tensor_valid(input_tensor))
			continue;

		syslog_info("Service %d times", count);

		StartEngine(align_engine, (char *)"video_align.onnx", use_gpu);
		StartEngine(color_engine, (char *)"video_color.onnx", use_gpu);
		UpdateEngineRunningTime();

		if (reqcode == VIDEO_REFERENCE_SERVICE) {
			// Save tensor to global reference tensor
			tensor_destroy(reference_rgb512_tensor);
			reference_rgb512_tensor = tensor_zoom(input_tensor, 512, 512);

			tensor_destroy(reference_lab512_tensor);
			reference_lab512_tensor = rgb2lab(reference_rgb512_tensor);

			// Init last_lab512_tensor
			tensor_destroy(last_lab512_tensor);
			last_lab512_tensor = tensor_create(1, 3, 512, 512);
			check_tensor(last_lab512_tensor);
			tensor_zero(last_lab512_tensor);

			// Respone echo input_tensor ...
			tensor_send(socket, VIDEO_REFERENCE_SERVICE, input_tensor);

			// Next for service ...
			tensor_destroy(input_tensor);
			continue;
		}

		// Real service ...
		time_reset();
		output_tensor = color_do(align_engine, color_engine, input_tensor);
		time_spend((char *)"Video coloring");

		service_response(socket, VIDEO_COLOR_SERVICE, output_tensor);
		tensor_destroy(output_tensor);

		tensor_destroy(input_tensor);

		count++;
	}

	tensor_destroy(reference_lab512_tensor);
	tensor_destroy(reference_rgb512_tensor);

	StopEngine(align_engine);
	StopEngine(color_engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

int server(char *endpoint, int use_gpu)
{
	return ColorService(endpoint, use_gpu);
}

TENSOR *color_load(char *filename)
{
	IMAGE *image;
	TENSOR *tensor;

	image = image_load(filename); CHECK_IMAGE(image);
	tensor = tensor_from_image(image, 0 /* without alpha */);
	image_destroy(image);

	return tensor;
}

int color_save(TENSOR *tensor, int index)
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
	printf("Usage: %s [option] <reference color gray images>\n", cmd);
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
	int reqcode;
	TENSOR *send_tensor, *recv_tensor;

	int option_index = 0;
	char *endpoint = (char *) VIDEO_COLOR_URL;

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

		for (i = optind; i < argc; i++) {
			if (i == optind)
				printf("Video send reference file %s ...\n", argv[i]);
			else
				printf("Video coloring file %s ...\n", argv[i]);

			reqcode = (i == optind)? VIDEO_REFERENCE_SERVICE : VIDEO_COLOR_SERVICE;
			send_tensor = color_load(argv[i]);

			if (tensor_valid(send_tensor)) {
				recv_tensor = OnnxRPC(socket, send_tensor, reqcode);
				if (i > optind && tensor_valid(recv_tensor))
					color_save(recv_tensor, i - optind);
				tensor_destroy(recv_tensor);
				tensor_destroy(send_tensor);
			}
		}

		client_close(socket);
		return RET_OK;
	}

	return RET_ERROR;
}
