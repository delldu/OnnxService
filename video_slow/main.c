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

#define VIDEO_SLOW_REQCODE 0x0203
// #define VIDEO_SLOW_URL "ipc:///tmp/image_facegan.ipc"
#define VIDEO_SLOW_URL "tcp://127.0.0.1:9203"

#define DEBUG 1

TENSOR *flow_backwarp(TENSOR *image, TENSOR *flow)
{
	return NULL;
}

// For scale == 4:
// input_tensor  -- [1, 6, 512, 960]
// output_tensor -- [1, 3, 512, 960] * 3
TENSOR *slow_do(OrtEngine *fc, OrtEngine *at, TENSOR *input_tensor, int scale)
{
	int i, j, n;
	float t, w[4], *to, a, b;
	TENSOR *I0, *I1, *flowOut, *F_0_1, *F_1_0, *F_t_0, *F_t_1, *output_tensor;

	CheckEngine(fc);
	CheckEngine(at);
	CHECK_TENSOR(input_tensor);

	// Suppose_X: input is 1x6xHxW input !!!
	if (input_tensor->batch != 1 || input_tensor->chan != 6) {
		syslog_error("NOW Only support 1x3xHxW input tensor.");
		return NULL;
	}

	I0 = tensor_slice_chan(input_tensor, 0, 3); CHECK_TENSOR(I0);
	I1 = tensor_slice_chan(input_tensor, 3, 6); CHECK_TENSOR(I1);

    flowOut = TensorForward(fc, input_tensor);
    CHECK_TENSOR(flowOut);
	// # flowOut.size() -- torch.Size([1, 4, 512, 960])

	// F_0_1 = flowOut[:, :2, :, :]
	// F_1_0 = flowOut[:, 2:, :, :]
	F_0_1 = tensor_slice_chan(flowOut, 0, 2); CHECK_TENSOR(F_0_1);
	F_1_0 = tensor_slice_chan(flowOut, 2, 4); CHECK_TENSOR(F_1_0);

    n = F_0_1->batch * F_0_1->chan * F_0_1->height * F_0_1->width;

	// Suppose_X
	F_t_0 = tensor_create(1, 2, input_tensor->height, input_tensor->width);
	CHECK_TENSOR(F_t_0);
	F_t_1 = tensor_create(1, 2, input_tensor->height, input_tensor->width);
	CHECK_TENSOR(F_t_1);

  	// Suppose_X
	output_tensor = tensor_create(1, (scale - 1) * input_tensor->chan, input_tensor->height, input_tensor->width);
	CHECK_TENSOR(output_tensor);

    for (j = 0; j < scale - 1; j++) {
	    t = 1.0 * j / scale;

        // temp = -t * (1 - t)
        // w = [temp, t * t, (1 - t) * (1 - t), temp]
	    w[0] = -t * (1.0 - t);
	    w[1] = t * t;
	    w[2] = (1.0 -t) * (1.0 - t);
	    w[3] = w[0];

	    // F_t_0 = w[0] * F_0_1 + w[1] * F_1_0
	    // F_t_1 = w[2] * F_0_1 + w[3] * F_1_0
	    for (i = 0; i < 2*n ; i++) {	// F_t_0, F_t_1 --2 Channels
	    	F_t_0->data[i] = w[0] * F_0_1->data[i] + w[1] * F_1_0->data[i];
	    	F_t_1->data[i] = w[2] * F_0_1->data[i] + w[3] * F_1_0->data[i];
	    }

        TENSOR *g_I0_F_t_0 = flow_backwarp(I0, F_t_0); CHECK_TENSOR(g_I0_F_t_0);
        TENSOR *g_I1_F_t_1 = flow_backwarp(I1, F_t_1); CHECK_TENSOR(g_I1_F_t_1);

	    // temp_interpolate_input = torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1);
	    // # (Pdb) temp_interpolate_input.size() -- torch.Size([1, 20, 512, 960])

	   	TENSOR *temp_interpolate_input = tensor_create(1, 20, input_tensor->height, input_tensor->width);
	   	CHECK_TENSOR(temp_interpolate_input);
	   	to = tensor_start_chan(temp_interpolate_input, 0, 0); memcpy(to, I0->data, 3 * n * sizeof(float));
	   	to = tensor_start_chan(temp_interpolate_input, 0, 3); memcpy(to, I1->data, 3 * n * sizeof(float));
	   	to = tensor_start_chan(temp_interpolate_input, 0, 6); memcpy(to, F_0_1->data, 2 * n * sizeof(float));
	   	to = tensor_start_chan(temp_interpolate_input, 0, 8); memcpy(to, F_1_0->data, 2 * n * sizeof(float));
	   	to = tensor_start_chan(temp_interpolate_input, 0, 10); memcpy(to, F_t_1->data, 2 * n * sizeof(float));
	   	to = tensor_start_chan(temp_interpolate_input, 0, 12); memcpy(to, F_t_0->data, 2 * n * sizeof(float));
	   	to = tensor_start_chan(temp_interpolate_input, 0, 14); memcpy(to, g_I1_F_t_1->data, 3 * n * sizeof(float));
	   	to = tensor_start_chan(temp_interpolate_input, 0, 17); memcpy(to, g_I0_F_t_0->data, 3 * n * sizeof(float));

        TENSOR *temp_interpolate_output = TensorForward(at, temp_interpolate_input);
        CHECK_TENSOR(temp_interpolate_output);
	    // # (Pdb) temp_interpolate_output.size() -- torch.Size([1, 5, 512, 960])
	    tensor_destroy(temp_interpolate_input);

	    tensor_destroy(g_I1_F_t_1);
	    tensor_destroy(g_I0_F_t_0);

        // F_t_0_f = temp_interpolate_output[:, 0:2, :, :] + F_t_0
        // F_t_1_f = temp_interpolate_output[:, 2:4, :, :] + F_t_1
	    TENSOR *F_t_0_f = tensor_slice_chan(temp_interpolate_output, 0, 2); CHECK_TENSOR(F_t_0_f);
	    for(i = 0; i < 2 * n; i++) 	// F_t_0_f, F_t_1_f --2 Channels
	    	F_t_0_f->data[i] += F_t_0->data[i];
	    TENSOR *F_t_1_f = tensor_slice_chan(temp_interpolate_output, 2, 4); CHECK_TENSOR(F_t_1_f);
	    for(i = 0; i < 2 * n; i++)  // F_t_0_f, F_t_1_f --2 Channels
	    	F_t_1_f->data[i] += F_t_1->data[i];

	    // g_I0_F_t_0_mask = torch.sigmoid(temp_interpolate_output[:, 4:5, :, :])
	    // g_I1_F_t_1_mask = 1 - g_I0_F_t_0_mask
	    TENSOR *g_I0_F_t_0_mask = tensor_slice_chan(temp_interpolate_output, 4, 5);
	    CHECK_TENSOR(g_I0_F_t_0_mask);
	    for (i = 0; i < n; i++)  	// g_I0_F_t_0_mask --1 Channels
	    	g_I0_F_t_0_mask->data[i] = 1.0/(1.0 + expf(-g_I0_F_t_0_mask->data[i]));

	    tensor_destroy(temp_interpolate_output);

        TENSOR *g_I0_F_t_0_fine = flow_backwarp(I0, F_t_0_f); CHECK_TENSOR(g_I0_F_t_0_fine);
        TENSOR *g_I1_F_t_1_fine = flow_backwarp(I1, F_t_1_f); CHECK_TENSOR(g_I1_F_t_1_fine);

	    // w = [1 - t, t]
	    // output_tensor = (w[0] * g_I0_F_t_0_mask * g_I0_F_t_0_fine 
	    //     + w[1] * g_I1_F_t_1_mask * g_I1_F_t_1_fine)/(w[0] * g_I0_F_t_0_mask + w[1] * g_I1_F_t_1_mask)
	    // # (Pdb) output_tensor.size()
	    // # torch.Size([1, 3, 512, 960])

        // Save to output_tensor
        w[0] = (1.0 - t); w[1] = t;
    	to = tensor_start_chan(output_tensor, 0, 3 * j); // for Suppose_X
        for (i = 0; i < 3 * n; i++) {	 // output_tensor --3 Channels
        	a = w[0] * g_I0_F_t_0_mask->data[i];
        	b = w[1] * (1.0 - g_I0_F_t_0_mask->data[i]); // g_I1_F_t_1_mask->data[i];
        	t = a + b + 0.000001; // t == 0 ?
        	to[i] = (a * g_I0_F_t_0_fine->data[i] + b * g_I1_F_t_1_fine->data[i])/t;
        }

        tensor_destroy(F_t_1_f);
        tensor_destroy(F_t_0_f);

        tensor_destroy(g_I1_F_t_1_fine);
        tensor_destroy(g_I0_F_t_0_fine);

        tensor_destroy(g_I0_F_t_0_mask);
	}
	tensor_destroy(F_t_1);
	tensor_destroy(F_t_0);
	tensor_destroy(F_0_1);
	tensor_destroy(F_1_0);
	tensor_destroy(I1);
	tensor_destroy(I0);

	return output_tensor;
}


void output_image(TENSOR *tensor, const char *prefix, int index)
{
	char filename[256];
	snprintf(filename, sizeof(filename), "output/%s-%04d.png", prefix, index);
	SaveTensorAsImage(tensor, filename);
}

int SlowService(char *endpoint, int use_gpu)
{
	int socket, reqcode, lambda, rescode;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *fc_engine = NULL;
	OrtEngine *at_engine = NULL;

	srand(time(NULL));

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	fc_engine = CreateEngine("video_slow_fc.onnx", use_gpu /*use_gpu*/);
	CheckEngine(fc_engine);

	at_engine = CreateEngine("video_slow_at.onnx", use_gpu /*use_gpu*/);
	CheckEngine(at_engine);

	lambda = 0;
	for (;;) {
		syslog_info("Service %d times", lambda);

		input_tensor = request_recv(socket, &reqcode);

		if (!tensor_valid(input_tensor)) {
			syslog_error("Request recv bad tensor ...");
			continue;
		}
		syslog_info("Request Code = %d", reqcode);

		// Real service ...
		time_reset();
		output_tensor = slow_do(fc_engine, at_engine, input_tensor, 4 /*scale */);
		time_spend((char *)"Infer");

		if (tensor_valid(output_tensor)) {
			rescode = reqcode;
			response_send(socket, output_tensor, rescode);
			tensor_destroy(output_tensor);
		}

		tensor_destroy(input_tensor);

		lambda++;
	}
	DestroyEngine(at_engine);
	DestroyEngine(fc_engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

int server(char *endpoint, int use_gpu)
{
	return SlowService(endpoint, use_gpu);
}

int slow(int socket, char *input_file)
{
	int rescode;
	IMAGE *send_image, *resize_send_image;
	TENSOR *send_tensor, *recv_tensor;

	printf("Video Slowing %s ...\n", input_file);

	send_image = image_load(input_file); check_image(send_image);
	resize_send_image = image_zoom(send_image, 256, 256, 1); check_image(resize_send_image);

	if (image_valid(resize_send_image)) {
		send_tensor = tensor_from_image(resize_send_image, 0);
		check_tensor(send_tensor);

		recv_tensor = OnnxRPC(socket, send_tensor, VIDEO_SLOW_REQCODE, &rescode);
		if (tensor_valid(recv_tensor)) {
			SaveTensorAsImage(recv_tensor, input_file);
			tensor_destroy(recv_tensor);
		}

		tensor_destroy(send_tensor);
		image_destroy(resize_send_image);
	}
	image_destroy(send_image);

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
	char *endpoint = (char *) VIDEO_SLOW_URL;

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
			slow(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	return RET_ERROR;
}
