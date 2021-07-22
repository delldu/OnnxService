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

// For scale == 4:
// input_tensor  -- [1, 6, 512, 960]
// output_tensor -- [4 - 1, 3, 512, 960]
TENSOR *slow_do(OrtEngine *fc, OrtEngine *at, TENSOR *input_tensor, int scale)
{
	int i, j, c, n;
	float t, w[4], *to, *from0, *from1, a, b;
	TENSOR *I0, *I1, *flowOut, *F_0_1, *F_1_0, *F_t_0, *F_t_1, *output_tensor;

	CheckEngine(fc);
	CheckEngine(at);
	CHECK_TENSOR(input_tensor);

	// Suppose_X: input is 1x6xHxW input !!!
	if (input_tensor->batch != 1 || input_tensor->chan != 6) {
		syslog_error("Now only support 1x3xHxW input tensor.");
		return NULL;
	}

	I0 = tensor_slice_chan(input_tensor, 0, 3); CHECK_TENSOR(I0);
	// SaveTensorAsImage(I0, "I0.png");

	I1 = tensor_slice_chan(input_tensor, 3, 6); CHECK_TENSOR(I1);
	// SaveTensorAsImage(I1, "I1.png");

    flowOut = TensorForward(fc, input_tensor);
    CHECK_TENSOR(flowOut);
	// # flowOut.size() -- torch.Size([1, 4, 512, 960])

	// F_0_1 = flowOut[:, :2, :, :]
	// F_1_0 = flowOut[:, 2:, :, :]
	F_0_1 = tensor_slice_chan(flowOut, 0, 2); CHECK_TENSOR(F_0_1);
	F_1_0 = tensor_slice_chan(flowOut, 2, 4); CHECK_TENSOR(F_1_0);

    n = F_0_1->height * F_0_1->width;

	// Suppose_X
	F_t_0 = tensor_create(1, 2, input_tensor->height, input_tensor->width);
	CHECK_TENSOR(F_t_0);
	F_t_1 = tensor_create(1, 2, input_tensor->height, input_tensor->width);
	CHECK_TENSOR(F_t_1);

  	// Suppose_X
	output_tensor = tensor_create(scale - 1, 3, input_tensor->height, input_tensor->width);
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

        TENSOR *g_I0_F_t_0 = tensor_flow_backwarp(I0, F_t_0); CHECK_TENSOR(g_I0_F_t_0);
        TENSOR *g_I1_F_t_1 = tensor_flow_backwarp(I1, F_t_1); CHECK_TENSOR(g_I1_F_t_1);

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

        TENSOR *g_I0_F_t_0_fine = tensor_flow_backwarp(I0, F_t_0_f); CHECK_TENSOR(g_I0_F_t_0_fine);
        TENSOR *g_I1_F_t_1_fine = tensor_flow_backwarp(I1, F_t_1_f); CHECK_TENSOR(g_I1_F_t_1_fine);

	    // w = [1 - t, t]
	    // output_tensor = (w[0] * g_I0_F_t_0_mask * g_I0_F_t_0_fine 
	    //     + w[1] * g_I1_F_t_1_mask * g_I1_F_t_1_fine)/(w[0] * g_I0_F_t_0_mask + w[1] * g_I1_F_t_1_mask)
	    // # (Pdb) output_tensor.size()
	    // # torch.Size([1, 3, 512, 960])

        // Save to output_tensor
        w[0] = (1.0 - t); w[1] = t;
        for (c = 0; c < output_tensor->chan; c++) {
	    	to = tensor_start_chan(output_tensor, j, c); // for Suppose_X
	    	from0 = tensor_start_chan(g_I0_F_t_0_fine, 0, c);
	    	from1 = tensor_start_chan(g_I1_F_t_1_fine, 0, c);

	        for (i = 0; i < n; i++) {	 // output_tensor --3 Channels
	        	a = w[0] * g_I0_F_t_0_mask->data[i];
	        	b = w[1] * (1.0 - g_I0_F_t_0_mask->data[i]); // g_I1_F_t_1_mask->data[i];
	        	t = a + b + 0.000001; // t == 0 ?
	        	to[i] = (a * from0[i] + b * from1[i])/t;
	        	to[i] = CLAMP(to[i], 0.0, 1.0);
	        }
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

// Video slow model
//
// video_slow_fc.onnx -- flow compute
// 		input: 1 x 6 x -1 x -1
// 		output: 1 x 4 x -1 x -1
// video_slow_at.onnx -- arbitary interpolate
// 		input: 1 x 20 x -1 x -1
// 		output: 1 x 5 x -1 x -1

int SlowService(char *endpoint, int use_gpu)
{
	int socket, count, msgcode;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *fc_engine = NULL;
	OrtEngine *at_engine = NULL;

	srand(time(NULL));
	InitEngineRunningTime();	// aviod compiler compaint

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	count = 0;
	for (;;) {
		if (EngineIsIdle()) {
			StopEngine(fc_engine);
			StopEngine(at_engine);
		}

		if (! socket_readable(socket, 1000))	// timeout 1 s
			continue;

		input_tensor = service_request(socket, &msgcode);
		if (!tensor_valid(input_tensor))
			continue;

		if (msgcode == VIDEO_SLOW_SERVICE) {
			syslog_info("Service %d times", count);

			StartEngine(fc_engine, (char *)"video_slow_fc.onnx", use_gpu);
			StartEngine(at_engine, (char *)"video_slow_at.onnx", use_gpu);

			// Real service ...
			time_reset();
			output_tensor = slow_do(fc_engine, at_engine, input_tensor, 4 /*scale */);
			time_spend((char *)"Video slowing");

			service_response(socket, VIDEO_SLOW_SERVICE, output_tensor);
			tensor_destroy(output_tensor);

			tensor_destroy(input_tensor);

			count++;
		} else {
			// ...
		}
	}
	StopEngine(fc_engine);
	StopEngine(at_engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

int server(char *endpoint, int use_gpu)
{
	return SlowService(endpoint, use_gpu);
}

TENSOR *slow_load(char *input_file1, char *input_file2)
{
	int n;
	float *to;

	IMAGE *image1, *image2;
	TENSOR *tensor1, *tensor2, *output = NULL;

	image1 = image_load(input_file1); CHECK_IMAGE(image1);
	image2 = image_load(input_file2); CHECK_IMAGE(image2);

	if (image1->height == image2->height && image1->width == image2->width) {
		tensor1 = tensor_from_image(image1, 0 /* without channel A */);	CHECK_TENSOR(tensor1);
		tensor2 = tensor_from_image(image2, 0 /* without channel A */);	CHECK_TENSOR(tensor2);

		output = tensor_create(1, 6, image1->height, image1->width);
		CHECK_TENSOR(output);
		n = output->height * output->width;
		to = output->data;
		memcpy(to, tensor1->data, 3 * n * sizeof(float));
		to = &output->data[3 * n];
		memcpy(to, tensor2->data, 3 * n * sizeof(float));

		tensor_destroy(tensor2);
		tensor_destroy(tensor1);
	} else {
		syslog_error("Image size is not same,");
	}

	image_destroy(image2);
	image_destroy(image1);

	return output;
}

int slow_save(TENSOR *tensor)
{
	int b;
	IMAGE *image;
	char filename[256];

	check_tensor(tensor);
	for (b = 0; b < tensor->batch; b++) {
		image = image_from_tensor(tensor, b);
		snprintf(filename, sizeof(filename), "output/%06d.png", b + 1);
		image_save(image, filename);
		image_destroy(image);
	}

	return RET_OK;
}

int slow(int socket, char *input_file1, char *input_file2)
{
	TENSOR *send_tensor, *recv_tensor;

	printf("Video slowing between %s and %s ...\n", input_file1, input_file2);

	send_tensor = slow_load(input_file1, input_file2);
	check_tensor(send_tensor);

	// Server limited: only accept 8 times tensor !!!
	recv_tensor = ZeropadOnnxRPC(socket, send_tensor, VIDEO_SLOW_SERVICE, 8);
	if (tensor_valid(recv_tensor)) {
		slow_save(recv_tensor);
		tensor_destroy(recv_tensor);
	}

	tensor_destroy(send_tensor);

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

	if (running_server) {
		if (IsRunning(endpoint))
			exit(-1);
		return server(endpoint, use_gpu);
	}
	else if (argc > 1) {
		if ((socket = client_open(endpoint)) < 0)
			return RET_ERROR;

		for (i = optind; i + 1 < argc; i++)
			slow(socket, argv[i], argv[i + 1]);

		client_close(socket);
		return RET_OK;
	}

	return RET_ERROR;
}
