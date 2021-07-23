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

int global_zoom_method = 0;

TENSOR *zoomx_do(OrtEngine *encoder, OrtEngine *transform, TENSOR *input_tensor, int scale)
{
	int h, w, n, start, stop, count;
	TENSOR *grid, *cell, *feat, *feat_grid, *output_tensor;
	TENSOR *sub_grid, *sub_cell;	// temp tensors
	TENSOR *tensor_list[4096], *input_list[4], *pred;

	CHECK_TENSOR(input_tensor);
	// Default 4x
	if (scale < 1)
		scale = 4;

	h = scale * input_tensor->height;
	w = scale * input_tensor->width;
	n = h * w;

	grid = tensor_make_grid(input_tensor->batch, h, w); CHECK_TENSOR(grid);
	tensor_view_(grid, input_tensor->batch, 2, h * w, 1); // [1, 2, h, w] ==> [1, 2, h * w, 1]
	cell = tensor_make_cell(input_tensor->batch, h, w); CHECK_TENSOR(cell);
	tensor_view_(cell, input_tensor->batch, 2, h * w, 1); // [1, 2, h, w] ==> [1, 2, h * w, 1]

	feat = SingleTensorForward(encoder, input_tensor); CHECK_TENSOR(feat);
	feat_grid = tensor_make_grid(feat->batch, feat->height, feat->width); CHECK_TENSOR(feat_grid);

    count = 0;
	start = 0;
	while (start < n && count < ARRAY_SIZE(tensor_list)) {
		stop = start + 65536;
		if (stop > n)
			stop = n;

		sub_grid = tensor_slice_row(grid, start, stop); CHECK_TENSOR(sub_grid);

		sub_cell = tensor_slice_row(cell, start, stop); CHECK_TENSOR(sub_cell);

		input_list[0] = feat;
		input_list[1] = feat_grid;
		input_list[2] = sub_grid;
		input_list[3] = sub_cell;
		pred = MultipleTensorForward(transform, ARRAY_SIZE(input_list), input_list);
		CHECK_TENSOR(pred);
		
		tensor_list[count++] = pred; // [1, 3, 65536, 1]

		tensor_destroy(sub_grid);
		tensor_destroy(sub_cell);
		start = stop;
	}

	output_tensor = tensor_stack_row(count, tensor_list); CHECK_TENSOR(output_tensor);
	for (n = 0; n < count; n++) {
		tensor_destroy(tensor_list[n]);
	}
	tensor_destroy(feat_grid);
	tensor_destroy(feat);
	tensor_destroy(cell);
	tensor_destroy(grid);

	tensor_view_(output_tensor, input_tensor->batch, 3, h, w);	// [1, 3, h, w]
	tensor_clamp_(output_tensor, 0.0, 1.0);

	return output_tensor;
}


// Patch model input: 1 x 3 x (-1) x (-1), 1 x 3 x (-1) x (-1)
int server(char *endpoint, int use_gpu)
{
	int socket, count, msgcode, scale = 1;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *engine = NULL;
	OrtEngine *zooms_engine = NULL;
	OrtEngine *zoomx_encoder_engine = NULL;
	OrtEngine *zoomx_transform_engine = NULL;

	InitEngineRunningTime();	// aviod compiler compaint

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	count = 0;
	for (;;) {
		if (EngineIsIdle()) {
			StopEngine(engine);
			StopEngine(zooms_engine);
			StopEngine(zoomx_encoder_engine);
			StopEngine(zoomx_transform_engine);
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
			output_tensor = SingleTensorForward(engine, input_tensor);
			time_spend((char *)"Image zoom4x");

			service_response(socket, msgcode, output_tensor);
			tensor_destroy(output_tensor);

			count++;
		} else if (msgcode == IMAGE_ZOOM_SERVICE_WITH_PAN) {
			syslog_info("Service %d times", count);
			StartEngine(zooms_engine, (char *)"image_zooms.onnx", use_gpu);

			// Real service ...
			time_reset();
			output_tensor = SingleTensorForward(zooms_engine, input_tensor);
			time_spend((char *)"Image zoom4x with PAN");

			service_response(socket, msgcode, output_tensor);
			tensor_destroy(output_tensor);

			count++;
		} else if (SERVICE_CODE(msgcode) == IMAGE_ZOOM_SERVICE_WITH_ANY_SIZE) {
			syslog_info("Service %d times", count);
			StartEngine(zoomx_encoder_engine, (char *)"image_zoomx_encoder.onnx", use_gpu);
			StartEngine(zoomx_transform_engine, (char *)"image_zoomx_transform.onnx", use_gpu);

			// Real service ...
			time_reset();
			scale = SERVICE_ARGUMENT(msgcode);
			output_tensor = zoomx_do(zoomx_encoder_engine, zoomx_transform_engine, input_tensor, scale);
			time_spend((char *)"Image zoomx with any size");

			service_response(socket, msgcode, output_tensor);
			tensor_destroy(output_tensor);

			count++;
		} else {
			service_response(socket, OUTOF_SERVICE_MESSAGE, NULL);
		}

		tensor_destroy(input_tensor);
	}
	StopEngine(engine);
	StopEngine(zooms_engine);
	StopEngine(zoomx_encoder_engine);
	StopEngine(zoomx_transform_engine);

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
		recv_tensor = OnnxRPC(socket, send_tensor, global_zoom_method);
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
	printf("    m, --methodr <0 | 1 | x>    Zoom method (0 -- Fast 4x, 1 -- Normal 4x, x -- Scale).\n");

	exit(1);
}

int main(int argc, char **argv)
{
	int i, optc;
	int use_gpu = 1;
	int running_server = 0;
	int socket;
	int method = 0;

	int option_index = 0;
	char *endpoint = (char *) IMAGE_ZOOM_URL;

	struct option long_opts[] = {
		{"help", 0, 0, 'h'},
		{"endpoint", 1, 0, 'e'},
		{"server", 1, 0, 's'},
		{"server", 1, 0, 'm'},
		{0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);

	while ((optc = getopt_long(argc, argv, "h e: s: m:" , long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'e':
			endpoint = optarg;
			break;
		case 's':
			running_server = 1;
			use_gpu = atoi(optarg);
			break;
		case 'm':
			method = atoi(optarg);
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
		// process method ...
		if (method == 0)
			global_zoom_method = IMAGE_ZOOM_SERVICE_WITH_PAN;
		else if (method == 1)
			global_zoom_method = IMAGE_ZOOM_SERVICE;
		else
			global_zoom_method = DEFINE_SERVICE(IMAGE_ZOOM_SERVICE_WITH_ANY_SIZE, method);

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
