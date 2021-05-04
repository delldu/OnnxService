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
#include <pthread.h>


// #include <iostream>
// #include <iomanip>
// #include <string>
// #include <map>
#include <random>
// #include <cmath>


#include <nimage/image.h>
#include <nimage/nnmsg.h>

#include "engine.h"

#define SEED_FACES 12
#define W_SPACE_DIM 512

#define DEBUG 1

OrtEngine *decoder_engine = NULL;
OrtEngine *transformer_engine = NULL;
OrtEngine *loss_engine = NULL;

TENSOR *standard_tensor = NULL;		// Search Reference tensor !!!

typedef struct {				/* Used as argument to fit_start() */
	pthread_t id;				/* ID returned by pthread_create() */
	int no;						/* Sub thread */
	float x[2][W_SPACE_DIM], y[2];
	float lr;	// learning rate
} FACE_CELL;

float random_gauss()
{
#if 1
	// Marsaglia and Bray 1964
    static double V1, V2, S;
    static int phase = 0;
    double X;
     
    if (phase == 0) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
             
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
         
        X = V1 * sqrt(-2 * log(S) / S);
    } else {
        X = V2 * sqrt(-2 * log(S) / S);
    }
         
    phase = 1 - phase;
 
    return (float)X;
#else
	double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
	double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
	double r = u * u + v * v;
	if (r == 0 || r > 1)
		return random_gauss();
	
	double c = sqrt(-2 * log(r) / r);

	return (float)(u * c);

#endif
}

int face_sample(FACE_CELL *optimizer)
{
	int i, j;

	for (i = 0; i < 2; i++) {
		for (j = 0; j < W_SPACE_DIM; j++)
			optimizer->x[i][j] = random_gauss();
	}

	// optimizer_calculate(optimizer, i);
	return RET_OK;
}


void output_image(TENSOR *tensor, const char *prefix, int index)
{
	char filename[256];
	snprintf(filename, sizeof(filename), "output/%s-%04d.png", prefix, index);
	SaveTensorAsImage(tensor, filename);
}

TENSOR *random_zcode()
{
	int i;
#if 0	
	float bad[] = {
0.04,-0.05,-1.74,0.12,0.83,0.21,0.40,0.51,1.94,1.19,1.17,0.43,-0.85,2.37,-0.71,-1.39,-1.10,0.07,0.34,0.00,0.31,0.37,-2.07,0.00,-0.60,-0.96,-0.38,-0.35,-0.06,0.13,-1.68,0.57,-0.11,-1.36,-0.71,-0.53,0.24,0.27,-0.72,-1.61,-0.23,1.47,1.21,2.23,-1.90,0.32,-0.47,-1.28,-0.60,0.35,1.71,-0.62,0.52,-1.33,0.95,-0.66,0.13,-1.03,1.34,-0.68,1.91,-0.76,0.33,-0.53,-0.35,-0.92,-2.00,-0.27,0.30,0.20,1.01,0.03,-0.47,1.35,-0.12,-0.00,1.62,0.32,2.51,0.06,0.77,0.15,1.79,0.90,-0.24,0.05,1.15,0.11,-0.53,-0.44,2.07,-0.25,1.94,0.24,0.17,2.58,-1.56,-0.80,-0.56,-0.30,-0.96,0.01,1.74,1.04,-1.79,0.73,-0.43,-0.39,0.12,0.60,-1.54,-0.32,-0.47,-2.44,0.42,-0.92,0.86,-0.34,1.54,-1.07,-0.34,0.51,0.56,-1.87,0.35,0.03,-0.92,0.34,-1.09,-0.15,-1.02,-0.67,-1.02,-1.94,0.00,-0.39,-0.85,1.11,-0.18,1.44,-0.50,2.39,-1.88,-1.37,0.27,-1.22,-0.11,1.32,0.50,-1.62,1.01,1.23,0.36,0.64,-1.64,0.14,-1.22,0.22,-1.39,0.75,-1.70,-1.14,1.02,-0.83,-0.45,-1.44,2.15,-0.29,-1.17,-0.21,-0.23,0.12,0.92,1.04,-1.61,-1.90,2.14,-1.01,0.20,1.11,-0.45,-1.46,-0.65,1.23,0.28,-0.82,-0.16,1.20,-0.47,0.31,0.60,-0.37,1.82,-0.82,-0.96,-1.14,-0.09,0.08,2.12,1.09,-0.98,1.21,-0.70,-0.17,1.45,-0.50,-1.45,1.40,-0.37,1.19,-3.06,1.20,0.92,-1.83,0.72,0.70,-0.75,0.93,0.14,0.39,0.97,-0.36,0.88,-0.07,-0.43,-1.56,1.93,0.03,2.11,-0.57,-0.55,-0.25,1.20,0.19,1.24,-0.31,0.86,0.22,-0.46,-0.48,-0.36,1.01,-1.03,0.20,1.05,-0.45,1.03,1.38,1.72,0.41,-1.00,0.42,1.05,2.35,0.38,-0.25,-0.88,0.09,0.30,1.45,-1.63,-0.87,-0.39,-0.86,0.66,-0.55,-0.21,-0.94,1.12,0.13,1.10,1.13,0.70,-0.30,-1.46,1.50,1.02,0.11,-1.59,-0.53,0.64,-1.15,1.35,-0.98,-0.84,-0.72,-1.11,0.10,2.08,-0.37,2.06,-0.24,1.63,-0.73,-0.03,-1.04,0.31,-2.15,0.61,-1.13,-0.56,0.07,1.21,-2.38,-1.23,-0.30,0.50,-0.06,2.54,-0.04,-1.07,-1.12,1.17,0.11,0.03,0.26,-0.29,1.69,0.51,-0.84,-0.37,-0.32,-1.23,0.94,0.06,0.69,1.01,0.22,-0.10,-2.64,-0.74,0.33,-0.33,0.16,-1.12,-0.42,0.84,0.16,1.41,-0.13,1.86,0.03,0.37,0.98,0.24,0.51,0.13,-1.58,1.03,-0.30,1.10,-0.50,0.46,0.48,-0.25,0.08,0.35,-0.42,-0.42,0.82,1.17,-0.04,-0.36,0.51,0.04,-1.17,-2.64,1.25,-0.50,-0.33,1.33,0.71,1.87,-0.55,0.26,-1.42,1.73,-0.84,2.50,-0.13,1.43,-0.46,-0.62,1.01,0.55,-0.16,0.43,-0.35,1.49,-1.13,-0.31,-0.82,0.94,-0.06,-0.28,0.79,-0.33,-0.67,0.01,0.24,0.11,-0.82,-0.71,-0.62,-0.02,1.41,0.91,-0.92,-0.26,-0.13,0.36,-0.90,1.40,0.02,-2.10,-0.37,0.71,0.37,0.43,1.41,2.03,-0.26,-0.62,2.33,-1.22,0.96,0.52,0.42,-0.84,2.54,-0.21,0.09,0.28,-0.85,-0.93,0.27,-0.53,-0.33,0.35,-0.47,0.76,-0.10,0.03,0.15,0.22,-0.63,0.16,0.68,1.61,-0.35,0.72,-0.96,0.67,0.25,-0.03,0.26,-0.21,-2.23,0.78,0.19,0.96,-1.52,1.85,1.49,-1.41,-0.92,0.73,-0.97,0.98,-0.58,-1.44,-0.21,-0.43,1.24,-0.31,-1.02,-0.26,-0.43,-0.86,-1.37,0.70,-0.44,0.29,0.15,-1.03,-0.38,1.38,0.90,1.41,0.77,-1.23,-0.72,-1.19,0.53,-0.03,-0.28,0.42,0.15,-0.39,-2.07,-0.09,1.15,-0.16,1.77,0.28,-1.71,-0.82,-1.09,-0.43,1.21,0.68,-0.00
	};
#endif

	TENSOR *zcode_tensor;

	srand(time(NULL));

	// Create zcode tensor
	zcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(zcode_tensor);


    std::random_device rd{};
    std::mt19937 gen{rd()};
 
    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution<float> dis{0.0, 1.0};
	for (i = 0; i < W_SPACE_DIM; i++) {
		zcode_tensor->data[i] = dis(gen);
	} 

#if 1
	float d, stdv, mean;
	mean = stdv = 0.0;
	for (i = 0; i < W_SPACE_DIM; i++) {
		// d = random_gauss();
		// d = bad[i];

		d = zcode_tensor->data[i];
		mean += d;
		stdv += d * d;
	}
	mean /= W_SPACE_DIM;
	stdv /= W_SPACE_DIM;
	stdv -= mean * mean;
	stdv = sqrtf(stdv);

	CheckPoint("---------- mean = %.2f, stdv = %.2f", mean, stdv);
	printf("[");
	for (i = 0; i < W_SPACE_DIM; i++) {
		printf("%0.2f", zcode_tensor->data[i]);
		if (i == W_SPACE_DIM - 1) {
			printf("]");
		} else {
			printf(",");
		}
	}
	printf("\n");
	CheckPoint("");
#endif

	return zcode_tensor;
}

TENSOR *random_wcode()
{
	TENSOR *zcode_tensor, *wcode_tensor;

	CheckEngine(transformer_engine);

	// Create zcode tensor
	zcode_tensor = random_zcode();
	CHECK_TENSOR(zcode_tensor);

	// Compute wcode
	wcode_tensor = TensorForward(transformer_engine, zcode_tensor);
	CHECK_TENSOR(wcode_tensor);

	tensor_destroy(zcode_tensor);

	return wcode_tensor;
}

TENSOR *create_wtensor(double *data)
{
	int i;
	TENSOR *wcode_tensor;

	wcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(wcode_tensor);
	for (i = 0; i < W_SPACE_DIM; i++)
		wcode_tensor->data[i] = (float)data[i];

	return wcode_tensor;
}

float wcode_loss(TENSOR *wcode_tensor)
{
	static int face_index = 1;
	int i, n;
	float loss = 11.0, p_loss;
	TENSOR *output_image_tensor, *output_tensor, *loss_input_tensor, *loss_tensor;

	CheckEngine(decoder_engine);

	output_image_tensor = TensorForward(decoder_engine, wcode_tensor);

	if (! tensor_valid(output_image_tensor)) {
		syslog_error("Generate image.");
		return loss;
	}

	output_tensor = tensor_zoom(output_image_tensor, 256, 256);
	if (! tensor_valid(output_tensor)) {
		syslog_error("Zoom image tensor to (256x256)");
		return loss;
	}

	#if 1 // DEBUG
	output_image(output_image_tensor, "face-", (face_index++) % SEED_FACES);
	#endif

	// Compute loss, maybe we need more complex method !!!
	loss = 0.0;
	for (i = 0; i < 256*256*3; i++) {
		loss += ABS(standard_tensor->data[i] - output_tensor->data[i]);
	}
	loss /= (256*256*3);

	// Get perception loss
	loss_input_tensor = tensor_create(2, 3, 256, 256);
	if (! tensor_valid(loss_input_tensor)) {
		return 11.0;
	}

	n = 3*256*256;
	memcpy(loss_input_tensor->data, standard_tensor->data, n * sizeof(float));
	memcpy(&loss_input_tensor->data[n], output_tensor->data, n * sizeof(float));

	loss_tensor = TensorForward(loss_engine, loss_input_tensor);
	if (! tensor_valid(loss_tensor)) {
		return 11.0;
	}

	p_loss = 0;
	n = loss_tensor->batch * loss_tensor->chan * loss_tensor->height * loss_tensor->width;
	for (i = 0; i < n; i++)
		p_loss += loss_tensor->data[i];
	p_loss /= n;

	loss += 10.0 * p_loss;		// perception loss's important is 10.0

	tensor_destroy(loss_tensor);
	tensor_destroy(loss_input_tensor);

	// Release tensors
	tensor_destroy(output_image_tensor);
	tensor_destroy(output_tensor);

	return loss;
}

static void *fit_start(void *arg)
{
	FACE_CELL *info = (FACE_CELL *)arg;

	CheckEngine(decoder_engine);

	// syslog_info("Running thread %d ...", info->no);
	// Create input tensor
	TENSOR *wcode_tensor;
	// xxxx8888 wcode_tensor = create_wtensor(info->x);
	// info->y = wcode_loss(wcode_tensor);
	// Release tensors
	tensor_destroy(wcode_tensor);
	// syslog_info("Thread %d result = %.4f", info->no, info->y);

	return NULL;
}

float *do_real_searching(int epochs)
{
	int i, j, ret, lambda;
	FACE_CELL *info;
	pthread_attr_t attr;

	lambda = SEED_FACES;

	info = (FACE_CELL *)calloc(lambda, sizeof(FACE_CELL));
	if (info == NULL)
		syslog_error("Allocte memory.");

	ret = pthread_attr_init(&attr);
	if (ret != 0)
		syslog_error("Init thread attr.");

	for (i = 0; i < lambda; i++) {
		info[i].no = i + 1;

		ret = pthread_create(&info[i].id, &attr, &fit_start, &info[i]);
		if (ret != 0)
			syslog_error("Create thread.");
	}
	ret = pthread_attr_destroy(&attr);
	if (ret != 0)
		syslog_error("Destropy thread attr.");

	/* Now join with each thread */
	for (i = 0; i < lambda; i++) {
		ret = pthread_join(info[i].id, NULL);
		if (ret != 0)
			syslog_error("Join thread.");
	}

	// Save best ...
	free(info);

	// xxxx8888
	return NULL;
}

TENSOR *face_search(TENSOR *reference_tensor)
{
	int i;
	float *best;
	TENSOR *wcode_tensor;

	CHECK_TENSOR(reference_tensor);

	standard_tensor = reference_tensor;
	best = do_real_searching(100);

	// Save best to wcode
	wcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(wcode_tensor);
	for (i = 0; i < W_SPACE_DIM; i++)
		wcode_tensor->data[i] = best[i];

	free(best);

	return wcode_tensor;
}

int FaceGanService111(char *endpoint, int use_gpu)
{
	int socket, lambda, msgcode;
	TENSOR *input_tensor, *output_tensor, *wcode_tensor;

	srand(time(NULL));

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	decoder_engine = CreateEngine((char *)"image_gandecoder.onnx", 0 /*not use_gpu for model bug !!!*/);
	CheckEngine(decoder_engine);

	transformer_engine = CreateEngine((char *)"image_gantransformer.onnx", use_gpu /*use_gpu*/);
	CheckEngine(transformer_engine);

	loss_engine = CreateEngine((char *)"image_ganloss.onnx", use_gpu /*use_gpu*/);
	CheckEngine(loss_engine);

	lambda = 0;
	for (;;) {
		syslog_info("Service %d times", lambda);

		input_tensor = service_request(socket, &msgcode);
		if (!tensor_valid(input_tensor))
			continue;

		// msgcode = IMAGE_FACEGAN_SERVICE;
		// Real service ...
		time_reset();
		wcode_tensor = face_search(input_tensor); check_tensor(wcode_tensor);

		output_tensor = TensorForward(decoder_engine, wcode_tensor);
		time_spend((char *)"Infer");

		if (tensor_valid(output_tensor)) {
#if 0	// For debug			
			IMAGE *image = image_from_tensor(output_tensor, 0);
			image_save(image, "output/last.png");
			image_destroy(image);
#endif
			service_response(socket, IMAGE_FACEGAN_SERVICE, output_tensor);
			tensor_destroy(output_tensor);
		}

		tensor_destroy(wcode_tensor);

		tensor_destroy(input_tensor);

		lambda++;
	}
	DestroyEngine(loss_engine);
	DestroyEngine(transformer_engine);
	DestroyEngine(decoder_engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

int FaceGanService(char *endpoint, int use_gpu, CustomSevice custom_service_function)
{
	int socket, count, msgcode;
	TENSOR *input_tensor, *output_tensor;
	OrtEngine *engine = NULL;

	srand(time(NULL));

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	if (!custom_service_function)
		custom_service_function = service_response;

	count = 0;
	for (;;) {
		if (EngineIsIdle())
			StopEngine(engine);

		if (!socket_readable(socket, 1000))	// timeout 1 s
			continue;

		input_tensor = service_request(socket, &msgcode);
		if (!tensor_valid(input_tensor))
			continue;

		if (msgcode == IMAGE_FACEGAN_SERVICE) {
			syslog_info("Service %d times", count);
			StartEngine(engine, "xxxx8888", use_gpu);

			// Real service ...
			time_reset();
			output_tensor = TensorForward(engine, input_tensor);
			time_spend((char *) "Predict");

			service_response(socket, IMAGE_FACEGAN_SERVICE, output_tensor);
			tensor_destroy(output_tensor);

			count++;
		} else {
			// service_response(socket, servicecode, input_tensor)
			custom_service_function(socket, OUTOF_SERVICE, NULL);
		}

		tensor_destroy(input_tensor);
	}
	StopEngine(engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}




int server(char *endpoint, int use_gpu)
{
	return FaceGanService(endpoint, use_gpu, NULL);
}

int facegan(int socket, char *input_file)
{
	IMAGE *send_image, *resize_send_image;
	TENSOR *send_tensor, *recv_tensor;

	printf("Face Searching %s ...\n", input_file);

	send_image = image_load(input_file); check_image(send_image);
	resize_send_image = image_zoom(send_image, 256, 256, 1); check_image(resize_send_image);

	if (image_valid(resize_send_image)) {
		send_tensor = tensor_from_image(resize_send_image, 0);
		check_tensor(send_tensor);

		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_FACEGAN_SERVICE);
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

int test(int use_gpu)
{
	TENSOR *wcode_tensor, *output_tensor;
	IMAGE *image;

	srand(time(NULL));

	transformer_engine = CreateEngine((char *)"image_gantransformer.onnx", use_gpu);
	CheckEngine(transformer_engine);

	decoder_engine = CreateEngine((char *)"image_gandecoder.onnx", use_gpu);
	CheckEngine(decoder_engine);

	time_reset();

	wcode_tensor = random_wcode();
	check_tensor(wcode_tensor);

	output_tensor = TensorForward(decoder_engine, wcode_tensor);
	time_spend((char *)"Infer");

	image = image_from_tensor(output_tensor, 0);
	image_show(image, "Face");
	image_destroy(image);
	
	tensor_destroy(output_tensor);
	tensor_destroy(wcode_tensor);

	// Release ...
	DestroyEngine(decoder_engine);
	DestroyEngine(transformer_engine);

	return 0;
}


void help(char *cmd)
{
	printf("Usage: %s [option] <image files>\n", cmd);
	printf("    h, --help                   Display this help.\n");
	printf("    e, --endpoint               Set endpoint.\n");
	printf("    s, --server <0 | 1>         Start server (use gpu).\n");
	printf("    t, --test <0 | 1>           Test (use gpu).\n");

	exit(1);
}

int main(int argc, char **argv)
{
	int i, optc;
	int use_gpu = 1;
	int running_server = 0;
	int socket;

	int option_index = 0;
	char *endpoint = (char *) IMAGE_FACEGAN_URL;

	struct option long_opts[] = {
		{"help", 0, 0, 'h'},
		{"endpoint", 1, 0, 'e'},
		{"server", 1, 0, 's'},
		{"test", 1, 0, 't'},

		{0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);

	while ((optc = getopt_long(argc, argv, "h e: s: t:", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'e':
			endpoint = optarg;
			break;
		case 's':
			running_server = 1;
			use_gpu = atoi(optarg);
			break;
		case 't':
			return test(atoi(optarg));
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
			facegan(socket, argv[i]);

		client_close(socket);
		return RET_OK;
	}

	return RET_ERROR;
}
