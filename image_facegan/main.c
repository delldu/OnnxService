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

#include <nimage/image.h>
#include <nimage/nnmsg.h>

#include "engine.h"
#include "cmaes.h"

#define IMAGE_FACEGAN_REQCODE 0x0108
// #define IMAGE_FACEGAN_URL "ipc:///tmp/image_facegan.ipc"
#define IMAGE_FACEGAN_URL "tcp://127.0.0.1:9108"

#define LAMBDA 31
#define W_SPACE_DIM 512

#define DEBUG 1

double global_wsamples[LAMBDA][W_SPACE_DIM];

OrtEngine *decoder_engine = NULL;
OrtEngine *transformer_engine = NULL;
TENSOR *stand_tensor = NULL;

typedef struct {				/* Used as argument to fit_start() */
	pthread_t id;				/* ID returned by pthread_create() */
	int no;						/* Sub thread */
	double x[W_SPACE_DIM], y;
} THREAD_INFOS;

double mean_latent[512] = {
	8.4830e-02,  4.1098e-02,  1.9256e-01, -7.5094e-02, -1.3458e-01,
	2.1817e-02, -6.8644e-02, -7.7883e-02,  1.7172e-01, -1.1452e-01,
	2.2819e-01,  9.4412e-03, -6.2359e-03,  1.3021e-01, -1.3147e-01,
	9.4488e-02,  3.0364e-03, -2.1321e-02,  8.4927e-02,  4.9487e-02,
	-4.9538e-02,  1.9818e-02, -1.0243e-01,  5.9621e-02,  1.9991e-01,
	4.6897e-01, -2.3885e-02, -3.3779e-02,  1.9056e-01, -3.0895e-02,
	1.0867e-01, -4.5064e-03,  2.2426e-01, -4.7917e-02,  9.8036e-02,
	1.9697e-01,  9.2002e-02,  7.3522e-02, -1.0455e-01,  1.8454e-01,
	1.5007e-01,  3.1591e-01,  3.2471e-03,  1.6538e-01,  3.6159e-01,
	-4.1785e-02,  1.5278e-01,  1.8543e-02,  2.6636e-01, -9.4499e-02,
	1.4898e-01,  9.5262e-02,  3.3032e-01, -5.6259e-02, -2.3106e-02,
	4.7743e-02,  9.9213e-02,  1.7754e-01, -1.9073e-01, -1.1595e-01,
	1.2504e-01,  7.6667e-02, -8.6093e-02,  6.7258e-02,  3.6644e-01,
	1.5740e-01,  1.2706e-01,  2.2272e-01, -3.3734e-02,  4.0461e-02,
	-1.1815e-01,  2.9166e-01, -1.4067e-01,  2.4630e-02,  1.4290e-02,
	1.9825e-01,  8.7887e-02,  2.2971e-01,  4.2879e-03,  2.8721e-01,
	1.2499e-02,  5.6291e-02,  6.7759e-02,  1.7536e-01,  7.4099e-02,
	2.3163e-01,  1.2122e-01,  1.8258e-01, -1.1756e-01,  2.5942e-01,
	4.5996e-01, -1.3456e-01, -3.4101e-02,  4.4396e-02,  3.1534e-01,
	8.2883e-03, -1.4263e-02,  4.0294e-03,  5.1468e-01,  1.1869e-01,
	1.2020e-01, -3.8729e-03,  1.3493e-03,  8.2184e-02,  1.9917e-01,
	1.8893e-01,  3.5443e-01,  2.0440e-01, -2.3813e-02,  2.0227e-01,
	1.1214e-01,  3.0420e-01,  1.4679e-01,  3.8808e-01, -9.6624e-02,
	2.4352e-01,  6.0181e-02,  1.4821e-01,  1.7735e-01, -1.0533e-02,
	2.6383e-01, -9.9352e-02,  3.1166e-02,  1.0622e-01, -1.0758e-01,
	-8.7507e-03,  1.1166e-01,  1.9429e-01,  1.5687e-01, -3.2020e-02,
	3.0421e-01,  1.3008e-01, -9.2902e-02,  1.6443e-01, -5.7518e-02,
	2.4325e-01,  1.9362e-02,  1.1081e-01,  4.4340e-01,  4.6955e-02,
	1.4171e-01, -4.4659e-02,  1.5274e-02,  7.3536e-02,  1.7344e-02,
	1.8293e-01, -1.9022e-02,  8.4306e-01,  4.3058e-02, -9.9164e-02,
	3.3957e-01, -8.0080e-02, -6.0625e-02,  2.2162e-01, -1.2412e-01,
	-8.0780e-03, -9.6610e-02,  2.9888e-01, -4.6292e-02,  7.7922e-03,
	1.5430e-01,  5.2698e-02,  1.8992e-02, -8.5507e-02,  1.6664e-02,
	9.4493e-02,  1.6730e-01,  1.3974e-01,  1.6970e-02,  5.7089e-02,
	8.7165e-02,  5.3118e-01,  4.3345e-02, -4.8959e-02, -3.5279e-02,
	1.4819e-01,  1.0832e-01,  5.3760e-01,  7.8886e-02, -6.2950e-02,
	2.3599e-01,  9.4801e-02,  5.5615e-02,  1.1593e-01, -1.2515e-02,
	7.9356e-02,  2.1756e-01,  2.1762e-01,  2.7805e-02, -1.1177e-01,
	1.7397e-01,  2.1792e-01,  5.4922e-02,  1.4850e-01,  3.1234e-01,
	-7.2025e-02, -8.0162e-03,  4.6104e-01, -3.7381e-02,  1.5385e-01,
	1.7162e-01, -5.3768e-02,  3.7578e-02,  3.0134e-01,  1.6812e-02,
	2.2638e-01, -3.4771e-02,  1.6851e-01,  8.1745e-01,  2.6582e-01,
	9.6249e-02, -1.1573e-01,  2.0971e-01, -2.2099e-01,  1.0387e-01,
	-2.3135e-02,  9.8151e-02,  2.4741e-01, -5.8547e-02, -7.5555e-02,
	3.9370e-01,  8.3409e-03,  1.2557e-01,  4.0184e-01,  3.3254e-01,
	3.4706e-02,  2.5260e-01,  3.3387e-01, -2.2968e-03, -1.0202e-02,
	-4.9909e-02,  1.8996e-01,  3.6061e-01,  2.0583e-01,  8.2774e-01,
	2.4300e-03,  3.8618e-02,  2.6221e-01,  2.7235e-02,  7.6243e-02,
	-5.2726e-04,  2.0698e-01, -1.2651e-01,  1.1702e-01,  5.4611e-02,
	-4.7576e-02,  2.3936e-02,  3.3698e-02,  2.6663e-01,  3.4276e-01,
	-3.1519e-02,  1.3329e-01,  9.7807e-02,  2.7063e-01,  3.1617e-01,
	2.5982e-01,  1.6747e-01, -8.1630e-02,  2.7858e-01,  7.7747e-02,
	1.4680e-01, -2.7333e-02,  4.3970e-01,  2.4296e-01, -5.3814e-02,
	1.1073e-01,  5.4595e-02,  1.4501e-01,  1.3133e-01,  2.2417e-01,
	1.4034e-01,  4.0676e-03,  2.1410e-02,  5.9309e-02,  1.8137e-01,
	1.7580e-02,  4.5905e-02,  1.7510e-01,  4.2271e-02, -8.7794e-02,
	1.9441e-02,  2.4818e-01,  7.3644e-02,  7.9178e-02,  1.2826e-01,
	-4.4199e-02,  1.0379e-01,  7.1856e-02,  4.3477e-01, -1.4341e-02,
	-1.0264e-01,  4.3257e-02, -5.4480e-02, -6.0761e-02,  6.9822e-02,
	6.3505e-02,  2.0802e-01,  2.1721e-01, -7.7134e-02, -1.1579e-01,
	1.5399e-01,  4.9731e-02,  1.4815e-02,  7.8466e-02, -1.1241e-01,
	2.0666e-01, -1.7469e-02,  1.1742e-01,  1.0209e-01, -2.5396e-02,
	2.9184e-01, -2.0622e-02,  5.9615e-02, -4.9244e-02,  9.7362e-02,
	9.9386e-02,  5.3365e-01, -2.6542e-02, -8.9255e-02,  1.5330e-01,
	2.8208e-01, -8.8573e-02,  7.4447e-02,  1.6907e-01,  8.9489e-02,
	2.6132e-01,  2.3570e-02,  3.4512e-02,  4.6942e-01, -8.6127e-03,
	9.9991e-02,  1.9110e-01, -9.4797e-02,  5.4886e-01,  1.8502e-01,
	2.1281e-01,  3.6820e-01,  1.9804e-01,  1.2879e-01,  1.7257e-01,
	2.4053e-01,  1.9864e-01,  3.8389e-02,  2.0983e-01,  2.5263e-01,
	3.8904e-01, -1.0090e-01,  1.4147e-01, -8.0818e-02,  1.0151e-01,
	1.9857e-02,  9.7885e-02,  2.3759e-02, -1.0218e-02, -2.2252e-02,
	1.9990e-01, -5.8607e-02, -6.4849e-02, -1.7058e-02,  1.5481e-01,
	5.1081e-02,  7.9846e-02,  8.7371e-02,  1.0921e-01,  1.1965e-01,
	2.6707e-01,  1.0598e-01,  1.2430e-01, -6.9812e-02,  2.3538e-01,
	4.6897e-02, -8.4242e-02,  1.5158e-01, -3.1253e-02,  1.7321e-02,
	1.4586e-01,  1.4075e-01,  3.4242e-03,  1.2520e-01,  8.8899e-02,
	3.6138e-01, -5.7710e-02, -3.8686e-02,  2.4065e-02,  5.6271e-02,
	1.8232e-01,  2.7215e-01,  1.3907e-01,  2.5285e-01,  1.3109e-01,
	-1.0659e-01, -1.4986e-01,  3.4659e-02, -7.9830e-02, -7.9304e-02,
	1.2144e-01,  2.3409e-02,  5.9649e-01,  8.2018e-02,  1.3215e-01,
	7.8173e-02,  3.0339e-01,  1.2105e-01,  6.8202e-02,  2.2401e-01,
	-1.0142e-01,  6.9298e-02, -1.9762e-02,  2.7448e-01, -7.3805e-03,
	1.1603e-01, -6.2766e-02,  4.5327e-02,  1.2890e-02, -6.4058e-02,
	5.4195e-02,  1.3014e-01,  8.2370e-02,  3.1145e-02,  9.6107e-03,
	1.3727e-01, -1.5560e-01, -1.0273e-01, -2.7787e-02, -3.8329e-02,
	-4.0685e-02, -2.5487e-02, -9.3993e-02,  3.0849e-01,  1.8160e-02,
	1.4265e-01,  3.7515e-01,  9.2315e-02, -1.1960e-01, -2.8990e-02,
	-7.7351e-03,  1.2239e-01,  2.5584e-01,  1.1317e-01,  2.6836e-02,
	2.5823e-02,  2.9781e-02,  3.5556e-02,  2.6102e-01,  2.0994e-01,
	1.2207e-01, -9.7807e-02,  1.4850e-01,  4.0529e-01,  1.4837e-01,
	3.3622e-02, -1.3196e-01,  1.4660e-01,  2.0044e-01, -1.5387e-01,
	1.9025e-01, -1.2100e-01,  4.0949e-01,  1.2281e-01,  1.0709e-02,
	-7.1071e-03, -6.1068e-02, -7.1646e-02,  1.5699e-01, -6.6039e-03,
	3.5162e-01,  7.7275e-03,  1.4500e-01,  2.6994e-01,  3.1305e-01,
	-5.3026e-02,  2.2951e-01,  1.4557e-01,  9.7931e-02,  2.0464e-01,
	-1.4425e-01,  1.3696e-01,  1.0969e-01, -9.6187e-02,  1.2268e-01,
	1.6475e-01, -3.3234e-02,  4.9491e-01, -3.0356e-02,  1.2827e-01,
	1.7526e-01, -2.1192e-02, -1.9877e-02, -4.5981e-02, -9.1168e-03,
	-1.3615e-01,  1.0328e-01,  2.5700e-02,  3.0204e-01,  1.9561e-02,
	5.3035e-01,  8.1167e-03,  1.9744e-01,  2.8584e-01,  2.3859e-01,
	1.0997e-01,  9.5329e-02,  4.0536e-02,  5.3059e-02,  8.0592e-02,
	1.2514e-01,  2.1641e-02,  1.4516e-02,  1.0819e-01, -6.3348e-02,
	-3.9933e-02,  1.8881e-01
};

float random_gauss()
{
	// Marsaglia and Bray 1964
    static double V1, V2, S;
    static int phase = 0;
    double X;
     
    if ( phase == 0 ) {
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
}

void output_image(TENSOR *tensor, const char *prefix, int index)
{
	char filename[256];
	snprintf(filename, sizeof(filename), "output/%s-%04d.png", prefix, index);
	SaveTensorAsImage(tensor, filename);
}


TENSOR *random_wcode()
{
	int i;
	TENSOR *zcode_tensor, *wcode_tensor;

	CheckEngine(transformer_engine);

	// Create zcode tensor
	zcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(zcode_tensor);
	for (i = 0; i < W_SPACE_DIM; i++)
		zcode_tensor->data[i] = random_gauss();

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
	int i;
	float loss = 1.0;
	TENSOR *output_image_tensor, *output_tensor;

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
	output_image(output_image_tensor, "face-", (face_index++) % LAMBDA);
	#endif

	// Compute loss, maybe we need more complex method !!!
	loss = 0.0;
	for (i = 0; i < 256*256*3; i++) {
		loss += (stand_tensor->data[i] - output_tensor->data[i]) * 
				(stand_tensor->data[i] - output_tensor->data[i]);
	}
	loss /= (256*256*3);

	// Release tensors
	tensor_destroy(output_image_tensor);
	tensor_destroy(output_tensor);

	return loss;
}


static void *fit_start(void *arg)
{
	THREAD_INFOS *info = (THREAD_INFOS *)arg;

	CheckEngine(decoder_engine);

	// syslog_info("Running thread %d ...", info->no);
	// Create input tensor
	TENSOR *wcode_tensor;
	wcode_tensor = create_wtensor(info->x);
	info->y = wcode_loss(wcode_tensor);
	// Release tensors
	tensor_destroy(wcode_tensor);

	// syslog_info("Thread %d result = %.4f", info->no, info->y);


	return NULL;
}


// Transform zcode to wcode
int sample_wcodes()
{
	int i, gen;
	TENSOR *wcode_tensor;

	CheckEngine(transformer_engine);

	for (gen = 0; gen < LAMBDA; gen++) {
		wcode_tensor = random_wcode();
		check_tensor(wcode_tensor);
		// Save wcode
		for (i = 0; i < W_SPACE_DIM; i++) {
			global_wsamples[gen][i] = (double)wcode_tensor->data[i];

			// Trrunc
			if (global_wsamples[gen][i] < -5.0)
				global_wsamples[gen][i] = -5.0;

			if (global_wsamples[gen][i] > 5.0)
				global_wsamples[gen][i] = 5.0;
		}

		tensor_destroy(wcode_tensor);
	}

	return 0;
}

double *cmaes_search(int epochs)
{
	cmaes_t evo;
	int i, j, lambda, ret;
	double *cost_values, *xfinal, *const *pop;
	THREAD_INFOS *info;
	pthread_attr_t attr;
	double xstart[W_SPACE_DIM] = {0};
	double stddev[W_SPACE_DIM] = {1.0};

	for (i = 0; i < W_SPACE_DIM; i++) {
		xstart[i] = 0.0;
		stddev[i] = 1.0;
	}

	// cost_values = cmaes_init(&evo, W_SPACE_DIM dimmesion, xstart, stddev, 0, / * lambda */, "none");
	cmaes_init_para(&evo, W_SPACE_DIM /*dimmesion*/, xstart, stddev, 0, LAMBDA  /* lambda */, "none");

	evo.sp.stopMaxIter = epochs;	// stop after given number of iterations (generations)
	evo.sp.stopFitness.flg = 1;
	evo.sp.stopFitness.val = 1e-2;	// stop if function value is smaller than stopFitness
	evo.sp.stopTolFun = 1e-3;		// stop if function value differences are small
	cost_values =  cmaes_init_final(&evo);

	// dimension = (unsigned int) cmaes_Get(&evo, "dimension");
	syslog_info("%s", cmaes_SayHello(&evo));

	lambda = cmaes_Get(&evo, "lambda");

	info = (THREAD_INFOS *)calloc(lambda, sizeof(THREAD_INFOS));
	if (info == NULL)
		syslog_error("Allocte memory.");

	while (!cmaes_TestForTermination(&evo)) {
		/* generate lambda new search points, sample population */
		sample_wcodes();
		pop = cmaes_SamplePopulation(&evo, global_wsamples);	/* do not change content of pop */
		/* Initialize thread creation attributes */
		ret = pthread_attr_init(&attr);
		if (ret != 0)
			syslog_error("pthread_attr_init");

		// for (i = 0; i < lambda; ++i) {
		// 	cost_values[i] = fitfun(pop[i], dimension);	/* evaluate */
		// }
		for (i = 0; i < lambda; i++) {
			info[i].no = i + 1;
			for (j = 0; j < W_SPACE_DIM; j++)
				// info[i].x[j] = global_wsamples[i][j];
				info[i].x[j] = pop[i][j];
			info[i].y = 0.0;

			ret = pthread_create(&info[i].id, &attr, &fit_start, &info[i]);
			if (ret != 0)
				syslog_error("pthread_create");
		}
		ret = pthread_attr_destroy(&attr);
		if (ret != 0)
			syslog_error("pthread_attr_destroy");

		/* Now join with each thread */
		for (i = 0; i < lambda; i++) {
			ret = pthread_join(info[i].id, NULL);
			if (ret != 0)
				syslog_error("pthread_join");
		}

		// Save cost
		for (i = 0; i < lambda; i++) {
			cost_values[i] = info[i].y;
			// syslog_info("Cost %d = %.4f", i, cost_values[i]);
		}

		cmaes_UpdateDistribution(&evo, cost_values);	/* assumes that pop[i] has not been modified */

		// if ((int)evo.gen % 5 == 0) {
		// 	syslog_info("Progress %6.2f %%, loss = %.2lf ...", 
		// 		(float)(100.0 * evo.gen/epochs),  evo.rgFuncValue[evo.index[0]]);
		// 	// fflush(stdout);
		// }
		syslog_info("Progress %6.2f %%, loss = %.4lf ...", 
			(float)(100.0 * evo.gen/epochs),  evo.rgFuncValue[evo.index[0]]);
#if DEBUG
		// Xbest ...
		xfinal = cmaes_GetNew(&evo, "xbest");
		TENSOR *wcode_tensor = create_wtensor(xfinal);
		TENSOR *output_tensor = TensorForward(decoder_engine, wcode_tensor);
		output_image(output_tensor, "best-", evo.gen);
		tensor_destroy(output_tensor);
		tensor_destroy(wcode_tensor);
		free(xfinal);
#endif

	}
	free(info);

	syslog_info("Stop Condition:%s", cmaes_TestForTermination(&evo));	/* print termination reason */

	cmaes_WriteToFile(&evo, "all", "output/cmaes_results.txt");	/* write final results */

	/* get best estimator for the optimum, xmean */
	xfinal = cmaes_GetNew(&evo, "xmean");

	cmaes_exit(&evo);			/* release memory */

	return xfinal;
}

TENSOR *wcode_search(TENSOR *reference_tensor)
{
	int i;
	double *best;
	TENSOR *wcode_tensor;

	CHECK_TENSOR(reference_tensor);

	stand_tensor = reference_tensor;
	best = cmaes_search(10);

	// Save best to wcode
	wcode_tensor = tensor_create(1, 1, 1, W_SPACE_DIM);
	CHECK_TENSOR(wcode_tensor);
	for (i = 0; i < W_SPACE_DIM; i++)
		wcode_tensor->data[i] = (float)best[i];

	free(best);

	return wcode_tensor;
}

int FaceGanService(char *endpoint, int use_gpu)
{
	int socket, reqcode, lambda, rescode;
	TENSOR *input_tensor, *output_tensor, *wcode_tensor;

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	decoder_engine = CreateEngine("image_gandecoder.onnx", 0 /*not use_gpu for model bug !!!*/);
	CheckEngine(decoder_engine);

	transformer_engine = CreateEngine("image_gantransformer.onnx", 0 /*use_gpu*/);
	CheckEngine(transformer_engine);

	lambda = 0;
	for (;;) {
		syslog_info("Service %d times", lambda);

		input_tensor = request_recv(socket, &reqcode);

		if (!tensor_valid(input_tensor)) {
			syslog_error("Request recv bad tensor ...");
			continue;
		}
		syslog_info("-------------------- Request Code = %d", reqcode);

		// Real service ...
		time_reset();
		wcode_tensor = wcode_search(input_tensor); check_tensor(wcode_tensor);

		output_tensor = TensorForward(decoder_engine, wcode_tensor);
		time_spend((char *)"Infer");

		if (tensor_valid(output_tensor)) {
			rescode = reqcode;
#if 1	// For debug			
			IMAGE *image = image_from_tensor(output_tensor, 0);
			image_save(image, "output/last.png");
			image_destroy(image);
#endif
			response_send(socket, output_tensor, rescode);
			tensor_destroy(output_tensor);
		}

		tensor_destroy(wcode_tensor);

		tensor_destroy(input_tensor);

		lambda++;
	}
	DestroyEngine(transformer_engine);
	DestroyEngine(decoder_engine);

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}

int server(char *endpoint, int use_gpu)
{
	return FaceGanService(endpoint, use_gpu);
}

int facegan(int socket, char *input_file)
{
	int rescode;
	IMAGE *send_image, *resize_send_image;
	TENSOR *send_tensor, *recv_tensor;

	printf("Face Searching %s ...\n", input_file);

	send_image = image_load(input_file); check_image(send_image);
	resize_send_image = image_zoom(send_image, 256, 256, 1); check_image(resize_send_image);

	if (image_valid(resize_send_image)) {
		send_tensor = tensor_from_image(resize_send_image, 0);
		check_tensor(send_tensor);

		recv_tensor = OnnxRPC(socket, send_tensor, IMAGE_FACEGAN_REQCODE, &rescode);
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

float gradient_descent(TENSOR *wcode_tensor, float delta, int epochs)
{
	int i;
	float loss1, loss2, loss3, min;
	TENSOR *wcode_tensor1, *wcode_tensor3;

	min = 1.0;
	wcode_tensor1 = tensor_copy(wcode_tensor);
	check_tensor(wcode_tensor1);
	wcode_tensor3 = tensor_copy(wcode_tensor);
	check_tensor(wcode_tensor3);

	min = 1.0;
	while(epochs > 0 && min > 0.001 && delta > 0.001) {
		// Set w +/- delta
		for (i = 0; i < W_SPACE_DIM; i++) {
			wcode_tensor1->data[i] = wcode_tensor->data[i] - delta *random_gauss();
			wcode_tensor3->data[i] = wcode_tensor->data[i] + delta * random_gauss();
		}

		loss1 = wcode_loss(wcode_tensor1);
		loss2 = wcode_loss(wcode_tensor);
		loss3 = wcode_loss(wcode_tensor3);

		if (loss1 < loss2 && loss1 < loss3) {
			min = loss1;
			memcpy(wcode_tensor->data, wcode_tensor1->data, W_SPACE_DIM*sizeof(float));
		} else if (loss3 < loss1 && loss3 < loss2) {
			min = loss3;
			memcpy(wcode_tensor->data, wcode_tensor3->data, W_SPACE_DIM*sizeof(float));
		} else {
			min = loss2;
			delta /= 2.0;
		}
		epochs--;

		syslog_info("Searching .... left epochs = %d, min = %.4f, delta = %.4f", epochs, min, delta);
	}
	tensor_destroy(wcode_tensor1);
	tensor_destroy(wcode_tensor3);

	return min;
}

int test()
{
	TENSOR *wcode_tensor, *output_tensor;
	IMAGE *image;

	srand(time(NULL));

	transformer_engine = CreateEngine("image_gantransformer.onnx", 0);
	CheckEngine(transformer_engine);

	decoder_engine = CreateEngine("image_gandecoder.onnx", 0);
	CheckEngine(decoder_engine);

	time_reset();

	wcode_tensor = random_wcode();
	check_tensor(wcode_tensor);

	output_tensor = TensorForward(decoder_engine, wcode_tensor);
	time_spend((char *)"Infer");

	image = image_from_tensor(output_tensor, 0);
	image_save(image, "test.png");
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
	printf("    t, --test                   Test.\n");

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
		{"test", 0, 0, 't'},

		{0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);

	while ((optc = getopt_long(argc, argv, "h e: s: t", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'e':
			endpoint = optarg;
			break;
		case 's':
			running_server = 1;
			use_gpu = atoi(optarg);
			break;
		case 't':
			return test();
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
