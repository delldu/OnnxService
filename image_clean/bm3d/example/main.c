/************************************************************************************
***
***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2020-11-20 13:13:09
***
************************************************************************************/
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>

#include "nimage/image.h"
#include "bm3d.h"

#define COLOR_IMAGE_CHANNELS 3
#define DEFAULT_NOISE_LEVEL 5

IMAGE *image_frombm3d(unsigned char *data, int channels, int height, int width)
{
	int i, j;
	IMAGE *image = NULL;
	unsigned char *d;

	if (channels != 1 && channels != 3)
		return NULL;

	image = image_create(height, width); CHECK_IMAGE(image);

	d = data;
	switch (channels) {
	case 1:
		image_foreach(image, i, j)
			image->ie[i][j].r = image->ie[i][j].g = image->ie[i][j].b = *d++;
		break;
	case 3:
		image_foreach(image, i, j)
			image->ie[i][j].r = *d++;
		image_foreach(image, i, j)
			image->ie[i][j].g = *d++;
		image_foreach(image, i, j)
			image->ie[i][j].b = *d++;
		break;
	default:
		// Error ?
		syslog_error("Strange channels: %d\n", channels);
		break;
	}

	return image;
}

// Set image to bm3d
int image_tobm3d(IMAGE * image, unsigned char *data, int channels)
{
	int i, j;
	unsigned char *d;

	check_image(image);

	d = data;
	switch (channels) {
	case 1:
		image_foreach(image, i, j)
			*d++ = image->ie[i][j].g;
		break;
	case 3:
		image_foreach(image, i, j)
			*d++ = image->ie[i][j].r;
		image_foreach(image, i, j)
			*d++ = image->ie[i][j].g;
		image_foreach(image, i, j)
			*d++ = image->ie[i][j].b;
		break;
	default:
		// Error ?
		syslog_error("Strange channels: %d\n", channels);
		break;
	}

	return 0;
}


void help(char *cmd)
{
	printf("Usage: %s [option]\n", cmd);
	printf("    -h, --help                   Display this help.\n");
	printf("    -i, --input <file>           Input image file.\n");
	printf("    -o, --output <file>          Output image file.\n");
	printf("    -s, --sigma <number>         Noise level (default: %d).\n", DEFAULT_NOISE_LEVEL);

	exit(1);
}

int main(int argc, char **argv)
{
	int optc, ret;
	int option_index = 0;
	char *input_file = NULL;
	char *output_file = NULL;
	unsigned int sigma = DEFAULT_NOISE_LEVEL;

	struct option long_opts[] = {
		{ "help", 0, 0, 'h'},
		{ "input", 1, 0, 'i'},
		{ "output", 1, 0, 'o'},
		{ "sigma", 1, 0, 's'},
		{ 0, 0, 0, 0}
	};

	if (argc <= 1)
		help(argv[0]);

	while ((optc = getopt_long(argc, argv, "h i: o: s:", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'i':
			input_file = optarg;
			break;
		case 'o':
			output_file = optarg;
			break;
		case 's':
			sigma = (unsigned)atoi(optarg);
			break;
		case 'h':	// help
		default:
			help(argv[0]);
			break;
	    }
	}

	if (! input_file) {
		help(argv[0]);
	}
	if (! output_file)
		output_file = (char *)"result.png";

	IMAGE *image = image_load(input_file); check_image(image);
	unsigned char *imgdata = (unsigned char *)malloc(3 * image->height * image->width);
	unsigned char *outdata = (unsigned char *)malloc(3 * image->height * image->width);

	image_tobm3d(image, imgdata, COLOR_IMAGE_CHANNELS);


	ret = bm3d(imgdata, COLOR_IMAGE_CHANNELS, image->height, image->width, sigma, outdata, 1);

	if (ret == RET_OK) {
		IMAGE *denoised = image_frombm3d(outdata, COLOR_IMAGE_CHANNELS, image->height, image->width);
		check_image(denoised);

		image_save(denoised, output_file);
		image_destroy(denoised);
	}

	free(imgdata);
	free(outdata);

	image_destroy(image);

	return ret;
}
