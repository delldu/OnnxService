/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-05-01 20:14:13
***
************************************************************************************/


#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <nimage/image.h>

#define X_DIMENSIONS 512

// Gan Optimizer
typedef struct {
	float x1[X_DIMENSIONS];
	float y1;	// cost value1
	float x2[X_DIMENSIONS];
	float y2;	// cost value2

	float lr;	// learning rate
} OPTIMIZER;

int optimizer_setup(OPTIMIZER *optimizer, float lr);
int optimizer_sample(OPTIMIZER *optimizer, int i);
int optimizer_calculate(OPTIMIZER *optimizer, int i);
int optimizer_update(OPTIMIZER *optimizer);
int optimizer_running(OPTIMIZER *optimizer, int max_epochs, float min_epison);


#if defined(__cplusplus)
}
#endif

#endif	// _OPTIMIZER_H

