/************************************************************************************
***
***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2021-05-01 20:14:33
***
************************************************************************************/


#include "optimizer.h"

int optimizer_setup(OPTIMIZER *optimizer, float lr)
{
	optimizer->lr = lr;
	return lr > 0.0 and lr < 1.0;
}

int optimizer_sample(OPTIMIZER *optimizer, int i)
{
	int j;
	if (i < 0 || i >= 2)
		return RET_ERROR;
	if (i == 0) {
		for (j = 0; j < X_DIMENSIONS; j++)
			optimizer->x1[j] = random_gauss();
	} else {
		for (j = 0; j < X_DIMENSIONS; j++)
			optimizer->x2[j] = random_gauss();
	}
	optimizer_calculate(optimizer, i);
	return RET_OK;
}

int optimizer_calculate(OPTIMIZER *optimizer, int i)
{
	if (i < 0 || i >= 2)
		return RET_ERROR;

	return RET_OK;
}

int optimizer_update(OPTIMIZER *optimizer)
{
	float i;

	if (y1 > y2) {
		//  x2 - lr * (x1 - x2) ==> x1
		for (i = 0; i < X_DIMENSIONS; i++)
			optimizer->x1[i] = optimizer->x2[i] - optimizer->lr * (optimizer->x1[i] - optimizer->x2[i]);

		optimizer_calculate(optimizer, 0);	// Update y1
		return RET_OK;
	} else if (y2 > y1) {
		//  x1 - lr * (x2 - x1) ==> x2
		for (i = 0; i < X_DIMENSIONS; i++)
			optimizer->x2[i] = optimizer->x1[i] - optimizer->lr * (optimizer->x2[i] - optimizer->x1[i]);
		optimizer_calculate(optimizer, 1);	// Update y2

		return RET_OK;
	}

	// now y1 == y2, we could not update !!!
	return RET_ERROR;
}

int optimizer_running(OPTIMIZER *optimizer, int max_epochs, float min_epison)
{
	int epoch = 0;

	optimizer_sample(optimizer, 0);
	optimizer_sample(optimizer, 1);

	while(epoch < max_epochs && MIN(optimizer->y1, optimizer->y2) > min_epison) {
		// if we got local min, could not more optimizing, breaking ...
		if (optimizer_update(optimizer) != RET_OK)
			break;
		epoch++;
	}

	return (epoch < max_epochs)? RET_OK : RET_ERROR;
}
