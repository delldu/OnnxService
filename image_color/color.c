/************************************************************************************
***
***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, Thu Apr 15 17:38:30 CST 2021
***
************************************************************************************/

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <syslog.h>

#include <nimage/image.h>

TENSOR *do_color(TENSOR *input_tensor)
{
	return NULL;
}

int ClassicService(char *endpoint, int use_gpu)
{
	int socket, reqcode, count, rescode;
	TENSOR *input_tensor, *output_tensor;

	(void)use_gpu;

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	count = 0;
	for (;;) {
		syslog_info("Service %d times", count);

		input_tensor = request_recv(socket, &reqcode);

		if (!tensor_valid(input_tensor)) {
			syslog_error("Request recv bad tensor ...");
			continue;
		}
		syslog_info("Request Code = 0x%x", reqcode);

		// Real service ...
		time_reset();

		output_tensor = do_color(input_tensor);

		time_spend((char *)"Predict");

		if (tensor_valid(output_tensor)) {
			rescode = reqcode;
			response_send(socket, output_tensor, rescode);
			tensor_destroy(output_tensor);
		}

		tensor_destroy(input_tensor);

		count++;
	}

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}
