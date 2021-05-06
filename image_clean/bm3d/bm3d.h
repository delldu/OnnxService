/************************************************************************************
***
***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, 2020-11-20 16:50:08
***
************************************************************************************/

#ifndef _BM3D_H
#define _BM3D_H

#if defined(__cplusplus)
extern "C"
{
#endif

#define RET_OK 0
#define RET_ERROR (-1)
    int bm3d(unsigned char *imgdata, int channels, int height, int width, int sigma, unsigned char *outdata, int debug);

#if defined(__cplusplus)
}
#endif

#endif // _BM3D_H
