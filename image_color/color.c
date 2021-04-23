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

#include <vector>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#define WINDOW_RADIUS	1
#define WINDOW_WIDTH	(2*WINDOW_RADIUS + 1)
#define WINDOW_PIXELS	(WINDOW_WIDTH*WINDOW_WIDTH)
#define IMAGE_COLOR_SERVICE 0x0102


typedef Eigen::Triplet<double> TD;

float variance(const std::vector<float>& vals, float eps=0.01)
{
    float sum = 0;
    float squaredSum = 0;
    for (auto v : vals) {
        sum += v;
        squaredSum += v * v;
    }

    float n = vals.size();
    return squaredSum / n - (sum * sum) / (n * n) + eps;
}

void getNeighbours(int i, int j, int nrows, int ncols, std::vector<int>& neighbors)
{
	int s, m, n, dx, dy;
    neighbors.clear();
    for (dx = -WINDOW_RADIUS; dx <= WINDOW_RADIUS; dx += 1) {
        for (dy = -WINDOW_RADIUS; dy <= WINDOW_RADIUS; dy += 1) {
            m = i + dy;
            n = j + dx;
            if ((dx == 0 && dy == 0) || m < 0 || n < 0 || m >= nrows || n >= ncols)
                continue;

            s = m * ncols + n;
            neighbors.push_back(s);
        }
    }
}

inline void getWeights(float *Lc, int r,
	const std::vector<int>& neighbors,
	std::vector<float>& neighborsWeights,
	float gamma)
{

    neighborsWeights.clear();
    std::vector<float> neighborsValues;
    neighborsValues.reserve(neighbors.size() + 1);

    for (auto s : neighbors) {
        neighborsWeights.push_back((Lc[r] - Lc[s]) * (Lc[r] - Lc[s]));	// Squared Difference
        neighborsValues.push_back(Lc[s]);
    }
    neighborsValues.push_back(Lc[r]);

    float var = variance(neighborsValues);
    float normalizer = 0.0;
    for (auto& w : neighborsWeights) {
        w = (float)std::exp(- gamma * w / (2 * var));
        normalizer += w;
    }

    for (auto& w : neighborsWeights)
        w /= normalizer;
}


void setupProblem(TENSOR *input_tensor,
	Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
	Eigen::VectorXd& bu,
    Eigen::VectorXd& bv, float gamma)
{
	int i, j, k, r;
    float *Lc, *ac, *bc, *mc;	// Lab + mask channels

    int nrows = input_tensor->height;
    int ncols = input_tensor->width;
    int nPixels = nrows * ncols;
    A.resize(nPixels, nPixels);

    std::vector<TD> coefficients;
    coefficients.reserve(nPixels * 3);
    bu.resize(nPixels);
    bv.resize(nPixels);
    bu.setZero();
    bv.setZero();

    Lc = tensor_start_chan(input_tensor, 0, 0);
    ac = tensor_start_chan(input_tensor, 0, 1);
    bc = tensor_start_chan(input_tensor, 0, 2);
    mc = tensor_start_chan(input_tensor, 0, 3);

    const int numNeighbors = 8;
    std::vector<float> weights;
    weights.reserve(numNeighbors);
    std::vector<int> neighbors;
    neighbors.reserve(numNeighbors);
    for (i = 0; i < nrows; ++i) {
        for (j = 0; j < ncols; ++j) {
            r = i * ncols + j;
            getNeighbours(i, j, nrows, ncols, neighbors);
            getWeights(Lc, r, neighbors, weights, gamma);
            coefficients.push_back(TD(r, r, 1));	// dialog set
            for (k = 0; k < (int)neighbors.size(); ++k) {
                auto s = neighbors[k];
                auto w = weights[k];
                if (mc[s] > 0.5) {
                     // Move value to RHS of Ax = b
                     bu(r) += w * ac[s];
                     bv(r) += w * bc[s];
                } else {
                     coefficients.push_back(TD(r, s, -w));
                }
            }
        }
    }

    A.setFromTriplets(coefficients.begin(), coefficients.end());
}

/* nimage prototype
void color_lab2rgb(float L, float a, float b, BYTE *R, BYTE *G, BYTE *B)
void color_rgb2ycbcr(BYTE R, BYTE G, BYTE B, BYTE * y, BYTE * cb, BYTE * cr)
void color_ycbcrtorgb(BYTE * y, BYTE * cb, BYTE * cr, BYTE R, BYTE G, BYTE B)
void color_rgb2lab(BYTE R, BYTE G, BYTE B, float *L, float *a, float *b)
*/

TENSOR *tensor_lab2yuv(TENSOR *lab_tensor)
{
	int i, n, bat;
	float L, a, b;
	BYTE R, G, B, y, cb, cr;
	float *Lc, *ac, *bc, *Labmc, *yc, *uc, *vc, *yuvmc;	// channels
	TENSOR *yuv_tensor;

	// input with mask, output YUV + Mask channel
	CHECK_TENSOR(lab_tensor);

	yuv_tensor = tensor_create(lab_tensor->batch, 4, lab_tensor->height, lab_tensor->width);
	CHECK_TENSOR(yuv_tensor);

	n = lab_tensor->height * lab_tensor->width;

	for (bat = 0; bat < lab_tensor->batch; bat++) {
		Lc = tensor_start_chan(lab_tensor, bat, 0);
		ac = tensor_start_chan(lab_tensor, bat, 1);
		bc = tensor_start_chan(lab_tensor, bat, 2);

		yc = tensor_start_chan(yuv_tensor, bat, 0);
		uc = tensor_start_chan(yuv_tensor, bat, 1);
		vc = tensor_start_chan(yuv_tensor, bat, 2);


		for (i = 0; i < n; i++) {
			L = Lc[i]; a = ac[i]; b = bc[i];

			L += 0.5; L *= 100.0;
			a *= 110.0;
			b *= 110.0;
			color_lab2rgb(L, a, b, &R, &G, &B);
			color_rgb2ycbcr(R, G, B, &y, &cb, &cr);
			yc[i] = y;
			uc[i] = cb;
			vc[i] = cr;
		}

		// Copy Lab mask to YUV mask channel
		Labmc = tensor_start_chan(lab_tensor, bat, 3);
		yuvmc = tensor_start_chan(yuv_tensor, bat, 3);
		memcpy(yuvmc, Labmc, n * sizeof(float));
	}

	return yuv_tensor;
}

TENSOR *tensor_yuv2ab(TENSOR *yuv_tensor)
{
	int i, n, bat;
	float L, a, b;
	BYTE R, G, B, y, cb, cr;
	float *ac, *bc, *yc, *uc, *vc;	// channels
	TENSOR *ab_tensor;

	CHECK_TENSOR(yuv_tensor);

    ab_tensor = tensor_create(yuv_tensor->batch, 2, yuv_tensor->height, yuv_tensor->width);
    CHECK_TENSOR(ab_tensor);

    n = yuv_tensor->height * yuv_tensor->width;
    for (bat = 0; bat < yuv_tensor->batch; bat++) {
		yc = tensor_start_chan(yuv_tensor, bat, 0);
		uc = tensor_start_chan(yuv_tensor, bat, 1);
		vc = tensor_start_chan(yuv_tensor, bat, 2);

		ac = tensor_start_chan(ab_tensor, bat, 0);
		bc = tensor_start_chan(ab_tensor, bat, 1);

		for (i = 0; i < n; i++) {
			y = yc[i];
			cb = uc[i];
			cr = vc[i];
			color_ycbcr2rgb(y, cb, cr, &R, &G, &B);
			color_rgb2lab(R, G, B, &L, &a, &b);
			// L -= 50; L /= 100.0;
			a /= 110;
			b /= 110;
			// CheckPoint("YUV --> ab: a = %.2f, b = %.2f, R = %d, G = %d, B = %d, y = %d, cb = %d, cr = %d", 
			// 	a, b, R, G, B, y, cb, cr);

			ac[i] = a;
			bc[i] = b;
		}
    }

	return ab_tensor;
}

TENSOR *do_color(TENSOR *input_tensor)
{
	int i, n;
	float *uc, *vc;		// U, V Channel

	TENSOR *output_tensor, *yuv_tensor;

	CHECK_TENSOR(input_tensor);

	yuv_tensor = tensor_lab2yuv(input_tensor);
	CHECK_TENSOR(yuv_tensor);

	// input_tensor: Lab + Mask -- mask == 1, is color else gray
	//               L in [-0.5, 0.5], ab in [-1.0, 1.0]
	// Output: 2 Channels, fake ab !!!

    // Set up matrices for U and V channels
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;
    Eigen::VectorXd bu;
    Eigen::VectorXd bv;

    setupProblem(yuv_tensor, A, bu, bv, 2.0 /*gamma*/);

    // Solve for U, V channels
    syslog_info("Solving for U channel.");
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double> > solver;

    solver.compute(A);
    Eigen::VectorXd U = solver.solve(bu);
    if (solver.info() != Eigen::Success) {
        syslog_error("Solve for U channel.");
        return NULL;
    }

    syslog_info("Solving for V channel.");
    Eigen::VectorXd V = solver.solve(bv);
    if (solver.info() != Eigen::Success) {
        syslog_error("Solve for V channel.");
        return NULL;
    }

    // Update YUV ...
    n = input_tensor->height * input_tensor->width;
    uc = tensor_start_chan(yuv_tensor, 0, 1);
    vc = tensor_start_chan(yuv_tensor, 0, 2);
    for (i = 0; i < n; i++) {
    	uc[i] = (float)U[i];
    	vc[i] = (float)V[i];
    }
    // memcpy(uc, (float *)U.data(), n * sizeof(float));
    // memcpy(vc, (float *)V.data(), n * sizeof(float));

    // Convert UV to Fake-ab ...
    output_tensor = tensor_yuv2ab(yuv_tensor);
    tensor_destroy(yuv_tensor);

    syslog_info("Finish coloring...");

	return output_tensor;
}

int ClassicService(char *endpoint, int use_gpu)
{
	int socket, count;
	TENSOR *input_tensor, *output_tensor;

	(void)use_gpu;

	if ((socket = server_open(endpoint)) < 0)
		return RET_ERROR;

	count = 0;
	for (;;) {
		syslog_info("Service %d times", count);

		input_tensor = service_request(socket, IMAGE_COLOR_SERVICE);
		if (!tensor_valid(input_tensor))
			continue;

		// Real service ...
		time_reset();
		output_tensor = do_color(input_tensor);
		time_spend((char *)"Image coloring");

        service_response(socket, IMAGE_COLOR_SERVICE, output_tensor);
        tensor_destroy(output_tensor);

		tensor_destroy(input_tensor);

		count++;
	}

	syslog(LOG_INFO, "Service shutdown.\n");
	server_close(socket);

	return RET_OK;
}
