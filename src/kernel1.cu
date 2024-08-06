#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int maxIteration)
{
	float z_re = c_re, z_im = c_im;
	int i;
	for (i = 0; i < maxIteration; ++i)
	{

		if (z_re * z_re + z_im * z_im > 4.f)
			break;

		float new_re = z_re * z_re - z_im * z_im;
		float new_im = 2.f * z_re * z_im;
		z_re = c_re + new_re;
		z_im = c_im + new_im;
	}

	return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float upperX, float upperY, int width, int height, int maxIterations, int* output) {
	// To avoid error caused by the floating number, use the following pseudo code
	//
	// float x = lowerX + thisX * stepX;
	// float y = lowerY + thisY * stepY;
	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if the current thread's indices are outside the image dimensions
	if (idxX >= width || idxY >= height) return;

	float dx = (upperX - lowerX) / width;
	float dy = (upperY - lowerY) / height;

	float x = lowerX + idxX * dx;
	float y = lowerY + idxY * dy;
	int index = idxY * width + idxX;

	output[index] = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
	int imageSize = resX * resY * sizeof(int);
	int *output;

	// Allocate memory
    int *host_output = (int*)malloc(imageSize);
	cudaMalloc((void **)&output, imageSize);

	// Number of threads per block and the number of blocks
	dim3 blockSize(16, 16);
	dim3 gridSize((resX + blockSize.x - 1) / blockSize.x, (resY + blockSize.y - 1) / blockSize.y);

	mandelKernel<<<gridSize, blockSize>>>(lowerX, lowerY, upperX, upperY, resX, resY, maxIterations, output);

	// Copy the results back to the host
	cudaMemcpy(host_output, output, imageSize, cudaMemcpyDeviceToHost);
	memcpy(img, host_output, imageSize);

	// Free memory
	free(host_output);
	cudaFree(output);
}
