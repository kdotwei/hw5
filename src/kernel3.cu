#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int maxIteration) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < maxIteration; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float upperX, float upperY, int width, int height, int maxIterations, int* output, int pixelPerThread) {
	// To avoid error caused by the floating number, use the following pseudo code
	//
	// float x = lowerX + thisX * stepX;
	// float y = lowerY + thisY * stepY;
    int idxX = blockIdx.x * blockDim.x * pixelPerThread + threadIdx.x * pixelPerThread;
    int idxY = blockIdx.y * blockDim.y * pixelPerThread + threadIdx.y * pixelPerThread;

    float dx = (upperX - lowerX) / width;
    float dy = (upperY - lowerY) / height;

    for (int px = 0; px < pixelPerThread; px++) {
        for (int py = 0; py < pixelPerThread; py++) {
            int x = idxX + px;
            int y = idxY + py;

            if (x >= width || y >= height) continue;

            float realX = lowerX + x * dx;
            float realY = lowerY + y * dy;
            int index = y * width + x;

            output[index] = mandel(realX, realY, maxIterations);
        }
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    int pixelPerThread = 2;
    int imageSize = resX * resY * sizeof(int);
    int *pinned;
    int *output;
    size_t pitch;

    // Allocate memory on the device
    cudaMallocPitch((void**)&output, &pitch, resX * sizeof(int), resY);

    // Allocate memory on the host
    cudaHostAlloc((void**)&pinned, imageSize, cudaHostAllocMapped);

    // Number of threads per block and the number of blocks
    dim3 blockSize(16, 16);
    dim3 gridSize((resX + (blockSize.x * pixelPerThread) - 1) / (blockSize.x * pixelPerThread), 
                  (resY + (blockSize.y * pixelPerThread) - 1) / (blockSize.y * pixelPerThread));

    mandelKernel<<<gridSize, blockSize>>>(lowerX, lowerY, upperX, upperY, resX, resY, maxIterations, output, pixelPerThread);

    // Waiting for all kernels
    cudaDeviceSynchronize();

    // Copy the results back to the host
    cudaMemcpy(pinned, output, imageSize, cudaMemcpyDeviceToHost);
    memcpy(img, pinned, imageSize);

    // Free memory
    cudaFree(output);
    cudaFreeHost(pinned);
}
