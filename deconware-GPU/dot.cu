
#include <stdio.h>
#include <cuda_runtime.h>

#define imin(a,b) (a<b?a:b)

const int threadsPerBlock = 256;

__global__ void dot( float *a, float *b, float *c, int N ) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

extern "C" {
float dot_device(size_t N, float *in1, float *in2);
}

float dot_device(size_t N, float *in1, float *in2) 
{
	fprintf(stderr, "Dot device %d\n", N);
	float out=0.0;

	float *partial_out;
	float   *dev_in1, *dev_in2, *dev_partial_out;

	cudaError_t err;

	fprintf(stderr,"N %d\n", N);

	cudaDeviceReset();

	// calculate blocks per grid
	const int blocksPerGrid = imin( N, (128+threadsPerBlock-1) / threadsPerBlock );

	// allocate memory

	fprintf(stderr, "allocate memory\n");

	err = cudaMalloc(&dev_in1, N*sizeof(float));
	if(err) goto cudaErr;
    
	err = cudaMalloc(&dev_in2, N*sizeof(float));
	if(err) goto cudaErr;

	// allocate memory for the partial result on the GPU
	err = cudaMalloc( (void**)&dev_partial_out, blocksPerGrid*sizeof(float) );
	if(err) goto cudaErr;

	// allocate memory for the partial result on the CPU
	partial_out = (float*)malloc( blocksPerGrid*sizeof(float) );

	fprintf(stderr, "transfer memory\n");

	// transfer memory from cpu to gpu
 	err = cudaMemcpy(dev_in1, in1, N*sizeof(float), cudaMemcpyHostToDevice);
    if(err) goto cudaErr;
    
    err = cudaMemcpy(dev_in2, in2, N*sizeof(float), cudaMemcpyHostToDevice);
    if(err) goto cudaErr;

	fprintf(stderr, "run kernel\n");

	// run the kernel
	dot<<<blocksPerGrid,threadsPerBlock>>>( dev_in1, dev_in2, dev_partial_out, N);

	// copy the partial reduction back from the GPU
 	cudaMemcpy( partial_out, dev_partial_out,
                              blocksPerGrid*sizeof(float),
                              cudaMemcpyDeviceToHost );

 	// finish up on the CPU side
    for (int i=0; i<blocksPerGrid; i++) {
        out += partial_out[i];
    }

	cudaErr:
	    fprintf(stderr, "Dot Device: CUDA error: %d\n", err);
	    goto cleanup;

	cleanup:
		if(dev_in1) cudaFree(dev_in1);
	    if(dev_in2) cudaFree(dev_in2);
		if(dev_partial_out) cudaFree(dev_partial_out);
		free( partial_out );
	    cudaDeviceReset();
	    return out;
}



