  # cuda
Getting started with CUDA: C++/C

    function<<<2, 4>>>()
     
      threadsPerBlock = 256;
      numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

# General Information

## gridDim.x
Is the number of blocks in a grid, 2 in this case.

## blockIdx.x
Is the index of the current block within the grid.

## blockDim.x
Is the number of threads in a block, 4 in our case.

## threadIdx.x
The index of the thread within a block.

## Getting a general index
To access a certain memory space as a[i].

    i = threadIdx.x + blockIdx.x * blockDim.x

# Grid-Stride Loops
To perform a grid-stride loop, you have to go forward with

    blockDim.x * gridDim.x

8 in our case.

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < DATA ; i += (blockIdx.x * blockDim.x))

# Malloc Cuda

    cudaMallocManaged(void *a, size_t size);

## Free

    cudaFree(a);

# Errors

## Type
    cudaError_t Err;

## ASynchronized Error
Occurs during kernel bad execution, can be retrieved by getting **cudaDeviceSynchronize()** return value.

    cudaError_t AsyncErr = cudaDeviceSynchronize();

## Synchronized Error
Occurs during kernel launch, such as too many requested threads or blocks, can be retrieved with **cudaGetLastError()** return value.

    cudaError_t syncErr = cudaGetLastError();

## Display Error

        if(Err != cudaSuccess)
            printf("%s\n", cudaGetErrorString(Err));

### Function example

    inline cudaError_t checkCuda(cudaError_t result)
    {
      if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
      }
      return result;
    }


#include <stdio.h>

#define N  64

__global__ void matrixMulGPU( int * a, int * b, int * c )
{
  
}

/*
 * This CPU function already works, and will run to create a solution matrix
 * against which to verify your work building out the matrixMulGPU kernel.
 */

void matrixMulCPU( int * a, int * b, int * c )
{
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

int main()
{
  int *a, *b, *c_cpu, *c_gpu; // Allocate a solution matrix for both the CPU and the GPU operations

  int size = N * N * sizeof (int); // Number of bytes of an N x N matrix

  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  // Initialize memory; create 2D matrices
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  /*
   * Assign `threads_per_block` and `number_of_blocks` 2D values
   * that can be used in matrixMulGPU above.
   */

  dim3 threads_per_block;
  dim3 number_of_blocks;

  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

  cudaDeviceSynchronize();

  // Call the CPU version to check our work
  matrixMulCPU( a, b, c_cpu );

  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");

  // Free all our allocated memory
  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
}

