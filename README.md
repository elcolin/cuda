# cuda
Getting started with CUDA: C++/C

    function<<<2, 4>>>()

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

    for (int i = threadIdx.x; i < DATA ; i += (blockIdx.x * blockDim.x))

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
