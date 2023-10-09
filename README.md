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

# dim3
dim3 foo(x, y, z);

    dim3 threads_per_block(16, 16, 1);  // 16x16 threads per block
    dim3 number_of_blocks(32, 32, 1);    // 32x32 blocks in the grid

    myKernel<<<number_of_blocks, threads_per_block>>>(...);

    ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

# nsys
    !nsys profile --stats=true ./a.out
# Device Properties
Get the properties of the device with **cudaDeviceProp** class.

    cudaGetDeviceProperties(&deviceProp, 0);

0 stands for the first CUDA device assuming there can be several.

    #include <stdio.h>
  
    int main()
    {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, 0);
      int deviceId = deviceProp.pciDeviceID;
      int computeCapabilityMajor = deviceProp.major;
      int computeCapabilityMinor = deviceProp.minor;
      int multiProcessorCount = deviceProp.multiProcessorCount;
      int warpSize = deviceProp.warpSize;

      printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp     Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
      }


## SMs

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);  // Assumes device 0
    numSMs = deviceProp.multiProcessorCount;
    int adjustedGridDim = (desiredGridDim / numSMs) * numSMs;

Good practice:

    int number_of_blocks = (N + threads_per_block - 1) / threads_per_block; // Adjusted block calculation

## Optimization

    int maxThreadsPerBlock;  // Maximum threads per block for the GPU
    int maxThreadsPerMultiprocessor;  // Maximum threads per multiprocessor for the GPU

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);  // Get device properties for GPU 0

    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    maxThreadsPerMultiprocessor = deviceProp.maxThreadsPerMultiProcessor;

    int maxBlocksPerMultiprocessor = maxThreadsPerMultiprocessor / maxThreadsPerBlock;

# Asynchronous Memory Prefetching

    int deviceId;
    cudaGetDevice(&deviceId);                                         // The ID of the currently active GPU device.

    cudaMemPrefetchAsync(pointerToSomeUMData, size, deviceId);        // Prefetch to GPU device.
    cudaMemPrefetchAsync(pointerToSomeUMData, size, cudaCpuDeviceId); // Prefetch to host. `cudaCpuDeviceId` is a
                                                                  // built-in CUDA variable.


# Rules Governing the Behavior of CUDA Streams
There are a few rules, concerning the behavior of CUDA streams, that should be learned in order to utilize them effectively:

Operations within a given stream occur in order.
Operations in different non-default streams are not guaranteed to operate in any specific order relative to each other.
The default stream is blocking and will both wait for all other streams to complete before running, and, will block other streams from running until it completes.
