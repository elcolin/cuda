# cuda
Getting started with CUDA: C++/C

    function<<<2, 4>>>()

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
