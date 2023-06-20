import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


kernel_code = """
#include <stdio.h>

__global__ void gen_data(float *X, float *y, int num_samples, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_samples) {
        for (int j = 0; j < num_features; j++) {
            X[idx * num_features + j] = idx + j;  // X[i][j] = i + j
        }
        
        y[idx] = idx + num_features;
    }
    
    printf("Block Index: %d\\n", blockIdx.x);
}
"""


def gen_data(_num_samples, _num_features):
    X_gpu = gpuarray.zeros((_num_samples, _num_features), dtype=np.float32)
    y_gpu = gpuarray.zeros(_num_samples, dtype=np.float32)

    mod = SourceModule(kernel_code)
    gen_data_kernel = mod.get_function("gen_data")

    block_dim = (256, 1, 1)
    grid_dim = ((_num_samples - 1) // block_dim[0] + 1, 1)  # Block size

    gen_data_kernel(X_gpu, y_gpu, np.int32(_num_samples), np.int32(_num_features), block=block_dim, grid=grid_dim)

    _X = X_gpu.get()
    _y = y_gpu.get()

    return _X, _y


num_samples = 400
num_features = 10
X, y = gen_data(num_samples, num_features)

print("X:", X)
print("y:", y)
