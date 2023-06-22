import matplotlib.pyplot as plt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


n = 1024
m = 512
s = 10

mod = SourceModule("""
    __global__ void initialize(
        float *x, 
        float *A, 
        float *b, 
        float *rand_vals,
        int *rand_indices, 
        int m, 
        int n, 
        int s
    ) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        
        if (idx < s) {
            int i = rand_indices[idx];
            x[i] = rand_vals[i * (m + 1)];
        }

        if (idx < n) {            
            for (int i = 0; i < m; i++) {
                A[i * n + idx] = rand_vals[idx + i + 1];
            }
            
            float sum_squares = 0.0f;
            for (int i = 0; i < m; i++) {
                sum_squares += A[i * n + idx] * A[i * n + idx];
            }
            float norm = sqrtf(sum_squares);
            
            for (int i = 0; i < m; i++) {
                A[i * n + idx] /= norm;
            }
        }
        
        if (idx < m) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += A[idx * n + j] * x[j];
            }
            b[idx] = sum;
        }
    }
""")

initialize = mod.get_function("initialize")
rand_vals = np.random.randn(n * (m + 1)).astype(np.float32)
rand_indices = np.random.choice(n, s, replace=False).astype(np.int32)

x = np.zeros(n, dtype=np.float32)
x[rand_indices] = rand_vals[:s]

# Allocate memory.
# For x, Allocate n * 4, (4 bytes for np.float32).
num_bytes = np.dtype("float32").itemsize
x_gpu = cuda.mem_alloc(n * num_bytes)
A_gpu = cuda.mem_alloc(m * n * num_bytes)
b_gpu = cuda.mem_alloc(m * num_bytes)
rand_vals_gpu = cuda.mem_alloc(rand_vals.nbytes)
rand_indices_gpu = cuda.mem_alloc(rand_indices.nbytes)

# Copy from host to device.
cuda.memcpy_htod(rand_vals_gpu, rand_vals)
cuda.memcpy_htod(rand_indices_gpu, rand_indices)

# Generate data.
num_blocks = (n + 255) // 256
initialize(
    x_gpu,
    A_gpu,
    b_gpu,
    rand_vals_gpu,
    rand_indices_gpu,
    np.int32(m),
    np.int32(n),
    np.int32(s),
    block=(256, 1, 1),
    grid=(num_blocks, 1)
)

# Copy back to CPU.
x = np.empty(n, dtype=np.float32)
A = np.empty((m, n), dtype=np.float32)
b = np.empty(m, dtype=np.float32)
cuda.memcpy_dtoh(x, x_gpu)  # Device to host.
cuda.memcpy_dtoh(A, A_gpu)
cuda.memcpy_dtoh(b, b_gpu)

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.stem(x)
plt.title("Sparse Signal (x)")

plt.subplot(1, 3, 2)
plt.imshow(A, aspect="auto", cmap="viridis")
plt.colorbar(label="Value")
plt.title("Measurement Matrix (A)")

plt.subplot(1, 3, 3)
plt.stem(b)
plt.title("Observed Signal (b)")

plt.tight_layout()
plt.show()
