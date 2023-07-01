#define N 256

__global__ void dot(int *a, int *b, int *c) {
    __shared__ int temp[N];
    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
    __syncthreads();
    if (0 == threadIdx.x) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += temp[i];
        }
        *c = sum;
    }
}

__global__ void add(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}

__global__ void sub(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] - b[idx];
}

__global__ void scalarMultiply(float *a, float scalar, float* res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        res[idx] = scalar * a[idx];
    }
}
