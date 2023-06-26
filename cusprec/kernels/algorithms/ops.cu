__global__ void dotProduct(double *a, double *b, double *res, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        res[idx] = a[idx] * b[idx];
    }
}


__global__ void scalarMultiply(double *a, double scalar, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a[idx] *= scalar;
    }
}


__global__ void vectorAdd(double *a, double *b, double *res, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        res[idx] = a[idx] + b[idx];
    }
}


__global__ void vectorSubtract(double *a, double *b, double *res, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        res[idx] = a[idx] - b[idx];
    }
}
