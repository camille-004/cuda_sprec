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