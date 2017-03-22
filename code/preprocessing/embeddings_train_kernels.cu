// Calculates the product of matrices A and B, and stores it in C
// A is assumed to be a p x q matrix, and B is a q x r matrix
__global__ void MatMulKernel(const int p, const int q, const int r, const float *A, const float *B, float *C){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= p || col >= r) return;

  float sum = 0.0;
  for (int i = 0; i < q; ++i) {
    val += A[row * q + i] * B[i * r + col];
  }
  C[row * r + col] = sum;
}

// Computer cost_inner in embeddings_train
// batch is assumed to be 1 x p, W and W_tilde have q columns, b 
__global__ void BatchCostInnerKernel(const int p, const int q, const float *A, const float *b, float *C){
  
}

// Copies the values from vector a to b, operating only on item indices specified by batch
// batch is assumed to be 1 x p
__global__ void BatchCopyVectorKernel(const int p, const float *a, const float *b, const int *batch) {
  int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_index >= p) return;

  index = batch[batch_index]
  b[index] = a[index];
}

// For matrix A and vector b, multiply the i'th row of A by b[i] and store the result into the i'th row of C
// Performs this operation for all rows of A
// A is assumed to be a p x q matrix and b is 1 x p                                                                
__global__ void MatVecRowMultKernel(const int p, const int q, const float *A, const float *b, float *C){           
  int row = blockIdx.y * blockDim.y + threadIdx.y;                                                                 
  int col = blockIdx.x * blockDim.x + threadIdx.x;                                                                 
  if (row >= p || col >= q) return;                                                                                
                                                                                                                   
  C[row * q + col] = A[row * q + col] * b[row];                                                                    
}          

// For matrix A and vector b, multiply the i'th row of A by b[i] and store the result into the j'th row of C       
// Performs this operations only for the corresponding arrays of integers batch_i, batch_j
// batch is assumed to be 1 x p, A has q columns
__global__ void BatchMatVecRowMultKernel(const int p, const int q, const float *A, const float *b, float *C, const int *batch_i, const int *batch_j){           
  int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
  if (batch_index >= p) return;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= q) return;                              
  int row_A = batch_i[batch_index];
  int row_C = batch_j[batch_index];                                                  
                                                                                                                   
  C[row_C * q + col] = A[row_A * q + col] * b[row_A];                                                                    
}          

// Updates matrix A by adding a scalar c multiplied by another matrix B (i.e. A -= c * B)
// A and B are assumed to be p x q matrices
__global__ void MatSubtractInplaceKernel(const int p, const int q, const float c, float *A, const float *B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= p || col >= q) return;

  A[row * q + col] -= c * B[row * q + col];
}

// Updates matrix A by adding a scalar c multiplied by another matrix B (i.e. A -= c * B)
// Only operates on the rows specified by batch
// batch is assumed to 1 x p, A and B have q columns, c is a scalar
__global__ void BatchMatSubtractInplaceKernel(const int p, const int q, const float c, float *A, const float *B, const int *batch) {
  int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
  if (batch_index >= p) return;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= q) return;
  int row = batch[batch_index];

  A[row * q + col] -= c * B[row * q + col];
}

