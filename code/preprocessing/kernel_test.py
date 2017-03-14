import pycuda.gpuarray as gpuarray
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import numpy as np

num_threads_x = 32
num_threads_y = 32
num_blocks_x = 8
num_blocks_y = 8

mod = SourceModule("""
    #include <math.h>

    // Updates matrix A by adding a scalar c multiplied by another matrix B (i.e. A -= c * B)
    // Only operates on the rows specified by batch: A[batch_j] -= lr * B[batch_i]
    // batch is assumed to 1 x p, A and B have q columns, c is a scalar
    __global__ void BatchMatSubtractInplaceKernel(const int p, const int q, const float c, float *A, 
    const float *B, const int *batch) {
      int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
      if (batch_index >= p) return;
      int row = batch[batch_index];
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= q) return;

      atomicAdd(&A[row * q + col], -c * B[row * q + col]);
    }

    // Perform the update step for b/b_tilde using gradient descent
    // a[batch] -= lr * b[batch]
    __global__ void BatchVecSubtractInplaceKernel(const int p, const float lr, float *a, const float *b, const int *batch) {
      int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
      if (batch_index >= p) return;

      int ind = batch[batch_index];

      atomicAdd(&a[ind], -lr * b[ind]);
    }

    // For matrix A and vector b, multiply the i'th row of A by b[i] and store the result into the j'th row of C       
    // Performs this operations only for the corresponding arrays of integers batch_i, batch_j
    // batch is assumed to be 1 x p, A has q columns
    __global__ void BatchMatVecRowMultKernel(const int p, const int q, const float *A, const float *b, float *C, 
    const int *batch_i, const int *batch_j){           
      int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
      if (batch_index >= p) return;
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= q) return;                              
      int row_A = batch_i[batch_index];
      int row_C = batch_j[batch_index];                                                  
                                                                                                                       
      C[row_C * q + col] = A[row_A * q + col] * b[batch_index];                                                                    
    }          

    // Copies the values from vector a to b, operating only on item indices specified by batch
    // batch is assumed to be 1 x p
    __global__ void BatchCopyVectorKernel(const int p, const float *a, float *b, const int *batch) {
      int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
      if (batch_index >= p) return;

      int ind = batch[batch_index];
      b[ind] = a[ind];
    }

    // result = np.array([W[i].dot(W_tilde[j]) for i, j in batch])
    // result must start off as a zero array
    __global__ void BatchMatColDotKernel(const int p, const int q, const float *W, const float *W_tilde, 
    const int *batch_i, const int *batch_j, float *result) {
      int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
      if (batch_index >= p) return;
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col >= q) return; 

      int row_W = batch_i[batch_index];
      int row_W_tilde = batch_j[batch_index];

      atomicAdd(&result[batch_index], W[row_W * q + col] * W_tilde[row_W_tilde * q + col]);
    }
    """)

batchMatSubtractInplace = mod.get_function("BatchMatSubtractInplaceKernel")
batchVecSubtractInplace = mod.get_function("BatchVecSubtractInplaceKernel")
batchMatVecRowMult = mod.get_function("BatchMatVecRowMultKernel")
batchCopyVectorKernel = mod.get_function("BatchCopyVectorKernel")
batchMatColDot = mod.get_function("BatchMatColDotKernel")

def assertResultsClose(gpu_result, actual_result):
  assert np.allclose(gpu_result, actual_result), "GPU Result:\n" + repr(gpu_result) \
    + "\nActual Result:\n" + repr(actual_result) + "\n"

def testBatchCopyVectorKernel():
  batch_size = 128  
  num_elems = 1000

  a_cpu = np.random.rand(num_elems).astype('f')
  b_cpu = np.random.rand(num_elems).astype('f')
  batch = np.random.choice(np.arange(num_elems, dtype=np.int32), size=batch_size, replace=False)

  a_gpu = gpuarray.to_gpu(a_cpu)
  b_gpu = gpuarray.to_gpu(b_cpu)
  batch_gpu = gpuarray.to_gpu(batch)

  func = mod.get_function("BatchCopyVectorKernel")
  func(np.int32(batch_size), a_gpu, b_gpu, batch_gpu, block=(num_threads_x, num_threads_y, 1), \
    grid=(num_blocks_x, num_blocks_y))

  context.synchronize()

  b_cpu[batch] = a_cpu[batch]

  gpu_result = b_gpu.get()
  actual_result = b_cpu
  assertResultsClose(gpu_result, actual_result)

def testBatchMatSubtractInplaceKernel():
  batch_size = 128
  num_rows = 1000  
  num_cols = 200
  c = 0.05

  a_cpu = np.random.rand(num_rows, num_cols).astype('f')
  b_cpu = np.random.rand(num_rows, num_cols).astype('f')
  batch = np.random.choice(np.arange(num_rows, dtype=np.int32), size=batch_size, replace=False)

  a_gpu = gpuarray.to_gpu(a_cpu)
  b_gpu = gpuarray.to_gpu(b_cpu)
  batch_gpu = gpuarray.to_gpu(batch)

  func = mod.get_function("BatchMatSubtractInplaceKernel")
  func(np.int32(batch_size), np.int32(num_cols), np.float32(c), a_gpu, b_gpu, batch_gpu, \
    block=(num_threads_x, num_threads_y, 1), grid=(num_blocks_x, num_blocks_y))

  context.synchronize()

  a_cpu[batch] -= c * b_cpu[batch]

  gpu_result = a_gpu.get()
  actual_result = a_cpu
  assertResultsClose(gpu_result, actual_result)

def testBatchVecSubtractInplace():
  batch_size = 128
  num_rows = 1000  
  num_cols = 200
  c = 0.05

  a_cpu = np.random.rand(num_rows).astype('f')
  b_cpu = np.random.rand(num_rows).astype('f')
  batch = np.random.choice(np.arange(num_rows, dtype=np.int32), size=batch_size, replace=False)

  a_gpu = gpuarray.to_gpu(a_cpu)
  b_gpu = gpuarray.to_gpu(b_cpu)
  batch_gpu = gpuarray.to_gpu(batch)

  func = mod.get_function("BatchVecSubtractInplaceKernel")
  func(np.int32(batch_size), np.float32(c), a_gpu, b_gpu, batch_gpu, \
    block=(num_threads_x, num_threads_y, 1), grid=(num_blocks_x, num_blocks_y))

  context.synchronize()

  a_cpu[batch] -= c * b_cpu[batch]

  gpu_result = a_gpu.get()
  actual_result = a_cpu
  assertResultsClose(gpu_result, actual_result)

def testBatchMatVecRowMultKernel():
  batch_size = 128
  num_rows = 1000  
  num_cols = 200

  a_cpu = np.random.rand(num_rows, num_cols).astype('f')
  b_cpu = np.random.rand(num_rows).astype('f')
  c_cpu = np.random.rand(num_rows, num_cols).astype('f')
  batch_i = np.random.choice(np.arange(num_rows, dtype=np.int32), size=batch_size, replace=False)
  batch_j = np.random.choice(np.arange(num_rows, dtype=np.int32), size=batch_size, replace=False)

  a_gpu = gpuarray.to_gpu(a_cpu)
  b_gpu = gpuarray.to_gpu(b_cpu)
  batch_i_gpu = gpuarray.to_gpu(batch_i)
  batch_j_gpu = gpuarray.to_gpu(batch_j)
  c_gpu = gpuarray.to_gpu(c_cpu)

  func = mod.get_function("BatchMatVecRowMultKernel")
  func(np.int32(batch_size), np.int32(num_cols), a_gpu, b_gpu, c_gpu, batch_i_gpu, batch_j_gpu, \
    block=(num_threads_x, num_threads_y, 1), grid=(num_blocks_x, num_blocks_y))

  context.synchronize()

  c_cpu[batch_j] = (a_cpu[batch_i].T * b_cpu).T

  gpu_result = c_gpu.get()
  actual_result = c_cpu
  assertResultsClose(gpu_result, actual_result)

def testBatchMatColDotKernel():
  batch_size = 128
  num_rows = 1000  
  num_cols = 200
  c = 0.05

  a_cpu = np.random.rand(num_rows, num_cols).astype('f')
  b_cpu = np.random.rand(num_rows, num_cols).astype('f')
  batch_i = np.random.choice(np.arange(num_rows, dtype=np.int32), size=batch_size, replace=False)
  batch_j = np.random.choice(np.arange(num_rows, dtype=np.int32), size=batch_size, replace=False)
  batch = zip(batch_i, batch_j)

  a_gpu = gpuarray.to_gpu(a_cpu)
  b_gpu = gpuarray.to_gpu(b_cpu)
  batch_i_gpu = gpuarray.to_gpu(batch_i)
  batch_j_gpu = gpuarray.to_gpu(batch_j)
  result = gpuarray.zeros(batch_size, dtype=np.float32)

  func = mod.get_function("BatchMatColDotKernel")
  func(np.int32(batch_size), np.int32(num_cols), a_gpu, b_gpu, batch_i_gpu, batch_j_gpu, result \
    block=(num_threads_x, num_threads_y, 1), grid=(num_blocks_x, num_blocks_y))

  context.synchronize()

  gpu_result = result.get()
  actual_result = np.array([a_cpu[i].dot(b_cpu[j]) for i, j in batch])
  assertResultsClose(gpu_result, actual_result)

if __name__ == "__main__":
  # basic tests
  testBatchCopyVectorKernel()
  testBatchMatSubtractInplaceKernel()
  testBatchMatVecRowMultKernel()
  testBatchVecSubtractInplaceKernel()
  testBatchMatColDotKernel()
  # composite tests