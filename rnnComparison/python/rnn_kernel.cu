#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>


#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


std::vector<torch::Tensor> rnn_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor init_h) {

  cudaStream_t inputStream;
  cudaStreamCreate(&inputStream);
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, inputStream);
  int batch = input.sizes()[0];
  int sequence_length = input.sizes()[1];
  int input_size = input.sizes()[2];
  int hidden_size = weights.sizes()[1];
  float alpha = 1.f;
  float beta = 0.f;
  const auto options = input.options();
  torch::Tensor output = torch::empty({batch, sequence_length, hidden_size}, options);

  
  // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
  //     sequence_length, hidden_size, input_size, 
  //     &alpha, (float *)input.data_ptr(), sequence_length,
  //     (float *)weights.data_ptr(), input_size, &beta, output.data_ptr<float>(), sequence_length);
  cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
      sequence_length, hidden_size, input_size, 
      &alpha, input.data_ptr<float>(), sequence_length, sequence_length*input_size,
      weights.data_ptr<float>(), input_size, 0, &beta, output.data_ptr<float>(), sequence_length, sequence_length*hidden_size, 1);
  

  return {output};
}

