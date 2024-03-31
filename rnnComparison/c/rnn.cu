#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

#define RowMajorInd(i, j, num_c) (i * num_c + j)

void printGPUArray(float *array, int size)
{
  float *host_array = (float *)malloc(size * sizeof(float));
  cudaMemcpy(host_array, array, size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; i++)
  {
    std::cout << host_array[i] << " ";
  }
  std::cout << std::endl;
}
void printMatrix(const float *matrix, const int rows, const int cols)
{
  float elem;
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      elem = matrix[RowMajorInd(i, j, cols)];
      std::cout << std::fixed << std::setw(8) << std::setprecision(4) << elem;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
#define cudaCheckErrors(msg)                             \
  do                                                     \
  {                                                      \
    cudaError_t __err = cudaGetLastError();              \
    if (__err != cudaSuccess)                            \
    {                                                    \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
              msg, cudaGetErrorString(__err),            \
              __FILE__, __LINE__);                       \
      fprintf(stderr, "*** FAILED - ABORTING\n");        \
      exit(1);                                           \
    }                                                    \
  } while (0)

__global__ void tanhh(
  float *tmp_inputs, 
  float *tmp_h, 
  float *out, 
  int hidden_dim, 
  int batch_size, 
  int sequence_index)
{
  int offset = hidden_dim*batch_size;
  int sequence_start = sequence_index*offset;
  int sequence_end = sequence_start + offset;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int global_index = index + sequence_start;
  if (global_index >= sequence_end){
    return;
  }
  out[global_index] = tanh(tmp_h[index] + tmp_inputs[global_index]);
}

void rnn_forward(
  float *inputs, 
  float *input_weight, 
  float *hidden_weight, 
  float *tmp_inputs, 
  float *tmp_h, 
  float *output, 
  const int batch_size, 
  const int sequence_len, 
  const int input_size, 
  const int hidden_size)
{

  cudaStream_t input_stream;
  cudaStreamCreate(&input_stream);
  cudaStream_t hidden_stream;
  cudaStreamCreate(&hidden_stream);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaCheckErrors("streams");
  const float alpha = 1.f;
  const float beta = 0.f;
  const int max_sequence = 1;

  // inputs [sequence_len, batch, input_size]
  // weight [input_size, hidden_size]
  // tmpH [sequence_len, batch, hidden_size]
  // hidden_state [batch, hidden_size]
  // output [sequence_len, batch, hidden_size]
  // std::cout << "input_size" << input_size << std::endl;
  // std::cout << "seq" << sequence_len << std::endl;
  // std::cout << "hidden_size" << hidden_size << std::endl;
  // std::cout << "batch_size" << batch_size << std::endl;
  // printGPUArray(inputs, sequence_len * batch_size * input_size);
  // printGPUArray(input_weight, input_size * hidden_size);
  // printGPUArray(hidden_weight, hidden_size * hidden_size);
  cudaEvent_t *events_i;
  events_i = (cudaEvent_t *)malloc(sequence_len * sizeof(cudaEvent_t));
  dim3 blockDim;
  dim3 gridDim;
  for (int seq_index = 0; seq_index < sequence_len; seq_index++)
  {
    int input_offset = seq_index * batch_size * input_size;
    int hidden_offset = seq_index * batch_size * hidden_size;
    cublasSetStream(handle, input_stream);
    cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, sequence_len, input_size,
        &alpha,
        input_weight, hidden_size, 0,
        inputs + input_offset, input_size, batch_size * input_size,
        &beta,
        tmp_inputs + hidden_offset, hidden_size, batch_size * hidden_size,
        max_sequence);

    // event triggers only when work currently scheduled on input stream done
    // because of this think you need an event for every possible batch
    cudaEventCreate(&events_i[seq_index], cudaEventDisableTiming);
    cudaEventRecord(events_i[seq_index], input_stream);

    cudaStreamWaitEvent(hidden_stream, events_i[seq_index], 0);
    cublasSetStream(handle, hidden_stream);
    int output_offset = (seq_index - 1) * batch_size * hidden_size;
    if (seq_index > 0) {
      cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, hidden_size,
        &alpha,
        hidden_weight, hidden_size,
        output + output_offset, hidden_size,
        &beta,
        tmp_h, hidden_size);
    }
    int numElements = batch_size*hidden_size;
    blockDim.x = 32;
    gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
    tanhh<<<gridDim, blockDim, 0, hidden_stream>>>(
      tmp_inputs, 
      tmp_h, 
      output,
      hidden_size, 
      batch_size, 
      seq_index);
    cudaCheckErrors("sgem");
  }
}

