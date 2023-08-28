#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

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

__global__ void tanhh(float *t, float *out)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  out[index] = tanh(t[index]);
}

void rnn_forward(float *inputs, float *inputWeight, float *hiddenWeight, float *tmpH, float *tmpAct, float *output, const int batchSize, const int sequenceLen, const int inputSize, const int hiddenSize)
{

  cudaStream_t inputStream;
  cudaStreamCreate(&inputStream);
  cudaStream_t hiddenStream;
  cudaStreamCreate(&hiddenStream);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaCheckErrors("streams");
  const float alpha = 1.f;
  const float beta = 0.f;
  const int maxSequence = 1;

  // inputs [sequence_len, batch, input_size]
  // weight [input_size, hidden_size]
  // tmpH [sequence_len, batch, hidden_size]
  // output [sequence_len, batch, hidden_size]
  cudaEvent_t *events_i;
  events_i = (cudaEvent_t *)malloc(sequenceLen * sizeof(cudaEvent_t));
  dim3 blockDim;
  dim3 gridDim;
  for (int batch = 0; batch < sequenceLen / maxSequence; batch++)
  {
    int sequenceIndex = batch * maxSequence;
    int inputOffset = sequenceIndex * batchSize * inputSize;
    int tmpHOffset = sequenceIndex * batchSize * hiddenSize;
    int outputOffset = sequenceIndex * batchSize * hiddenSize;

    int numElements = hiddenSize * maxSequence;
    blockDim.x = 128;
    gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

    cublasSetStream(handle, inputStream);
    cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        hiddenSize, sequenceLen, inputSize,
        &alpha,
        inputWeight, hiddenSize, 0,
        inputs + inputOffset, inputSize, batchSize * inputSize,
        &beta,
        tmpH + tmpHOffset, hiddenSize, batchSize * hiddenSize,
        maxSequence);

    // event triggers only when work currently scheduled on input stream done
    // because of this think you need an event for every possible batch
    cudaEventCreate(&events_i[sequenceIndex], cudaEventDisableTiming);
    cudaEventRecord(events_i[sequenceIndex], inputStream);

    cudaStreamWaitEvent(hiddenStream, events_i[sequenceIndex], 0);
    cublasSetStream(handle, hiddenStream);
    cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        hiddenSize, sequenceLen, hiddenSize,
        &alpha,
        hiddenWeight, hiddenSize, 0,
        tmpH + tmpHOffset, hiddenSize, batchSize * hiddenSize,
        &beta,
        tmpAct + outputOffset, hiddenSize, batchSize * hiddenSize,
        maxSequence);
    tanhh<<<gridDim, blockDim, 0, hiddenStream>>>(tmpAct, output);
    cudaCheckErrors("sgem");
  }
}
