
#include <vector>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define RowMajorInd(i, j, num_c) (i * num_c + j)

float *getMatrix(const int rows, const int cols, const float val);
float *getTensor(const int batch, const int rows, const int cols, const float val);
void printMatrix(const float *matrix, const int rows, const int cols);
void rnn_forward(float *inputs, float *inputWeight, float *hiddenWeight, float *tmpH, float *tmpAct, float *output, const int batchSize, const int sequenceLen, const int inputSize, const int hiddenSize);

int main()
{
  const int batch = 2;
  const int sequenceLen = 3;
  const int inputSize = 2;
  const int hiddenSize = 1;
  float *input = getTensor(sequenceLen, batch, inputSize, 1.f);
  float *tmpH = getTensor(sequenceLen, batch, hiddenSize, 0.f);
  float *tmpAct = getTensor(sequenceLen, batch, hiddenSize, 0.f);
  float *output = getTensor(sequenceLen, batch, hiddenSize, 0.f);
  float *inputWeight = getMatrix(inputSize, hiddenSize, 0.f);
  inputWeight[0] = 1;
  inputWeight[1] = 1;
  float *hiddenWeight = getMatrix(hiddenSize, hiddenSize, 0.f);
  hiddenWeight[0] = 4.f;

  rnn_forward(input, inputWeight, hiddenWeight, tmpH, tmpAct, output, batch, sequenceLen, inputSize, hiddenSize);

  cudaDeviceSynchronize();
  printf("inputWeight\n");
  printMatrix(inputWeight, inputSize, hiddenSize);
  printf("hiddenWeight\n");
  printMatrix(hiddenWeight, hiddenSize, hiddenSize);
  for (int i = 0; i < sequenceLen; i++)
  {
    printf("input:\n");
    printMatrix(input + i * batch * inputSize, batch, inputSize);
    printf("tmpH:\n");
    printMatrix(tmpH + i * batch * hiddenSize, batch, hiddenSize);
    printf("tmpAct:\n");
    printMatrix(tmpAct + i * batch * hiddenSize, batch, hiddenSize);
    printf("output:\n");
    printMatrix(output + i * batch * hiddenSize, batch, hiddenSize);
  }
  return 0;
}

float *getMatrix(const int rows, const int cols, const float val)
{
  float *pf_matrix = nullptr;
  cudaMallocManaged((void **)&pf_matrix, sizeof(float) * rows * cols);
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      pf_matrix[RowMajorInd(i, j, cols)] = (float)rand() * val / RAND_MAX;
    }
  }
  return pf_matrix;
}

float *getTensor(const int batch, const int rows, const int cols, const float val)
{
  float *pf_matrix = nullptr;
  cudaMallocManaged((void **)&pf_matrix, sizeof(float) * rows * cols * batch);
  for (int b = 0; b < batch; b++)
  {
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        pf_matrix[b * rows * cols + RowMajorInd(i, j, cols)] = (float)rand() * val / RAND_MAX;
      }
    }
  }

  return pf_matrix;
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