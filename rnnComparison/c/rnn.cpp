
#include <vector>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define RowMajorInd(i, j, num_c) (i * num_c + j)

float *getMatrix(const int rows, const int cols, const float val);
float *getMatrixFromArray(const int rows, const int cols, float* arr);
float *getTensor(const int batch, const int rows, const int cols, const float val);
void printMatrix(const float *matrix, const int rows, const int cols);
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
  const int hidden_size);

int main()
{
  const int batch = 1;
  const int sequence_len = 2;
  const int input_size = 2;
  const int hidden_size = 2;

  float *tmp_inputs = getMatrix(batch, hidden_size, 0.f);
  float *tmp_h = getMatrix(batch, hidden_size, 0.f);
  float *output = getTensor(sequence_len, batch, hidden_size, 0.f);

  // float *input = getTensor(sequence_len, batch, input_size, 0.123f);
  // float *input_weight = getMatrix(input_size, hidden_size, 5.f);
  // float *hidden_weight = getMatrix(hidden_size, hidden_size, 3.2f); 
  float* input_weight_val = new float[4]{0.3958f, 0.3258f, -0.7066, -0.3648};
  float* hidden_weight_val = new float[4]{0.6341, 0.1225, -0.3486, -0.6147};
  float* input_val = new float[4]{0.5894, 0.0409, 0.8995, 0.5538};

  float *input = getMatrixFromArray(2, 2, input_val);
  float *input_weight = getMatrixFromArray(2, 2, input_weight_val);
  float *hidden_weight = getMatrixFromArray(2, 2, hidden_weight_val);

  // [[[0.2016, 0.1752]],
  // [[0.0314, 0.0079]]]
  rnn_forward(
    input, 
    input_weight,
    hidden_weight,
    tmp_inputs,
    tmp_h,
    output,
    batch,
    sequence_len,
    input_size,
    hidden_size);

  cudaDeviceSynchronize();
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


float *getMatrixFromArray(const int rows, const int cols, float *arr)
{
  float *pf_matrix = nullptr;
  cudaMallocManaged((void **)&pf_matrix, sizeof(float) * rows * cols);
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      pf_matrix[RowMajorInd(i, j, cols)] = arr[RowMajorInd(i, j, cols)];
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
