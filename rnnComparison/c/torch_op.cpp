#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void rnn_forward(
  float *inputs, 
  float *input_weight, 
  float *hidden_weight, 
  float *tmp_inputs, 
  float *tmp_h, 
  float *output, 
  const int batch_size, 
  const int sequence_length, 
  const int input_size, 
  const int hidden_size);

std::vector<torch::Tensor> ben_rnn(
    torch::Tensor input,
    torch::Tensor input_weight,
    torch::Tensor hidden_weight)
    {
        CHECK_INPUT(input);
        CHECK_INPUT(input_weight);
        CHECK_INPUT(hidden_weight);
        auto input_ptr = input.data_ptr<float>();
        auto input_weight_ptr = input_weight.data_ptr<float>();
        auto hidden_weight_ptr = hidden_weight.data_ptr<float>();
        auto tmp_inputs = torch::zeros({input.size(0), input.size(1), hidden_weight.size(1)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto tmp_h = torch::zeros({input.size(0), input.size(1), hidden_weight.size(1)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto output = torch::zeros({input.size(0), input.size(1), hidden_weight.size(1)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        rnn_forward(
          input_ptr, 
          input_weight_ptr, 
          hidden_weight_ptr, 
          tmp_inputs.data_ptr<float>(), 
          tmp_h.data_ptr<float>(), 
          output.data_ptr<float>(), 
          input.size(1), 
          input.size(0), 
          input_weight.size(0), 
          hidden_weight.size(0));
        return std::vector<torch::Tensor>{output};
    }
    
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ben_rnn", &ben_rnn, "custom basic rnn");
}