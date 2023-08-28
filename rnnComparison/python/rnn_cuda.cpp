#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> rnn_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor init_h);


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> rnn_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor init_h) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  CHECK_INPUT(init_h);

  return rnn_cuda_forward(input, weights, bias, init_h);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rnn_forward, "rnn forward (CUDA)");
  // m.def("backward", &rnn_backward, "rnn backward (CUDA)");
}

int main(){
  int batch = 2;
  int sequence_length = 3;
  int input_size = 5;
  int hidden_size = 5;
  auto options =torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 1)
    .requires_grad(true);


  torch::Tensor input = torch::empty({batch, sequence_length, input_size}, options);
  torch::Tensor weight = torch::eye(input_size, options);
  torch::Tensor bias = torch::zeros({1});
  torch::Tensor init_h = torch::zeros({1});
  torch::Tensor output = rnn_cuda_forward(input, weight, bias, init_h)[0];
  return 0;

}