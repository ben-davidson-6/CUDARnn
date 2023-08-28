import math

import torch
from torch import nn
from torch.autograd import Function

import rnn_cuda  # isort: skip


# class RNNCudaFunction(Function):
#     @staticmethod
#     def forward(ctx, input, weights, bias, init_h):
#         hs = rnn_cuda.forward(input, weights, bias, init_h)
#         ctx.save_for_backward(hs)
#         return hs

#     @staticmethod
#     def backward(ctx, grad_h):
#         outputs = rnn_cuda.backward(grad_h, *ctx.saved_variables)
#         d_input, d_weights, d_bias, d_init_h = outputs
#         return d_input, d_weights, d_bias, d_init_h


class RNNCuda(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.weights = nn.Parameter(
            torch.eye(input_size, hidden_size, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.zeros(1, hidden_size, dtype=torch.float32))

    def forward(self, input, state):
        return rnn_cuda.forward(input, self.weights, self.bias, state)[0]


if __name__ == "__main__":
    inputs = torch.rand((2, 4, 3), dtype=torch.float32).to("cuda")
    state = torch.empty((1), dtype=torch.float32, device="cuda")
    rnn = RNNCuda(3, 3)
    rnn.to("cuda")
    outputs = rnn.forward(inputs, state)
    assert torch.allclose(inputs, outputs, rtol=0.001)
