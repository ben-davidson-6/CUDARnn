from time import perf_counter

import torch
from torch import nn
from torch.autograd import Function

import ben_rnn  # isort:skip must be after torch


def compare_rnn():
    input_size = 2
    hidden_size = 2
    batch_size = 1
    sequence_len = 2
    N = 100
    rnn = torch.nn.RNN(input_size, hidden_size, bias=False)
    rnn.to("cuda")
    rnn_input = torch.rand(
        (sequence_len, batch_size, input_size), dtype=torch.float32, device="cuda"
    )
    total_time = 0.0
    for _ in range(N):
        start = perf_counter()
        torch_out = rnn.forward(rnn_input)[0]
        torch.cuda.synchronize()
        total_time += perf_counter() - start
    weights = rnn.all_weights[0]
    input_weight = weights[0].T.contiguous()
    hidden_weight = weights[1].T.contiguous()
    total_time_mine = 0.0
    for _ in range(N):
        start = perf_counter()
        our_out = ben_rnn.ben_rnn(
            rnn_input, input_weight, hidden_weight
        )
        torch.cuda.synchronize()
        total_time_mine += perf_counter() - start

    assert torch.allclose(torch_out, our_out), "Outputs don't match"
    print(f"torch rnn time: {total_time/N}")
    print(f"my rnn time: {total_time_mine/N}")


if __name__ == "__main__":
    compare_rnn()
