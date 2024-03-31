from time import perf_counter

import numpy as np
import torch

import ben_rnn  # isort:skip must be after torch


def compare_rnn():
    input_size = 512
    hidden_size = 128
    batch_size = 1  # if goes to 2 get big diffs
    sequence_len = 3  # if goes to 4 get nans
    N = 100
    rnn = torch.nn.RNN(input_size, hidden_size, bias=False)
    rnn.to("cuda")
    rnn_input = torch.randn(
        (sequence_len, batch_size, input_size), device="cuda", dtype=torch.float32
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
        our_out = ben_rnn.ben_rnn(rnn_input, input_weight, hidden_weight)[0]
        torch.cuda.synchronize()
        total_time_mine += perf_counter() - start

    np.testing.assert_allclose(
        torch_out.cpu().detach().numpy(),
        our_out.cpu().detach().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    print(f"torch rnn time: {total_time/N}")
    print(f"my rnn time: {total_time_mine/N}")


if __name__ == "__main__":
    compare_rnn()
