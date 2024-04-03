import itertools
import random
from time import perf_counter

import numpy as np
import torch

import ben_rnn  # isort:skip must be after torch


def compare_rnn(
    input_size: int = 2,
    hidden_size: int = 3,
    batch_size: int = 4,
    sequence_len: int = 13,
    N: int = 100,
) -> None:
    rnn = torch.nn.RNN(input_size, hidden_size, bias=False)
    rnn.to("cuda")
    rnn_input = torch.randn(
        (sequence_len, batch_size, input_size), device="cuda", dtype=torch.float32
    )
    total_time = 0.0
    for _ in range(N):
        start = perf_counter()
        torch_out = rnn.forward(rnn_input)[0]
        torch_out.record_stream(torch.cuda.current_stream())
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


def test_with_many_values() -> None:
    input_sizes = [1] + [random.randint(2, 100) for _ in range(10)]
    hidden_sizes = [1] + [random.randint(2, 100) for _ in range(10)]
    batch_sizes = [1] + [random.randint(2, 100) for _ in range(10)]
    sequence_lens = [1] + [random.randint(2, 100) for _ in range(10)]

    # Generate all combinations of parameters
    all_combinations = list(
        itertools.product(input_sizes, hidden_sizes, batch_sizes, sequence_lens)
    )

    for combination in all_combinations:
        input_size, hidden_size, batch_size, sequence_len = combination
        try:
            compare_rnn(input_size, hidden_size, batch_size, sequence_len, N=1)
        except Exception as e:
            print("Failed with combination:", combination)


if __name__ == "__main__":
    compare_rnn(64, 128, 32, 128)
