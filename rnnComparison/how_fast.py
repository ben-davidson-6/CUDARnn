import time

import torch

assert torch.cuda.is_available()


class Pure(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[0]


def how_fast(module_class) -> None:
    batch_size = 8
    sequence_length = 2048
    input_features = 43
    state_size = 256
    module = module_class(input_features, state_size).to(cuda_device)

    X = torch.randn(sequence_length, batch_size, input_features, device=cuda_device)

    # force initialisation of cuda
    new_h = module.forward(X)
    new_h.sum().backward()

    forward = 0.0
    backward = 0.0
    num = 100
    for _ in range(num):
        start = time.time()
        new_h = module.forward(X)
        torch.cuda.synchronize()
        forward += time.time() - start
        s = new_h.sum()
        torch.cuda.synchronize()

        start = time.time()
        s.backward()
        torch.cuda.synchronize()
        backward += time.time() - start

    print(
        "Forward: {:.7f} us | Backward {:.7f} us".format(
            forward * 1e4 / num, backward * 1e4 / num
        )
    )


def time_pure():
    how_fast(Pure)


# def time_cuda():
#     from cudaVersion.module import LLTM

#     how_fast(LLTM)


if __name__ == "__main__":
    cuda_device = torch.device("cuda")
    time_pure()
    # time_cpp()
    # time_cuda()
