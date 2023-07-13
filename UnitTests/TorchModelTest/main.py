import torch

class Perc(torch.nn.Module):
    def __init__(self, in_size=16, out_size=64) -> None:
        super().__init__()
        self.in_size = in_size
        self.lin = torch.nn.Linear(16, 64)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        # assert x.ndimension() == 1
        # assert x.size(0) == self.in_size
        return self.lin(self.relu(x))



if __name__ == '__main__':
    model = Perc()
    input = torch.arange(0, 16, dtype=torch.float32, device='cpu')[None, ...]
    
    traced_policy = torch.jit.trace(model, input)
    traced_policy.save('test.jit')