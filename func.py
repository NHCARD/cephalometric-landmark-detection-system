import torch

def argsoftmax(x, index, beta=1e-2):
    a = torch.exp(-torch.abs(x - x.max(dim=1).values.unsqueeze(1)) / (beta))
    b = torch.sum(a, dim=1).unsqueeze(1)
    softmax = a / b
    return torch.mm(softmax, index)