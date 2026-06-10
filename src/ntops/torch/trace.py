import torch


def trace(input):
    diagonal = torch.diagonal(input)
    return diagonal.sum()
