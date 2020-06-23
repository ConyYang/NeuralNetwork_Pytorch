import torch
from TensorCalculation.printCharacter import print_character
import numpy as np

def rational_cal():
    t1 = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    t2 = torch.tensor([[7., 8., 9.], [10., 11., 12.]])

    t3 = t1 + t2
    t4 = t1 - t2
    t5 = t1 * t2
    t6 = t1 / t2
    t7 = t1 ** t2
    t8 = t1 ** (1 / t2)

    torchs = [t1, t2, t3, t4, t5, t6, t7, t8]
    for i, t in enumerate(torchs):
        print("t{}".format(i+1))
        print_character(t)


def broadcast():
    t1 = torch.zeros(3, 4)+5
    t2 = -6 * torch.ones(2)
    t3 = torch.ones(2, 3, 4) + torch.ones(4)
    t4 = torch.ones(2, 3, 4) + torch.ones(3, 4)
    torchs = [t1, t2, t3, t4]
    for i, t in enumerate(torchs):
        print("t{}".format(i+1))
        print_character(t)


def special():
    arr = [i * 1.0 for i in range(24)]
    t1 = torch.tensor(arr)
    t2 = t1.reciprocal()
    t3 = t1.sqrt()
    torchs = [t1, t2, t3]
    for i, t in enumerate(torchs):
        print("t{}".format(i+1))
        print_character(t)

special()