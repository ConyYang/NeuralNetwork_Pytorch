import torch
from TensorCalculation.printCharacter import print_character


def power():
    t1 = torch.arange(1, 4)
    t2 = torch.arange(3)
    tp = torch.pow(t1, t2)
    torchs = [t1, t2, tp]
    for i, t in enumerate(torchs):
        print("t{}".format(i + 1))
        print_character(t)


def exponential():
    t1 = torch.tensor([0.1, -0.01])
    te = torch.exp(t1)
    torchs = [t1, te]
    for i, t in enumerate(torchs):
        print("t{}".format(i + 1))
        print_character(t)


def sin():
    t1 = torch.tensor([[3.14 / 4], ])
    ts = torch.sin(t1)
    torchs = [t1, ts]
    for i, t in enumerate(torchs):
        print("t{}".format(i + 1))
        print_character(t)


def fraction():
    t1 = torch.arange(5)
    tf = torch.frac(t1*0.3)
    tc = torch.clamp(t1, 0.5, 3.5)
    torchs = [t1, tf, tc]
    for i, t in enumerate(torchs):
        print("t{}".format(i + 1))
        print_character(t)

fraction()