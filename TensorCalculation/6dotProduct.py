import torch
from TensorCalculation.printCharacter import print_character


def dotProduct():
    x = torch.arange(4)
    y = torch.arange(1, 5)
    xy = torch.dot(x, y)
    torchs = [x, y, xy]
    for i, t in enumerate(torchs):
        print("t{}".format(i + 1))
        print_character(t)


def matrix():
    x = torch.arange(4).view(2, 2)
    y = torch.arange(2)
    xy = torch.mv(x, y)
    torchs = [x, y, xy]
    for i, t in enumerate(torchs):
        print("t{}".format(i + 1))
        print_character(t)


def matrix2():
    x = torch.arange(6).view(2, 3)
    y = torch.arange(6).view(3, 2)
    xy = torch.mm(x, y)
    torchs = [x, y, xy]
    for i, t in enumerate(torchs):
        print("t{}".format(i + 1))
        print_character(t)


def einstein():
    loper = torch.arange(4).reshape(2, 2)
    roper = torch.arange(6).reshape(2, 3)
    result = torch.einsum('ij,jk>ik', (loper, roper))
    torchs = [loper, roper, result]
    for i, t in enumerate(torchs):
        print("t{}".format(i + 1))
        print_character(t)

einstein()
