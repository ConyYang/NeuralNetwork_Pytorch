# hint:
# reshape, squeeze, unsqueeze will not change the position of element
# permute, transpose, t will change the position of element
import torch
from TensorCalculation.printCharacter import print_character

def funct1_reshape():
    t1 = torch.arange(12)
    t2 = t1.reshape(3, 2, 2)
    t3 = t2.reshape(4, 3)
    torchs = [t1, t2, t3]
    for i, t in enumerate(torchs):
        print("t{}".format(i))
        print_character(t)


def funct2_sqeeze():
    t1 = torch.arange(24)
    t2 = t1.reshape(2, -1, 4)
    t3 = torch.arange(24).reshape(2, 1, 3, 1, 4)
    t4 = t3.squeeze()
    t5 = t4.unsqueeze(dim=2)
    t6 = t5.unsqueeze(dim=1)
    torchs = [t1, t2, t3, t4, t5, t6]
    for i, t in enumerate(torchs):
        print("t{}".format(i + 1))
        print_character(t)


def funct3_permute():
    # permute
    t1 = torch.arange(8).reshape(2, 4)
    t2 = t1.permute(dims=[1, 0])
    t3 = torch.arange(24).reshape(2, 3, 4)
    t4 = t3.permute(dims=[1, 2, 0])
    torchs = [t1, t2, t3, t4]
    for i, t in enumerate(torchs):
        print("t{}".format(i + 1))
        print_character(t)


def funct4_transpose():
    t1 = torch.tensor([[5., -9.], ])
    t2 = t1.transpose(0,1)
    t3 = t1.t()
    torchs = [t1, t2, t3]
    for i, t in enumerate(torchs):
        print("t{}".format(i + 1))
        print_character(t)

