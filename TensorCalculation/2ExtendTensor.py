# hint:
# repeat, cat

import torch
from TensorCalculation.printCharacter import print_character


def funct1_repeat():
    t1 = torch.tensor([[5., -9.], ])
    t2 = t1.repeat(3, 2)
    # get (3*3*2) elements
    torchs = [t1, t2]
    for i, t in enumerate(torchs):
        print('t{}'.format(i))
        print_character(t)


def funct2_cat():
    t1 = torch.arange(12).reshape(3, 4)
    t2 = -1 * t1
    t3 = torch.cat([t1, t2], dim=0)
    t4 = torch.cat([t1, t2], dim=1)
    t5 = torch.cat([t1, t2, t1, t2], dim=0)
    t6 = torch.cat([t1, t1, t2, t2], dim=1)
    torchs = [t1, t2, t3, t4, t5, t6]
    for i, t in enumerate(torchs):
        print('t{}'.format(i+1))
        print_character(t)

def funct3_stack():
    t1 = torch.arange(12).reshape(3, 4)
    t2 = -1 * t1
    t3 = torch.stack([t1, t2], 0)
    t4 = torch.stack([t1, t2], 1)
    t5 = torch.stack([t1, t2, t1, t2], dim=0)
    t6 = torch.stack([t1, t1, t2, t2], dim=1)
    torchs = [t1, t2, t3, t4, t5, t6]
    for i, t in enumerate(torchs):
        print('t{}'.format(i+1))
        print_character(t)

funct2_cat()
funct3_stack()