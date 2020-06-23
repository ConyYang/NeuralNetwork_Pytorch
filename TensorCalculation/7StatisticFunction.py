import torch
from TensorCalculation.printCharacter import print_character


def funct1():
    t = torch.arange(10, dtype=torch.float32)
    print_character(t)
    print('variation = {}'.format(t.var()))
    print('production = {}'.format(t.prod()))
    print('max = {}'.format(t.max()))
    print('min = {}'.format(t.min()))
    print('mean = {}'.format(t.mean()))
    print('median = {}'.format(t.median()))
    print('2nd largest = {}'.format(t.kthvalue(2)))


def funct2():
    t = torch.arange(24, dtype=torch.float32).reshape(2,3,4)
    print_character(t)
    print('variation = {}'.format(t.var(dim=2)))
    print('production = {}'.format(t.prod(dim=2)))
    print('max = {}'.format(t.max(dim=2)))
    print('min = {}'.format(t.min(dim=2)))
    print('mean = {}'.format(t.mean(dim=2)))
    print('median = {}'.format(t.median(dim=2)))
    print('2nd largest = {}'.format(t.kthvalue(2, dim=2)))

funct2()