import torch
from TensorCalculation.printCharacter import print_character


def compare():
    t1 = torch.tensor([-1, 1, 3], dtype=torch.float32)
    t2 = torch.arange(3, dtype=torch.float32)
    print_character(t1)
    print_character(t2)
    print('<:{}'.format(t1<t2))
    print('=:{}'.format(t1 == t2))
    print(torch.equal(t1, t2))
    # print(torch.nonzero(t2))
    print(torch.max(t1, t2))

def ifelsefunction():
    condition = torch.tensor([1, 0, 1], dtype = torch.uint8)
    x = torch.tensor([-0.2, 3.0, 5.6])
    y = torch.tensor([3.0, 2.3, -9.0])
    t3 = torch.where(condition, x, y)
    print_character(t3)
    # 1- x; 0 -y
ifelsefunction()