import torch
from TensorCalculation.printCharacter import print_character


def funct1():
    t2 = torch.tensor([[0, 1, 2], [3, 4, 5]])
    t2 = t2.reshape(3, 2)
    t2 = t2 + 1
    print_character(t2)


def funct2():
    t0 = torch.tensor(0)
    t1 = torch.tensor([0., 1., 2.])
    t2 = torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])
    t3 = torch.tensor([[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
                       [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
                       [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]]])
    torchs = [t0, t1, t2, t3]
    for index, a in enumerate(torchs):
        print('t {}'.format(index))
        print_character(a)


def funct3():
    t_int8 = torch.tensor([1, 2], dtype=torch.int8)
    t_int16 = torch.tensor([1, 2], dtype=torch.int16)
    t_int32 = torch.tensor([1, 2], dtype=torch.int32)
    t_int64 = torch.tensor([1, 2], dtype=torch.int64)
    t_float16 = torch.tensor([1, 2], dtype=torch.float16)
    t_float32 = torch.tensor([1, 2], dtype=torch.float32)
    t_float64 = torch.tensor([1, 2], dtype=torch.float64)

    torchs = [
        t_int8,
        t_int16,
        t_int32,
        t_int64,
        t_float16,
        t_float32,
        t_float64
    ]
    for index, a in enumerate(torchs):
        print('t {}'.format(index))
        print_character(a)


def funct4():
    t1 = torch.empty(2)
    t2 = torch.zeros(2, 2)
    t3 = torch.ones(2, 2, 2)
    t4 = torch.full((2, 2, 2, 2), 3)
    t5 = torch.ones_like(t2)

    torchs = [t1, t2, t3, t4, t5]
    for index, a in enumerate(torchs):
        print('t {}'.format(index + 1))
        print_character(a)


def create_array():
    t1 = torch.arange(0, 4, step=1)
    t2 = torch.range(0, 3, step=1)
    t3 = torch.linspace(0, 3, steps=4)
    t4 = torch.logspace(0, 3, steps=4)

    torchs = [t1, t2, t3, t4]
    for index, a in enumerate(torchs):
        print('t {}'.format(index + 1))
        print_character(a)


def random_tensor():
    probs = torch.full((3, 4), 0.9)  # higher more ones
    t1 = torch.bernoulli(probs)

    weights = torch.tensor([[1, 100], [100, 1], [1, 1]], dtype=torch.float32)
    t2 = torch.multinomial(weights, 1)

    t3 = torch.randperm(8, dtype=torch.float32)

    t4 = torch.randint(low=0, high=5, size=(2, 5))

    t5 = torch.randint_like(torch.ones(3, 4), low=0, high=4)

    torchs = [t1, t2, t3, t4, t5]
    for index, a in enumerate(torchs):
        print('t {}'.format(index + 1))
        print_character(a)


def rand_sample():
    t1 = torch.rand(2, 3)
    t2 = torch.rand_like(torch.ones(2, 3))

    t3 = torch.randn(2, 3)
    t4 = torch.randn_like(torch.ones(2, 3))

    torchs = [t1, t2, t3, t4]
    for index, a in enumerate(torchs):
        print('t {}'.format(index + 1))
        print_character(a)


def create_normal():
    mean = torch.tensor([0., 1.])
    std = torch.tensor([3., 2.])
    normal = torch.normal(mean, std)
    torchs = [mean, std, normal]
    for index, a in enumerate(torchs):
        print('t {}'.format(index + 1))
        print_character(a)

