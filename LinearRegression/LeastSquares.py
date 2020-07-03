import torch
x = torch.tensor([
    [1., 1., 1.],
    [2., 4., 1.],
    [3., 5., 1.],
    [4., 2., 1.],
    [2., 4., 1.]
])

y = torch.tensor([-10, 12., 14., 16., 18.])
wr,_ = torch.lstsq(y, x)
print(wr)