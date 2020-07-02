from math import pi
import torch
x = torch.tensor([pi/3, pi/6], requires_grad=True)
f = -((x.cos() ** 2).sum()) ** 2
print('value = {}'.format(f))
f.backward()
print('grad = {}'.format(x.grad))
