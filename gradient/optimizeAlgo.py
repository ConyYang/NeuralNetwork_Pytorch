from math import pi
import torch
import torch.optim
x = torch.tensor([pi/3, pi/6], requires_grad=True)
optimizer = torch.optim.SGD([x, ], lr=0.1, momentum=0)
for step in range(11):
    if step:
        optimizer.zero_grad()
        f.backward()
        optimizer.step()
    f = -((x.cos() ** 2).sum()) ** 2
    print('step{}: x={} f(x)={}'.format(step, x.tolist(), f))
