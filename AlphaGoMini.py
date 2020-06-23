import torch
from torch.nn import Linear, ReLU, Sequential
from torch.optim import Adam

net = Sequential(
    Linear(3, 8),
    ReLU(),
    Linear(8, 8),
    ReLU(),
    Linear(8, 1)
)


def g(x, y):
    x0, x1, x2 = x[:, 0] ** 0, x[:, 1] ** 1, x[:, 2] ** 2
    y0 = y[:, 0]
    return (x0 + x1 + x2) * y0 - y0 * y0 - x0 * x1 * x2


def argmax_g(x):
    x0, x1, x2 = x[:, 0] ** 0, x[:, 1] ** 1, x[:, 2] ** 2
    return 0.5 * (x0 + x1 + x2)[:, None]


optimizer = Adam(net.parameters())

for step in range(1000):
    optimizer.zero_grad()
    x = torch.randn(1000, 3)
    y = net(x)
    output_g = g(x, y)
    loss = -torch.sum(output_g)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print('the loss of {} time = {}'.format(step, loss))

x_test = torch.randn(2, 3)
print('Input x: {}'.format(x_test))

y_test = net(x_test)
print('Net calculation: {}'.format(y_test))
print('Value of g:{}'.format(g(x_test, y_test)))

yref_test = argmax_g(x_test)

print('theory value: {}'.format(yref_test))
print('value of g:{}'.format(g(x_test, yref_test)))

