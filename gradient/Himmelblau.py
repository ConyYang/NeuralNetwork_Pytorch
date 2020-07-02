import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import torch

def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
X, Y = np.meshgrid(x, y)
z = himmelblau([X, Y])


def plot_himmelblau():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, z)
    ax.view_init(60, -30)
    ax.set_xlabel('x[0]')
    ax.set_ylabel('y[0]')
    fig.show()

# find the minimum value of Himmelblau
def find_minimum():
    x = torch.tensor([-1.0, 0.], requires_grad=True)
    optimizer = torch.optim.Adam([x, ])
    for step in range(20001):
        if step:
            optimizer.zero_grad()
            f.backward()
            optimizer.step()
        f = himmelblau(x)
        if step % 1000 == 0:
            print('step {}: x = {}, f( x) = {}'.format(step, x.tolist(), f))

def find_maximum():
    x = torch.tensor([-1.0, 0.], requires_grad=True)
    optimizer = torch.optim.Adam([x, ])
    for step in range(20001):
        if step:
            optimizer.zero_grad()
            (-f).backward()
            optimizer.step()
        f = himmelblau(x)
        if step % 1000 == 0:
            print('step {}: x = {}, f( x) = {}'.format(step, x.tolist(), f))


find_maximum()