import torch
import torch.nn
import torch.optim



def test1():
    criterion = torch.nn.MSELoss()
    pred = torch.arange(5., requires_grad=True)
    y = torch.ones(5)
    loss = criterion(pred, y)
    print(loss)
    loss.backward()
    print(loss)

def test2():
    x = torch.tensor([
        [1., 1., 1.],
        [2., 4., 1.],
        [3., 5., 1.],
        [4., 2., 1.],
        [2., 4., 1.]
    ])
    y = torch.tensor([-10, 12., 14., 16., 18.])
    w = torch.zeros(3, requires_grad=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([w, ], )
    for step in range(30001):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred = torch.mv(x, w)
        loss = criterion(pred, y)
        if step % 1000 == 0:
            print(' step = {}, loss = {:g}, W = {}'.format(step, loss, w.tolist()))

def test3():
    x = torch.tensor([
        [1., 1.],
        [2., 3.],
        [3., 5.],
        [4., 2.],
        [5., 4.]
    ])
    y = torch.tensor([-10, 12., 14., 16., 18.]).reshape(-1, 1)
    fc = torch.nn.Linear(in_features=2, out_features=1, bias=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(fc.parameters())
    weights, bias = fc.parameters()

    for step in range(30001):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred = fc(x)
        loss = criterion(pred, y)

        if step % 1000 == 0:
            print(' step = {}, loss = {:g}, weights = {}, bias = {}'.format
                  (step, loss, weights.tolist(), bias.item()))

test3()