import torch
num_sample=10000000
sample=torch.rand(num_sample,2)
dist=sample.norm(p=2,dim=1)
ratio=(dist<1).float().mean()
pi=ratio*4
print('pi={}'.format(pi))

