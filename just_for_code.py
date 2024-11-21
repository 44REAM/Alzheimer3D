from sklearn import metrics
import torch

x = torch.randn(2,3)
y = torch.randn(4,3)
print(torch.cat([x,y]))