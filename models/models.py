import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial


class C3D(nn.Module):

    """
    This is the c3d implementation with batch norm.
    [1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
    Proceedings of the IEEE international conference on computer vision. 2015.
    """

    def __init__(self, num_classes=3, in_channels=1):

        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0))

        self.fc1 = nn.Sequential(
            nn.Linear(1536 , 512),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(512, num_classes))   
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc(out)
        # out = self.softmax(out)

        return out
    
if __name__ == '__main__':
    model = C3D(num_classes = 2)
    img_size = 64 
    # 1536 for 64
    data = torch.randn((4, 1, img_size , img_size,img_size))
    model(data)