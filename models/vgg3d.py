# https://www.frontiersin.org/articles/10.3389/fbioe.2020.534592/full

import torch
import torch.nn as nn





class VGG3D(nn.Module):
    def __init__(
        self, n_classes: int = 3, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.cnn1 = nn.Conv3d(1, 32, 3, padding='same')
        self.bn1 = nn.BatchNorm3d(32)

        self.cnn2 = nn.Conv3d(32, 64, 3, padding='same')
        self.bn2 = nn.BatchNorm3d(64)

        self.cnn3 = nn.Conv3d(64, 128, 3, padding='same')
        self.bn3 = nn.BatchNorm3d(128)
        self.cnn4 = nn.Conv3d(128, 128, 3, padding='same')
        self.bn4 = nn.BatchNorm3d(128)

        self.cnn5 = nn.Conv3d(128, 256, 3, padding='same')
        self.bn5 = nn.BatchNorm3d(256)
        self.cnn6 = nn.Conv3d(256, 256, 3, padding='same')
        self.bn6 = nn.BatchNorm3d(256)

        self.cnn7 = nn.Conv3d(256, 256, 3, padding='same')
        self.bn7 = nn.BatchNorm3d(256)
        self.cnn8 = nn.Conv3d(256, 256, 3, padding='same')
        self.bn8 = nn.BatchNorm3d(256)
        self.maxpool = nn.MaxPool3d(2)

        self.relu = nn.ReLU(inplace = True)
        self.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(512, n_classes),
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.cnn1(x))
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.relu(self.cnn2(x))
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.relu(self.cnn3(x))
        x = self.bn3(x)
        x = self.relu(self.cnn4(x))
        x = self.bn4(x)
        x = self.maxpool(x)

        x = self.relu(self.cnn5(x))
        x = self.bn5(x)
        x = self.relu(self.cnn6(x))
        x = self.bn6(x)
        x = self.maxpool(x)

        x = self.relu(self.cnn7(x))
        x = self.bn7(x)
        x = self.relu(self.cnn8(x))
        x = self.bn8(x)
        x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        if self.n_classes <2:
            x = x.ravel()

        return x

if __name__ == "__main__":
    model = VGG3D(num_classes=3, dropout=0.5)
    x = torch.randn((3,1,64,64,64))

    print(model)
    model(x)