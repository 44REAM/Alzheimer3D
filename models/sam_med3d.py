import torch
import torch.nn as nn
from torchinfo import summary
from .segment_anything.build_sam3D import sam_model_registry3D


class SAMEncoder3D(nn.Module):
    def __init__(self, encoder, n_classes) -> None:
        super(SAMEncoder3D, self).__init__()
        self.encoder = encoder
        self.pooling = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(384, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pooling(x).view(-1, 384)
        x = self.classifier(x)

        return x


def generate_model_sam_med3d_turbo(n_classes):

    model = sam_model_registry3D['vit_b_ori'](checkpoint=None)
    state_dict = torch.load(
        '/home/image/users/Dream/Alzheimer3D/pretrained/sam_med3d_turbo.pth')['model_state_dict']
    model.load_state_dict(state_dict=state_dict)
    model = SAMEncoder3D(model.image_encoder, n_classes)
    return model


def generate_model_sam_med3d_brain(n_classes):

    model = sam_model_registry3D['vit_b_ori'](checkpoint=None)
    state_dict = torch.load(
        '/home/image/users/Dream/Alzheimer3D/pretrained/sam_med3d_brain.pth')['model_state_dict']
    model.load_state_dict(state_dict=state_dict)
    model = SAMEncoder3D(model.image_encoder, n_classes)
    return model


if __name__ == '__main__':
    model = generate_model_sam_med3d_turbo(5).cuda()
    x = torch.randn(5, 1, 128, 128, 128).to('cuda')
    summary(model)
    print(model(x).shape)
