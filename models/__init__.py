from .models import *
from .resnet3d import *
from .vgg3d import *
from .resnet_med3d import *
from .dense_net3d import *
from .sam_med3d import *

class Med3D(nn.Module):
    def __init__(self, med3d, inplane, expansion, n_classes = 1) -> None:
        super().__init__()
        self.module = med3d
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(inplane * expansion, n_classes)

    def forward(self, x):
        x = self.module(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = x.ravel()
        return x

def get_model(model_name, n_classes = 1):
    if model_name == 'resnet10':
        model = generate_model(10, [64, 128, 256, 512], n_input_channels = 1,conv1_t_size = 5, conv1_t_stride=1, n_classes = n_classes)
        # model = C3D(num_classes = 2)

    elif model_name == 'resnet50':
        # model = generate_model(50, [64, 128, 256, 512], n_input_channels = 1,conv1_t_size = 5, conv1_t_stride=1, n_classes = n_classes)
        model = resnet50(sample_input_D = 0, sample_input_H = 0, sample_input_W = 0, num_seg_classes = 1, shortcut_type = 'B')
        model = Med3D(model, 512, 4, n_classes = n_classes)

    elif model_name == 'vgg11':
        model = VGG3D(n_classes = n_classes, dropout=0.3)

    # elif model_name == 'resnet10_med3d':
    #     tmp = resnet10(sample_input_D = 0, sample_input_H = 0, sample_input_W = 0, num_seg_classes = 1, shortcut_type = 'B')
    #     model = Med3D(tmp, 512, 1, n_classes = n_classes)
    #     pretrained = torch.load('/home/image/users/Dream/Alzheimer3D/pretrained/resnet_10.pth')['state_dict']
    #     model.load_state_dict(pretrained, strict=False)

    elif model_name == 'resnet50_med3d':
        tmp = resnet50(sample_input_D = 0, sample_input_H = 0, sample_input_W = 0, num_seg_classes = 1, shortcut_type = 'B')
        pretrained = torch.load('/home/image/users/Dream/Alzheimer3D/pretrained/resnet_50.pth')['state_dict']
        model = Med3D(tmp, 512, 4, n_classes = n_classes)
        model.load_state_dict(pretrained, strict = False)

    elif model_name == 'densenet_201':
        model  = generate_model_densenet(201)
        model.classifier = nn.Linear(in_features=1920, out_features=n_classes, bias=True)
    elif model_name == 'sam_med3d_turbo':
        model = generate_model_sam_med3d_turbo(n_classes)

    elif model_name == 'sam_med3d_brain':
        model = generate_model_sam_med3d_brain(n_classes)

    else:
        raise ValueError("no model")
    return model