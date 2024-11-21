import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn import metrics
from copy import deepcopy
import os


from models import get_model
from dataset import *
from utils import *


def cal_metrics_multiclass(test_labels_all, test_outputs_all):

    results_class = test_outputs_all.argmax(axis=-1)

    bacc = metrics.balanced_accuracy_score(test_labels_all, results_class)
    acc = metrics.accuracy_score(test_labels_all, results_class)

    # specificity = metrics.recall_score(test_labels_all, results_class)

    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        test_labels_all, results_class, beta=1, average='macro')

    rocauc = metrics.roc_auc_score(
        test_labels_all, test_outputs_all, average='macro', multi_class='ovo')
    results = {
        'acc': acc,
        'bacc': bacc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rocauc': rocauc,
        # 'specificity': specificity
    }
    return results


def run_test(test_loader, model, device):
    model.eval()
    test_outputs_all = []
    test_labels_all = []
    with torch.no_grad():
        for test_images, test_labels in tqdm(test_loader):
            test_images, test_labels = test_images.to(
                device), test_labels.to(device)
            test_outputs = model(test_images)

            test_outputs = F.softmax(test_outputs, dim=-1).cpu()
            test_outputs_all.append(test_outputs)
            test_labels_all.append(test_labels.cpu())

    test_outputs_all = torch.cat(test_outputs_all, dim=0)
    test_labels_all = torch.cat(test_labels_all, dim=0)

    print(test_outputs_all)
    results = cal_metrics_multiclass(test_labels_all, test_outputs_all)
    print(results)
    return results

if __name__ == '__main__':
    torch.cuda.empty_cache()
    all_model = [
        # 'vgg11',
        # 'resnet10',
        # 'resnet50',
        # 'resnet50_med3d',
        'densenet_201'
    ]
    modalities = [
        # 'DTI/wdtifit_FA.nii',
        # 'DTI/wdtifit_MD.nii',
        # 'DTI/wfdt_paths.nii',
        # 'DTI/wnum_fcd.nii',
        'DTI/wnodif.nii',
        # 't2s/wrR2S.nii',
        # 'working_memory/wrcon_0001.nii'
    ]
    should_normalize = [
        # False,
        # False,
        # True,
        # True,
        True,
        # True,
        # True
    ]

    checkpoints_dir = './checkpoints'
    checkpoints_suffix = "ad_first_visit"
    dataset_suffix = "ad_first_visit"
    version = 0

    img_size = 64
    max_epochs = 100
    batch_size = 16
    num_workers = 8
    random_seed = 42
    lr = 1e-4
    betas = (0.9, 0.995)
    torch.backends.cudnn.benchmark = True
    pin_memory = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    X_train_base = np.load(f'indices/train_filenames_{dataset_suffix}.npy', allow_pickle=True)
    X_val_base = np.load(f'indices/val_filenames_{dataset_suffix}.npy', allow_pickle=True)
    X_test_base = np.load(f'indices/test_filenames_{dataset_suffix}.npy', allow_pickle=True)

    y_train_base = np.load(f'indices/train_labels_{dataset_suffix}.npy', allow_pickle=True)
    y_val_base = np.load(f'indices/val_labels_{dataset_suffix}.npy', allow_pickle=True)
    y_test_base = np.load(f'indices/test_labels_{dataset_suffix}.npy', allow_pickle=True)

    df = pd.DataFrame()
    for mo, norm in zip(modalities, should_normalize):
        print(f"Start {mo}")
        ###########################################################################################################
        for model_name in all_model:

            X_train = [os.path.join(name, mo) for name in X_train]
            X_val = [os.path.join(name, mo) for name in X_val]
            X_test = [os.path.join(name, mo) for name in X_test]
            y_train = deepcopy(y_train_base)
            y_val = deepcopy(y_val_base)
            y_test = deepcopy(y_test_base)


            checkpoint_name = f"{mo.split('/')[-1]}-{model_name}-{version}{checkpoints_suffix}"
            tensorboard_suffix = f"{model_name}-{version}"
            print(f"RUN: {checkpoint_name}")

            try:

                mean_data, std_data = 0, 1
                val_aug = get_transform(
                    mean_data, std_data, mode='val', self_normalized=norm)

                test_loader = get_loader(X_test, y_test, val_aug, mode='val', batch_size=batch_size,
                                            pin_memory=pin_memory, num_workers=num_workers, img_size=img_size)

                model = get_model(model_name, n_classes=3)
                model = model.to(device)
                model.load_state_dict(load_checkpoint(
                    checkpoints_dir, checkpoint_name, 'bacc'))

                weight = torch.tensor([1, len(y_train[y_train == 0])/len(
                    y_train[y_train == 1]), len(y_train[y_train == 0])/len(y_train[y_train == 2])])
                # weight = weight.double()
                loss_function = torch.nn.CrossEntropyLoss(
                    weight=weight).to(device)
                results = run_test(test_loader, model, device=device)

                df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
                df.to_csv(f'./results/alzheimer_{checkpoints_suffix}.csv', index=False)
            except Exception as e:
                print(e)
