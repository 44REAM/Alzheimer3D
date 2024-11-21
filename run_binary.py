import torch
import numpy as np
from copy import deepcopy
import os

from tqdm import tqdm

from torch import nn
from sklearn import metrics

from models import get_model
from dataset import *
from utils import *


def cal_metrics_binary(test_labels_all, test_outputs_all):

    results_binary = deepcopy(test_outputs_all)
    results_binary[results_binary >= 0.5] = 1
    results_binary[results_binary < 0.5] = 0

    bacc = metrics.balanced_accuracy_score(test_labels_all, results_binary)
    acc = metrics.accuracy_score(test_labels_all, results_binary)
    specificity = metrics.recall_score(
        test_labels_all, results_binary, pos_label=0)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        test_labels_all, results_binary, beta=1)

    rocauc = metrics.roc_auc_score(test_labels_all, test_outputs_all)
    results = {
        'acc': acc,
        'bacc': bacc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rocauc': rocauc,
        'specificity': specificity
    }
    return results


def train_model(
        model, train_loader, val_loader,
        max_epochs, optimizer,
        loss_function, writer,
        checkpoints_dir, checkpoint_name):
    best_bacc = 0
    best_acc = 0

    for epoch in range(max_epochs):
        model.train()
        losses = []
        train_outputs_all = []
        train_labels_all = []

        for inputs, labels in tqdm(train_loader):

            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            outputs = outputs.squeeze(-1)

            # calculate loss from normalized labels and outputs
            loss = loss_function(outputs, labels.float())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            train_outputs_all.extend(outputs.sigmoid().detach().cpu().tolist())
            train_labels_all.extend(labels.cpu().tolist())

        train_outputs_all = np.array(train_outputs_all)
        train_labels_all = np.array(train_labels_all)

        train_outputs_all[train_outputs_all >= 0.5] = 1
        train_outputs_all[train_outputs_all < 0.5] = 0

        # Calculate RMSE and MAE for the entire validation dataset
        bacc_train = metrics.balanced_accuracy_score(
            train_labels_all, train_outputs_all)
        acc_train = metrics.accuracy_score(train_labels_all, train_outputs_all)

        train_loss = np.mean(losses)
        model.eval()
        val_outputs_all = []
        val_labels_all = []
        losses = []

        with torch.no_grad():
            for val_images, val_labels in tqdm(val_loader):
                val_images, val_labels = val_images.to(
                    device), val_labels.to(device)
                val_outputs = model(val_images)

                # loss = loss_function(val_outputs, val_labels.float())
                # losses.append(loss.item())
                val_outputs = val_outputs.sigmoid().cpu()
                val_labels = val_labels.cpu()

                val_outputs_all.append(val_outputs)
                val_labels_all.append(val_labels)

        val_outputs_all = torch.cat(val_outputs_all)
        val_labels_all = torch.cat(val_labels_all)

        val_outputs_all[val_outputs_all >= 0.5] = 1
        val_outputs_all[val_outputs_all < 0.5] = 0

        # Calculate RMSE and MAE for the entire validation dataset
        bacc = metrics.balanced_accuracy_score(val_labels_all, val_outputs_all)
        acc = metrics.accuracy_score(val_labels_all, val_outputs_all)

        if bacc >= best_bacc:
            best_bacc = bacc
            print('Save')
            save_checkpoint(checkpoints_dir, model, checkpoint_name, 'bacc')
        if acc >= best_acc:
            best_acc = acc
            save_checkpoint(checkpoints_dir, model, checkpoint_name, 'acc')

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('BACC/val', bacc, epoch)

        print(f'''Epoch {epoch + 1}, Validation BACC: {bacc:.4f}, Validation ACC: {acc:.4f}, Train Loss: {train_loss}\n
        Train BACC: {bacc_train:.4f}, Train ACC: {acc_train:.4f},
        ''')


def change_label(X_train, X_val, X_test, y_train, y_val, y_test):
    X_train = [name for name, label in zip(X_train, y_train) if label != 1]
    X_val = [name for name, label in zip(X_val, y_val) if label != 1]
    X_test = [name for name, label in zip(X_test, y_test) if label != 1]

    y_train = np.array(
        [label for name, label in zip(X_train, y_train) if label != 1])
    y_val = np.array(
        [label for name, label in zip(X_val, y_val) if label != 1])
    y_test = np.array(
        [label for name, label in zip(X_test, y_test) if label != 1])

    y_train[y_train > 1] = 1
    y_val[y_val > 1] = 1
    y_test[y_test > 1] = 1
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    torch.cuda.empty_cache()
    all_model = [
        'sam_med3d_turbo',
        # 'vgg11',
        # 'resnet10',
        # 'resnet50',
        # 'resnet50_med3d',
        'densenet_201'
    ]
    modalities = [
        'DTI/wdtifit_FA.nii',
        'DTI/wdtifit_MD.nii',
        'DTI/wfdt_paths.nii',
        'DTI/wnfdt_paths.nii',
        'DTI/wnodif.nii',
        'T2s/wR2S.nii',
    ]
    should_normalize = [
        False,
        False,
        True,
        True,
        True,
        True
    ]
    # modalities = [
    #     'DTI/wdtifit_FA.nii',
    #     'DTI/wdtifit_MD.nii',
    #     'DTI/wfdt_paths.nii',
    #     'DTI/wnum_fcd.nii',
    #     'DTI/wnodif.nii',
    #     't2s/wrR2S.nii',
    #     # 'working_memory/wrcon_0001.nii'
    # ]

    # # should_normalize depend on modality, some modality not need normalized
    # should_normalize = [
    #     False,
    #     False,
    #     True,
    #     True,
    #     True,
    #     True,
    #     # True
    # ]

    checkpoints_dir = './checkpoints'
    tensorboard_dir = './runs'
    checkpoints_suffix = "tbm"
    dataset_suffix = "tbm"
    load_checkpoints_suffix = 'tbm'
    train_from_checkpoint = False
    change_label_or_not = False
    version = 0

    img_size = 128
    max_epochs = 100
    batch_size = 16
    num_workers = 8
    random_seed = 42
    lr = 1e-4
    betas = (0.9, 0.995)

    pin_memory = True
    cuda_deterministic = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    if cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        # Faster, less reproducible
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    X_train_base = np.load(
        f'indices/train_filenames_{dataset_suffix}.npy', allow_pickle=True)
    X_val_base = np.load(
        f'indices/val_filenames_{dataset_suffix}.npy', allow_pickle=True)
    X_test_base = np.load(
        f'indices/test_filenames_{dataset_suffix}.npy', allow_pickle=True)

    y_train_base = np.load(
        f'indices/train_labels_{dataset_suffix}.npy', allow_pickle=True)
    y_val_base = np.load(
        f'indices/val_labels_{dataset_suffix}.npy', allow_pickle=True)
    y_test_base = np.load(
        f'indices/test_labels_{dataset_suffix}.npy', allow_pickle=True)

    for mo, norm in zip(modalities, should_normalize):
        print(f"Start {mo}")

        ###########################################################################################################
        for model_name in all_model:

            X_train = [os.path.join(name, mo) for name in X_train_base]
            X_val = [os.path.join(name, mo) for name in X_val_base]
            X_test = [os.path.join(name, mo) for name in X_test_base]
            y_train = deepcopy(y_train_base)
            y_val = deepcopy(y_val_base)
            y_test = deepcopy(y_test_base)

            if change_label_or_not:
                X_train, X_val, X_test, y_train, y_val, y_test = change_label(
                    X_train, X_val, X_test, y_train, y_val, y_test)

            try:

                checkpoint_name = f"{mo.split('/')[-1]}-{model_name}-{version}{checkpoints_suffix}"
                load_checkpoints_name = f"{mo.split('/')[-1]}-{model_name}-{version}{load_checkpoints_suffix}"
                tensorboard_suffix = f"{model_name}-{version}"
                print(f"RUN: {checkpoint_name}")

                mean_data, std_data = 0, 1
                train_aug = get_transform(
                    mean_data, std_data, self_normalized=norm)
                val_aug = get_transform(
                    mean_data, std_data, mode='val', self_normalized=norm)

                train_loader = get_loader(X_train, y_train, train_aug, mode='train', batch_size=batch_size,
                                          pin_memory=pin_memory, num_workers=num_workers, img_size=img_size)
                val_loader = get_loader(X_val, y_val, val_aug, mode='val', batch_size=batch_size,
                                        pin_memory=pin_memory, num_workers=num_workers, img_size=img_size)
                test_loader = get_loader(X_test, y_test, val_aug, mode='val', batch_size=batch_size,
                                         pin_memory=pin_memory, num_workers=num_workers, img_size=img_size)

                model = get_model(model_name)
                if train_from_checkpoint:
                    model.load_state_dict(load_checkpoint(
                        checkpoints_dir, load_checkpoints_name, 'bacc'))
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                model = model.to(device)

                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, betas=betas)
                pos_weight = torch.tensor(
                    (len(y_train) - np.sum(y_train))/np.sum(y_train))
                loss_function = torch.nn.BCEWithLogitsLoss(
                    pos_weight=pos_weight).to(device)

                writer = get_tensorboard_writer(tensorboard_dir)

                train_model(
                    model, train_loader, val_loader,
                    max_epochs, optimizer,
                    loss_function, writer,
                    checkpoints_dir, checkpoint_name
                )

                writer.close()
            except Exception as e:
                print(e)

# run 65696
