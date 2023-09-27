import os
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import monai.transforms as monai_transforms

from dataset import (
    MRIDataset, 
    get_loader, get_data, get_covid_data, get_kidlead_data,
    get_normalization_param, 
    get_transform, get_normalization_param_nomask)
from models import C3D, generate_model, ResNet, get_model
from train import epoch_iter, add_metrics, save_checkpoint, get_metrics

from config import (
    kidlead_modalities,
    covid_test_modalities,
    tbi_modalities
)

no_data = []

# ! เปลี่ยน logit ใน model ไปเพราะทำ regression ยังไม่ได้แก้รันแล้ว error แน่ๆ

# ลืมแล้วว่าเขียนไว้ทำไม
# form = 'process'
# น่าจะเพราะ path แบบนี้แยก folder (1 คน 1 folder)
# # use for raw file
# modality = "T2s"
# use_file = "R2S.nii"
# น่าจะเพราะ path แบบนี้แยก folder ( 1 folder มีหลายคน)
# # use for process file
# prefix = 'r'
# suffix = '_fdt_paths.nii'


# ################ START comment
# # TODO comment this if not want TBI data
# project = "TBI"
# modalities = tbi_modalities
# _, _, labels_data, _ = get_data(
#         modalities[0])
# ################ END comment

# ############### START comment
# # TODO comment this if not want TEST covid data
# project = "Covidtest"
# modalities  = covid_test_modalities
# _, labels_data = get_covid_data(modalities[0])
# ############### END comment

############### START comment
# TODO comment this if not want Kid Lead data
project = "KidLead"
modalities  = kidlead_modalities
_, labels_data = get_kidlead_data(modalities[0])
############### END comment

# 65489132
random_state = 65489132
torch.manual_seed(random_state)
random.seed(random_state)
np.random.seed(random_state)
kf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)


batch_size = 2
img_size = 64
test_size = 0.2
val_size = 0.2
n_classes = 1
val_size = val_size/(1-test_size)
epochs = 50
lr = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_names = [ 'vgg11', 'resnet10']
checkpoint_types = ['rocauc', 'bacc']
if n_classes<2:
    loss_function = nn.BCEWithLogitsLoss()
else:
    loss_function = nn.CrossEntropyLoss()

results_val = []
results_test = []

all_index = list(range(len(labels_data)))
train_index_all, test_index = train_test_split(all_index,  test_size= test_size, random_state=random_state, stratify=labels_data)

for modality in modalities:
    mask_filenames = []

    # ################ START comment
    # # TODO comment this if not want TBI data
    # filenames, mask_filenames, labels_data, no_data = get_data(
    #     modality)
    # ################ END comment

    ################ START comment
    # TODO comment this if not want Kid Lead data
    filenames, labels_data = get_kidlead_data(modalities[0])
    ################ END comment

    print(f"NODATA {modality['modality']}: {no_data}")
    print(len(filenames))

    print("all positive ",np.sum([labels_data[i] for i in train_index_all]))
    for i_fold, (train_index, val_index) in enumerate(kf.split(train_index_all, np.array(labels_data)[train_index_all])):
        train_index = [train_index_all[i] for i in train_index]
        val_index = [train_index_all[i] for i in val_index]
        # print("val index ",np.sum([labels_data[i] for i in val_index]))
        # print("train index ",np.sum([labels_data[i] for i in train_index]))
        x_train = [filenames[i] for i in train_index]
        x_val = [filenames[i] for i in val_index]
        x_test = [filenames[i] for i in test_index]

        y_train = [labels_data[i] for i in train_index]
        y_val = [labels_data[i] for i in val_index]
        y_test = [labels_data[i] for i in test_index]

        if len(mask_filenames) ==0:

            mean_data, std_data = get_normalization_param_nomask(x_train)
        else:
            mean_data, std_data = get_normalization_param(x_train, mask_filenames)

        train_transform = get_transform(mean_data, std_data, mode = 'train')
        val_transform = get_transform(mean_data, std_data, mode = 'val')

        train_loader = get_loader(x_train, y_train, train_transform, mode = 'train', batch_size =batch_size, img_size = img_size)
        val_loader = get_loader(x_val, y_val,val_transform, mode = 'val', batch_size =batch_size, img_size = img_size)
        test_loader = get_loader(x_test, y_test,val_transform, mode = 'test', batch_size =batch_size, img_size = img_size)

        for model_name in model_names:
            print(f"-------------------MODEL {model_name} MOD {modality['modality']}-------------------")
            run_name = f"{project} {modality['prefix']}_{modality['modality']}_{model_name}"
            model = get_model(model_name, device, n_classes = n_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)
            
            writer = SummaryWriter(f"runs/{i_fold}_{run_name}")
            current_epoch = 0
            best_bacc = 0
            best_auc = 0
            savepath = f"{i_fold}_{run_name}"

            for i in range(current_epoch , epochs+1):

                train_loss, preds_prob, labels = epoch_iter(train_loader, model, loss_function, optimizer, device)
                print(f"train_loss: {train_loss};")
                writer.add_scalar('train_loss', train_loss, i)
                _, _, _ = add_metrics(writer, preds_prob, labels, train_loss, i, mode = 'train')

                val_loss, preds_prob, labels = epoch_iter(val_loader, model, loss_function, optimizer, device, mode = 'val')
                print(f"val_loss: {val_loss};")
                writer.add_scalar('val_loss', val_loss, i)

                f1, rocauc, bacc = add_metrics(writer, preds_prob, labels, val_loss, i, mode = 'val')
                if bacc>best_bacc:
                    best_bacc = bacc
                    save_checkpoint(savepath, model, optimizer, 'bacc', i)
                if rocauc>best_auc:
                    best_auc = rocauc
                    save_checkpoint(savepath, model, optimizer, 'rocauc', i)

            def testing(checkpoint_type):
                checkpoints_dir = f"checkpoints/{savepath}"

                model_path = os.path.join(checkpoints_dir, f'model-{checkpoint_type}.ckpt')

                model_state_dict = torch.load(model_path, map_location=torch.device(device))
                model.load_state_dict(model_state_dict)

                val_loss, preds_prob, labels = epoch_iter(test_loader, model, loss_function, optimizer, device, mode = 'test')
                acc, bacc, precision, recall, f1,rocauc, specificity = get_metrics(preds_prob, labels)

                results_test.append({
                    'model': f"{model_name}",
                    'modality':modality['modality'],
                    'checkpoint_type': checkpoint_type,
                    'fold': i_fold,
                    'acc': acc,
                    'bacc': bacc,
                    'precision': precision,
                    'specificity': specificity,
                    'recall': recall,
                    'f1': f1,
                    'rocauc': rocauc
                })

                val_loss, preds_prob, labels = epoch_iter(val_loader, model, loss_function, optimizer, device, mode = 'validation')
                acc, bacc, precision, recall, f1,rocauc, specificity = get_metrics(preds_prob, labels)
                results_val.append({
                    'model': f"{model_name}",
                    'modality':modality['modality'],
                    'checkpoint_type': checkpoint_type,
                    'fold': i_fold,
                    'acc': acc,
                    'bacc': bacc,
                    'precision': precision,
                    'specificity': specificity,
                    'recall': recall,
                    'f1': f1,
                    'rocauc': rocauc
                })

            for checkpoint_type in checkpoint_types:
                testing(checkpoint_type)

results_val = pd.DataFrame(results_val)
results_test = pd.DataFrame(results_test)

results_test.to_csv(f'results/test_{project}.csv', index = False)
results_val.to_csv(f'results/val_{project}.csv', index = False)

# nohup python run.py