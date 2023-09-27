import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader





def epoch_iter(loader, model, loss_function, optimizer, device,  mode = 'train'):
    losses = []
    print(f"**************{mode}*************")
    preds = []
    labels = []
    for img, label in tqdm(loader):
        optimizer.zero_grad()
        model.zero_grad()

     
        label = label.to(device)
        img = img.to(device)

        if mode == 'train':
            model.train()
            pred = model(img)
        else:
            model.eval()
            with torch.no_grad():
                pred = model(img)

  
        loss = loss_function(pred, label.to(torch.float))
        preds.extend(pred.cpu().detach().tolist())
        labels.extend(label.cpu().detach().tolist())

        if mode == 'train':
            loss.backward()
            optimizer.step()

        loss = loss.cpu().detach().numpy()
        losses.append(loss)
        
    
    preds = np.array(preds)

    labels = np.array(labels)


    optimizer.zero_grad()
    model.zero_grad()

    return np.sum(losses)/(len(losses) + 1e-16), preds, labels

def get_metrics(preds_prob, labels, round_to = 3):
    preds = preds_prob

    # preds = np.argmax(preds, axis = 1)
    preds = [1 if p>0.5 else 0 for p in preds]
    acc =round( metrics.accuracy_score(labels, preds),round_to)
    bacc = round(metrics.balanced_accuracy_score(labels, preds),round_to)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        labels, preds, beta = 1, average = 'binary')
    precision = round(precision,round_to)
    recall = round(recall,round_to)
    f1 = round(f1,round_to)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn+fp)

    
    # labels_onehot = np.zeros((labels.size, labels.max() + 1))
    # labels_onehot[np.arange(labels.size), labels] = 1
    rocauc = round(
        metrics.roc_auc_score( 
        labels, preds_prob, average = 'macro', multi_class = 'raise')
        ,round_to)
    # ovr
    # precision, recall, _ = metrics.precision_recall_curve(labels, preds)
    # prauc = metrics.auc(recall, precision)
    return acc, bacc, precision, recall, f1,rocauc, specificity

def add_metrics_regression(preds, labels, round_to = 3, mode = 'train'):
    r2 = metrics.r2_score(labels, preds)
    mape = metrics.mean_absolute_percentage_error(labels, preds)
    mae = metrics.mean_absolute_error(labels, preds)
    rmse = np.sqrt(metrics.mean_squared_error(labels, preds))

    r2 = round(r2,round_to)
    mape = round(mape,round_to)
    mae = round(mae,round_to)
    rmse = round(rmse,round_to)

    print(
        f"r2_{mode}: {r2}; mape_{mode}: {mape};" +
        f" mae_{mode}: {mae}; rmse_{mode}: {rmse}"
        )
    return rmse, mae, mape, r2

def add_metrics(
    writer, preds_prob, labels, losses, current_epoch, 
    mode = 'train', round_to = 3):

    # preds_prob = F.softmax(torch.Tensor(preds_prob)).tolist()

    acc, bacc, precision, recall, f1,rocauc, specificity =   get_metrics(preds_prob, labels, round_to = round_to)

    # precision, recall, _ = metrics.precision_recall_curve(labels, preds)
    # prauc = metrics.auc(recall, precision)

    loss = round(np.mean(losses), round_to)
    writer.add_scalar(f'loss_{mode}',loss,current_epoch)
    print(f"loss_{mode}: {loss};")

    print(
        f"acc_{mode}: {acc}; bacc_{mode}: {bacc};" +
        f" precision_{mode}: {precision}; recall_{mode}: {recall}; f1_{mode}: {f1};" +
        f" rocauc_{mode}: {rocauc};"
        )

    writer.add_scalar(f'acc_{mode}',acc,current_epoch)
    writer.add_scalar(f'bacc_{mode}',bacc,current_epoch)
    writer.add_scalar(f'precision_{mode}',precision,current_epoch)
    writer.add_scalar(f'recall_{mode}',recall,current_epoch)
    writer.add_scalar(f'f1_{mode}',f1,current_epoch)
    writer.add_scalar(f"specificity_{mode}", specificity, current_epoch)

    writer.add_scalar(f'rocauc_{mode}',rocauc,current_epoch)

    return f1, rocauc, bacc

def create_dir_ifnot_exist(folder):
    if os.path.exists(folder):
        return
    
    os.mkdir(folder)

def save_checkpoint(folder, model, optimizer, save_type, current_epoch):
    create_dir_ifnot_exist("checkpoints")
    savepath = os.path.join("checkpoints", folder)
    create_dir_ifnot_exist(savepath)
    

    torch.save(model.state_dict(), os.path.join(savepath, f'model-{save_type}.ckpt'))
    torch.save(optimizer.state_dict(), os.path.join(savepath, f'opt-{save_type}.ckpt'))

