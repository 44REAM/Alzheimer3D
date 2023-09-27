import os

import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom

from pycaret.classification import *


basepath = "/data0/AI_SAMPLES/AI_SAMPLES_18DEC2022"
csvpath = '/data0/AI_SAMPLES/mapping_18DEC2022.csv'

df_data = pd.read_csv(csvpath, header=None, index_col = 0, usecols=[0, 1, 4])
df_data.columns = ['class', 'name']
df_data.index.name = None

filenames = []
labels = []

for name, label in zip(df_data.name, df_data['class']):
    folderpath = os.path.join(basepath, name)
    try:
        fdt_paths_path = os.path.join(folderpath, 'DTI', 'fdt_paths_vol0.nii.gz')
        fdt_paths = sitk.ReadImage(fdt_paths_path)
        a = sitk.GetArrayFromImage(fdt_paths)
        if a.shape[0] != 62 or a.shape[1] != 128 or a.shape[2] != 128:
            print(a.shape)
            continue
    except:
        continue
    filenames.append(name)
    if label == 'mci':
        labels.append(1)
    elif label == 'normal':
        labels.append(0)
    elif label == 'mmd':
        labels.append(2) 


df_data = pd.DataFrame()
df_data['label'] = labels
df_data['name'] = filenames


datas = []
y = []

for name in df_data.name:
    folderpath = os.path.join(basepath, name)
    try:
        fdt_paths_path = os.path.join(folderpath, 'DTI', 'fdt_paths_vol0.nii.gz')
        fdt_paths = sitk.ReadImage(fdt_paths_path)
        a = sitk.GetArrayFromImage(fdt_paths)
        if a.shape[0] != 62 or a.shape[1] != 128 or a.shape[2] != 128:
            print(a.shape)
            continue
        if 'mci' in name:
            y.append(1)
        elif 'normal' in name:
            y.append(0)
        elif 'mmd' in name:
            y.append(2)

        a = zoom(a, (64/62*0.5, 0.5*0.5, 0.5*0.5))
        datas.append(a)
    except Exception as e:
        print(e)

datas = np.array(datas)
y = np.array(y)
df = pd.DataFrame(datas.reshape((198,-1 )))
df = df.loc[:, df.mean(axis = 0) >0.01 ]

corr =np.corrcoef(df.values, rowvar=False)
with open('corr.npy', 'wb') as f:

    np.save(f, corr, allow_pickle = False)
corr_matrix = pd.DataFrame(corr, columns = df.columns)


# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
df.drop(to_drop, axis=1, inplace=True)