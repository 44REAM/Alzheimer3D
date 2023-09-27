import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import monai.transforms as monai_transforms

def get_ref_image(img, img_size):

    reference_origin = img.GetOrigin()
    reference_direction = img.GetDirection()
    reference_size = [img_size]*3 # Arbitrary sizes, smallest size that yields desired results. 
    reference_spacing = (img.GetSize()[0]/img_size*img.GetSpacing()[0],
                         img.GetSize()[1]/img_size*img.GetSpacing()[1],
                         img.GetSize()[2]/img_size*img.GetSpacing()[2])

    reference_image = sitk.Image(reference_size,  sitk.sitkFloat64)
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    return reference_image

def resample(image, reference_image):

    interpolator = sitk.sitkLinear
    default_value = 0.0
    transform = sitk.Transform(3, sitk.sitkIdentity)
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

class MRIDataset(Dataset):
    def __init__(self, x, y, img_size = 64, transform=None):
        self.x = x
        self.y = y
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label = self.y[idx]
        filepath = self.x[idx]
        fdt_paths = sitk.ReadImage(filepath)
        
        # resample image to equal voxel spacing 
        fdt_paths = resample(fdt_paths, get_ref_image(fdt_paths, self.img_size))
        image = sitk.GetArrayFromImage(fdt_paths)

        image = image.reshape(1,self.img_size,self.img_size,self.img_size)
        
        
        if self.transform:
            image = self.transform(image)

        return image, label

def get_loader_regression(x, y, labels_class, trans, mode, img_size = 64, batch_size = 2, num_workers = 0):

    train_dataset = MRIDataset(x, y, transform=trans, img_size  =img_size)
    
    if mode == 'train':
        # for imbalance class
        labels_unique, counts = np.unique(labels_class, return_counts=True)
        weights = counts.sum()/counts
        weights = weights/weights.sum()
        weights = [weights[int(i)] for i in labels_class]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        x_loader = DataLoader(
            train_dataset, 
            batch_size= batch_size, 
            drop_last=False,
            shuffle =False,
            num_workers=num_workers,
            sampler = sampler,
            pin_memory = True)
    else:

        x_loader = DataLoader(
            train_dataset, 
            batch_size= batch_size, 
            drop_last=False,
            shuffle =False,
            num_workers=num_workers,
            pin_memory = True)
    return x_loader

def get_loader(x, y, trans, mode, img_size = 64, batch_size = 2, num_workers = 0):

    train_dataset = MRIDataset(x, y, transform=trans, img_size  =img_size)
    
    if mode == 'train':
        # for imbalance class
        labels_unique, counts = np.unique(train_dataset.y, return_counts=True)
        weights = counts.sum()/counts
        weights = weights/weights.sum()
        weights = [weights[int(i)] for i in train_dataset.y]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        x_loader = DataLoader(
            train_dataset, 
            batch_size= batch_size, 
            drop_last=False,
            shuffle =False,
            num_workers=num_workers,
            sampler = sampler,
            pin_memory = True)
    else:

        x_loader = DataLoader(
            train_dataset, 
            batch_size= batch_size, 
            drop_last=False,
            shuffle =False,
            num_workers=num_workers,
            pin_memory = True)
    return x_loader

def get_normalization_param(filenames, mask_filenames):
    mean_data = []
    var_data = []
    for img_path, mask_path in zip(filenames, mask_filenames):
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage( img)

        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage( mask)

        mean_img_array = img_array[mask_array>0].mean()
        var_img_array = img_array[mask_array>0].var()
        
        mean_data.append(mean_img_array)
        var_data.append(var_img_array)

    mean_data = np.mean(mean_data)
    std_data = np.sqrt(np.mean(var_data))
    return mean_data, std_data

def get_normalization_param_nomask(filenames):
    mean_data = []
    var_data = []
    for img_path in filenames:
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage( img)

        lower = np.percentile(img_array, 5)
        upper = np.percentile(img_array, 95)
        mean_img_array = img_array[(img_array>lower) & (img_array<upper)].mean()
        var_img_array = img_array[(img_array>lower) & (img_array<upper)].var()
        
        mean_data.append(mean_img_array)
        var_data.append(var_img_array)
    mean_data = np.array(mean_data)
    var_data = np.array(var_data)

    lower = np.percentile(mean_data, 5)
    upper = np.percentile(mean_data, 95)

    mean_data = mean_data[(mean_data>lower) & (mean_data<upper)].mean()

    lower = np.percentile(var_data, 5)
    upper = np.percentile(var_data, 95)
    std_data = np.sqrt(var_data[(var_data>lower) & (var_data<upper)].mean())

    # print(std_data)
    # print(mean_data)
    return mean_data, std_data

def get_transform(mean, std, mode= 'train'):
    transforms_list = []
    
    def standard(img):
        return (img - mean)/std
    if mode == 'train':
        # transforms_list.append(monai_transforms.RandRotate90(prob = 0.3))
        transforms_list.append(
            monai_transforms.RandGaussianSmooth(
            sigma_x=(0.1, 0.5), 
            sigma_y=(0.1, 0.5), 
            sigma_z=(0.1, 0.5), prob=0.3))
        transforms_list.append(monai_transforms.RandAffine(prob = 0.3, rotate_range =(np.pi/24, np.pi/24, np.pi/24), translate_range  =[(-2,2), (-2,2), (-2,2)], padding_mode  = 'zeros'))


    transforms_list.append(torch.Tensor)
    transforms_list.append(standard)
    return monai_transforms.Compose(transforms_list)

def get_data(modality, 
             mask_prefix = "", mask_suffix = "_wnodif_brain_mask.nii"):
    
    csvpath = '/data1/TBM/data_for_AI/subjects_info/final_TBM_subjects_info.csv'
    mask_basepath = "/data1/TBM/data_for_AI/new_data/DTI/nodif_brain_mask"
    basepath = modality['basepath']
    modality_name = modality['modality']
    use_file = modality['use_file']
    prefix = modality['prefix']
    suffix = modality['suffix']

    df_data = pd.read_csv(csvpath)

    if (modality_name is not None) and (use_file is not None):
        form = 'raw'
    elif (prefix is not None) and (suffix is not None):
        form = 'process'
    else:
        raise Exception("")

    filenames = []
    mask_filenames = []
    labels_data = []

    def get_path_and_label(name, label, form = 'raw', prefix = 'wr', suffix = '_fdt_paths.nii'):
        
        if form == 'raw':
            if label.lower() == 'mci':
                category = 1
                fdt_paths_path = os.path.join(basepath, 'MCI',name, modality_name, use_file)
            elif label.lower() == 'normal':
                category = 0
                fdt_paths_path = os.path.join(basepath, 'Normal',name, modality_name, use_file)
            elif label.lower() == 'mmd':
                category = 2
                fdt_paths_path = os.path.join(basepath, 'AD',name, modality_name, use_file)
            else:
                raise ValueError(f"No label name {label}")

        else:
            if label.lower() == 'mci':
                category = 1
            elif label.lower() == 'normal':
                category = 0

            elif label.lower() == 'mmd':
                category = 2
            else:
                raise ValueError(f"No label name {label}")
            fdt_paths_path = os.path.join(basepath, prefix + name + suffix)
        

        return fdt_paths_path, category


    first_shape_flag = False
    first_shape = (0,0,0)
    no_data = []
    names = []
    for name, label in zip(df_data.label_id, df_data['label']):

        fdt_paths_path, category = get_path_and_label(name, label, form = form, prefix = prefix, suffix = suffix)

        try:
            img = sitk.ReadImage(fdt_paths_path)
            img_array = sitk.GetArrayFromImage( img)

            if not first_shape_flag:
                first_shape = img_array.shape
                first_shape_flag = True

            elif img_array.shape[0] != first_shape[0] or img_array.shape[1] != first_shape[1] or img_array.shape[2] != first_shape[2]:
                print(f"SHAPEERROR: {name} {img_array.shape}")
                continue
        except:
            no_data.append(name)
            continue
        mask_filenames.append(os.path.join(mask_basepath, mask_prefix + name + mask_suffix))
        
        labels_data.append(category)
        filenames.append(fdt_paths_path)
        names.append(name)


        
    assert len(labels_data) == len(filenames)

    return filenames, mask_filenames,  labels_data, no_data


def get_covid_data(modality):

    tmp = os.listdir(modality['basepath'])
    tmp = [os.path.join(modality['basepath'],name) for name in tmp]
    filenames = []
    for img_path in tmp:
        
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage( img)
        if img_array.mean() == 0:
            print(f"ERROR {img_path}")
            continue
        filenames.append(img_path)

    labels_data = [1 if "Covid" in name else 0 for name in filenames]

    return filenames, labels_data


def get_kidlead_data(modality):

    tmp = os.listdir(modality['basepath'])
    tmp = [os.path.join(modality['basepath'],name)  for name in tmp]
    filenames = []
    for img_path in tmp:
        
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage( img)
        if img_array.sum() == 0:
            print(f"ERROR {img_path}")
            continue
        filenames.append(img_path)

    labels_data = [1 if "PAT" in name else 0 for name in filenames]


    return filenames, labels_data

def get_kidlead_lead(filenames):
    basepath = '/data1/Kid_projects/KidLead/KidLead_analysis/project_info/PBproject_info.xlsx'
    data = pd.read_excel(basepath, sheet_name='Total')
    mapping_lead = pd.read_csv('mapping_lead.csv', header=None)
    mapping_normal = pd.read_csv('mapping_normal.csv', header=None)
    data['combine_name'] = data['First_Name'].str.strip() + "_" + data['Surname'].str.strip()

    data = data.merge(mapping_lead, left_on = 'combine_name', right_on = 0, how = 'left')
    data = data.merge(mapping_normal, left_on = 'HN', right_on = 0, how = 'left')
    data['1_x'] = data['1_x'].fillna(data['1_y'])
    data['filename'] = data['1_x']

    leads = []
    for name in filenames:
        for n, lead in zip(data['filename'], data['Lead(Pb)']):
            if n in name:
                leads.append(lead)
                break
    assert len(leads) == len(filenames)
    return leads

if __name__ == '__main__':
    pass