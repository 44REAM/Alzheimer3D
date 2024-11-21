import os

from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
import SimpleITK as sitk
import torch
import monai.transforms as monai_transforms


class MRIDataset(Dataset):
    def __init__(self, x, y, img_size=64, transform=None, img_spacing=2):
        self.x = x
        self.label = y
        self.img_size = img_size
        self.img_spacing = img_spacing
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]
        filepath = self.x[idx]
        image = sitk.ReadImage(filepath)

        # resample image to equal voxel spacing
        image = self.resample(image, self.get_ref_image(image, self.img_size))
        image = sitk.GetArrayFromImage(image)

        image = image.reshape(1, *image.shape)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_ref_image(self, img, img_size):

        reference_origin = img.GetOrigin()
        reference_direction = img.GetDirection()
        # Arbitrary sizes, smallest size that yields desired results.
        reference_size = [img_size]*3
        # reference_size = (int(img.GetSize()[0]/img.GetSpacing()[0] * img_spacing),
        #                     int(img.GetSize()[1]/img.GetSpacing()[1] * img_spacing),
        #                     int(img.GetSize()[2]/img.GetSpacing()[2] * img_spacing))

        reference_spacing = (img.GetSize()[0]/img_size*img.GetSpacing()[0],
                             img.GetSize()[1]/img_size*img.GetSpacing()[1],
                             img.GetSize()[2]/img_size*img.GetSpacing()[2])

        reference_image = sitk.Image(reference_size,  sitk.sitkFloat64)
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)
        return reference_image

    def resample(self, image, reference_image):

        interpolator = sitk.sitkLinear
        default_value = 0.0
        transform = sitk.Transform(3, sitk.sitkIdentity)
        return sitk.Resample(image, reference_image, transform,
                             interpolator, default_value)


def get_list_files_and_labels_full_tbm(basepath, modalities):

    real_folderpaths = []
    labels = []
    nodata = []
    for foldername in os.listdir(basepath):
        folderpath = os.path.join(basepath, foldername)

        have_all = True
        for mod in modalities:
            filepath = os.path.join(folderpath, mod)
            if not os.path.exists(filepath):
                have_all = False

        if have_all:
            label = 1 if 'mci' in foldername else 0
            real_folderpaths.append(folderpath)
            labels.append(label)
        else:
            nodata.append(folderpath)
            continue

    return np.array(real_folderpaths), np.array(labels), np.array(nodata)


def get_list_files_and_labels_ad(basepath, excel_ad_path, modalities, require_first_visit = True):

    label_map = {
        'ad': 2,
        'mci': 1,
        'normal': 0
    }

    df = pd.read_csv(excel_ad_path)
    middle_paths = list(df['label'] + '/' +
                        df['subject_id'] + '/' + df['scan_id'])
    labels = list(df['label'])
    followups = list(df['followup'])
    real_folderpaths = []
    real_labels = []
    nodata = []
    for foldername, label, followup in zip(middle_paths, labels, followups):

        if label == 'unknown':
            continue

        folderpath = os.path.join(basepath, foldername)

        have_all = True
        for mod in modalities:
            filepath = os.path.join(folderpath, mod)

            if not os.path.exists(filepath):
                have_all = False
        if have_all:
            if followup != 'firstvisit' and require_first_visit:
                continue
            real_folderpaths.append(folderpath)
            real_labels.append(label_map[label])

        else:
            nodata.append(folderpath)
            continue

    return np.array(real_folderpaths), np.array(real_labels), np.array(nodata)


def get_loader(file_list, labels, transform, mode='val', batch_size=32, pin_memory=True, num_workers=4, img_size=64):

    dataset = MRIDataset(
        file_list, labels, transform=transform, img_size=img_size)

    if mode == 'train':
        # clc, class_sample_count = np.unique(dataset.label, return_counts=True)
        # class_weight = 1/ (class_sample_count  )* len(dataset)/2
        # class_sample_weight = class_weight[dataset.label]
        # sampler =  WeightedRandomSampler(class_sample_weight, num_samples=len(dataset), replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=pin_memory, num_workers=num_workers)

    return loader


def get_normalization_param(filenames, mask_filenames):
    print("Get normalization param ... ")
    mean_data = []
    var_data = []
    for img_path, mask_path in zip(filenames, mask_filenames):
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)

        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)

        mean_img_array = img_array[mask_array > 0].mean()
        var_img_array = img_array[mask_array > 0].var()

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
        img_array = sitk.GetArrayFromImage(img)

        # lower = np.percentile(img_array, 5)
        # upper = np.percentile(img_array, 95)
        mean_img_array = img_array[(img_array != 0)].mean()
        var_img_array = img_array[(img_array != 0)].var()

        mean_data.append(mean_img_array)
        var_data.append(var_img_array)
    mean_data = np.array(mean_data)
    var_data = np.array(var_data)

    mean_data = mean_data.mean()

    lower = np.percentile(var_data, 5)
    upper = np.percentile(var_data, 95)
    std_data = np.sqrt(var_data.mean())

    # print(std_data)
    # print(mean_data)
    return mean_data, std_data


def prepare_prelim_index(file_list, labels, random_seed=65489132, suffix=''):


    index_list = np.array(range(len(labels)))
    train_index, test_index = train_test_split(
            index_list, test_size=0.20, random_state=random_seed, stratify=labels)

    train, test = file_list[train_index], file_list[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    index_list = np.array(range(len(y_train)))

    train_index, val_index = train_test_split(
        index_list, test_size=0.10, random_state=random_seed, stratify=y_train)

    train, val = train[train_index], train[val_index]
    y_train, y_val = y_train[train_index], y_train[val_index]

    train_filenames = np.array(train, dtype=object)
    val_filenames = np.array(val, dtype=object)
    test_filenames = np.array(test, dtype=object)

    train_labels = np.array(y_train, dtype=object)
    val_labels = np.array(y_val, dtype=object)
    test_labels = np.array(y_test, dtype=object)

    np.save(f'indices/train_filenames_{suffix}.npy', train_filenames)
    np.save(f'indices/val_filenames_{suffix}.npy', val_filenames)
    np.save(f'indices/test_filenames_{suffix}.npy', test_filenames)

    np.save(f'indices/train_labels_{suffix}.npy', train_labels)
    np.save(f'indices/val_labels_{suffix}.npy', val_labels)
    np.save(f'indices/test_labels_{suffix}.npy', test_labels)


def prepare_prelim_index_subject(file_list, labels, random_seed=65489132, suffix=''):

    subject_list = [name.split('/')[6] for name in file_list]

    test = {}
    for subject, label in zip(subject_list, labels):
        test_label = test.get(subject, None)
        if test_label is None:
            test[subject] = label
        elif test_label != label:
            print(subject)
            raise ValueError("same subject get different labels")
        
    unique_subject_list = np.unique(subject_list)
    unique_labels = np.array([test[subject] for subject in unique_subject_list])

    index_list = np.array(range(len(unique_subject_list)))
    train_index, test_index = train_test_split(
            index_list, test_size=0.20, random_state=random_seed, stratify=unique_labels)

    train, test = unique_subject_list[train_index], unique_subject_list[test_index]
    y_train, y_test = unique_labels[train_index], unique_labels[test_index]
    index_list = np.array(range(len(y_train)))

    train_index, val_index = train_test_split(
        index_list, test_size=0.10, random_state=random_seed, stratify=y_train)

    train, val = train[train_index], train[val_index]
    y_train, y_val = y_train[train_index], y_train[val_index]

    train_combine = np.array([[name, label] for name, label in zip(file_list, labels) if name.split('/')[6] in train])
    val_combine = np.array([[name, label] for name, label in zip(file_list, labels) if name.split('/')[6] in val])
    test_combine = np.array([[name, label] for name, label in zip(file_list, labels) if name.split('/')[6] in test])

    train, y_train = train_combine[:, 0], train_combine[:, 1].astype(int)
    val, y_val= val_combine[:, 0], val_combine[:, 1].astype(int)
    test, y_test= test_combine[:, 0], test_combine[:, 1].astype(int)

    train_filenames = np.array(train, dtype=object)
    val_filenames = np.array(val, dtype=object)
    test_filenames = np.array(test, dtype=object)

    train_labels = np.array(y_train, dtype=object)
    val_labels = np.array(y_val, dtype=object)
    test_labels = np.array(y_test, dtype=object)

    np.save(f'indices/train_filenames_{suffix}.npy', train_filenames)
    np.save(f'indices/val_filenames_{suffix}.npy', val_filenames)
    np.save(f'indices/test_filenames_{suffix}.npy', test_filenames)

    np.save(f'indices/train_labels_{suffix}.npy', train_labels)
    np.save(f'indices/val_labels_{suffix}.npy', val_labels)
    np.save(f'indices/test_labels_{suffix}.npy', test_labels)

def get_transform(mean, std, mode='train', self_normalized=False):
    transforms_list = []

    def standard(img):
        return (img - mean)/std
    if self_normalized:
        transforms_list.append(
            monai_transforms.NormalizeIntensity(nonzero=True))
    if mode == 'train':
        pass
        # transforms_list.append(monai_transforms.RandRotate90(prob = 0.3))
        transforms_list.append(
            monai_transforms.RandGaussianSmooth(
                sigma_x=(0.1, 0.5),
                sigma_y=(0.1, 0.5),
                sigma_z=(0.1, 0.5), prob=0.5))
        transforms_list.append(monai_transforms.RandAffine(prob=1, rotate_range=(
            np.pi/36, np.pi/36, np.pi/36), translate_range=[(-2, 2), (-2, 2), (-2, 2)], padding_mode='zeros'))

    transforms_list.append(torch.Tensor)
    transforms_list.append(standard)
    return monai_transforms.Compose(transforms_list)


if __name__ == '__main__':
    basepath_tbm = '/data1/TBM/TBM-AI_data/data_by_subject'
    basepath_ad = '/data1/ADCRA/normalized_data/data'
    excel_ad_path = '/data1/ADCRA/normalized_data/subj_mapping/ad2_mapping_table.csv'
    modalities_tbm = [
        'DTI/wdtifit_FA.nii',
        'DTI/wdtifit_MD.nii',
        'DTI/wfdt_paths.nii',
        'DTI/wnfdt_paths.nii',
        'DTI/wnodif.nii',
        'T2s/wR2S.nii',
        'fmri_mirror/wcon_0001.nii'
    ]
    modalities_ad = [
        'DTI/wdtifit_FA.nii',
        'DTI/wdtifit_MD.nii',
        'DTI/wfdt_paths.nii',
        'DTI/wnum_fcd.nii',
        'DTI/wnodif.nii',
        't2s/wrR2S.nii',
        # 'working_memory/wrcon_0001.nii'
    ]
    ########################################
    filepaths, labels, nodata = get_list_files_and_labels_ad(
        basepath_ad, excel_ad_path, modalities_ad, require_first_visit=False)

    prepare_prelim_index_subject(filepaths, labels, suffix='ad_subject')
    ########################################
    filepaths, labels, nodata = get_list_files_and_labels_ad(
        basepath_ad, excel_ad_path, modalities_ad)
    prepare_prelim_index(filepaths, labels, suffix='ad_first_visit')
    ########################################
    filepaths, labels, nodata = get_list_files_and_labels_full_tbm(basepath_tbm, modalities_tbm)
    prepare_prelim_index(filepaths, labels, suffix = 'tbm')
