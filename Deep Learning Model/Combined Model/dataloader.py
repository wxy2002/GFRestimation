import torch
import nibabel as nib
import os
import tqdm
from skimage.transform import resize
import numpy as np
import pandas as pd
import math
import random

traget_slices = 16  # 目标切片数
traget_shape = 128  # 目标图像大小

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_nii(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return torch.tensor(data).permute(2, 0, 1)  # Change to (C, H, W) format

def show_CT(image_tensor, slice_index):
    # Show a specific 2D slice
    import matplotlib.pyplot as plt
    plt.imshow(image_tensor[slice_index, :, :], cmap='gray')
    plt.axis('off')
    plt.show()
    plt.close()

def reshape_CT_bao(ct_image, target_slices=64, n=512):
    # 假设每个CT图像的第一个维度是切片数
    '''resized_slices = []
    for slice_idx in range(ct_image.shape[0]):
        slice_data = ct_image[slice_idx, :, :]
        resized_slice = resize(slice_data, (n, n), mode='constant', anti_aliasing=True)
        resized_slices.append(resized_slice)
    # 如果切片数少于目标值，则进行填充
    if len(resized_slices) < target_slices:
        while len(resized_slices) < target_slices:
            resized_slices.append(np.zeros((n, n)))
    # 如果切片数多于目标值，则进行压缩
    elif len(resized_slices) > target_slices:
        step = len(resized_slices) / target_slices
        # resized_slices = [resized_slices[int(i * step)] for i in range(target_slices)]
        # resized_slices = [mean(resized_slices[int(i * step):int((i + 1) * step)]) for i in range(target_slices)]
        resized_slices = [np.mean(resized_slices[int(i * step):int((i + 1) * step)], axis=0) for i in range(target_slices)]
    resized_slices = np.array(resized_slices)'''
    resized_slices = resize(ct_image, (target_slices, n, n), mode='constant', anti_aliasing=True)
    return torch.tensor(resized_slices)

def CT_preprocess(image_tensor):
    hu_min, hu_max = -100, 400
    # 限制HU值范围，不在范围内的值将被设置为hu_min
    # image_tensor = torch.clamp(image_tensor, min=hu_min, max=hu_max)
    image_tensor[image_tensor < hu_min] = hu_min
    image_tensor[image_tensor > hu_max] = hu_min
    # 归一化到[0, 1]范围
    image_tensor = (image_tensor - hu_min) / (hu_max - hu_min)
    # image_tensor = (image_tensor - image_tensor.mean()) / image_tensor.std()  # 标准化
    image_tensor = reshape_CT_bao(image_tensor, traget_slices, n=traget_shape)
    return image_tensor

def load_CT_data(file_path, use_mask=False):
    ct_names = pd.read_csv(file_path + "filtered_data.csv")['ID'].tolist()
    # 判断ct_names中的文件是否都在file_paths中。先为ct_names添加nii.gz后缀，使用字符串拼接
    ct_names = [os.path.join(file_path + "CT/", f"{name}.nii.gz") for name in ct_names]
    print(f"Total CT files found: {len(ct_names)}")
    print(f"Found {len(ct_names)} CT files to process.")
    data = []
    for file in tqdm.tqdm(ct_names, desc="Loading CT files"):
        try:
            ct_image = load_nii(file)
            if use_mask:
                mask_image = load_nii(file.replace("CT", "Mask"))
                ct_image[mask_image == 0] = -100
            ct_image = CT_preprocess(ct_image)
            data.append(ct_image)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    data = torch.stack(data)  # Stack the list of tensors into a single tensor
    print(f"Loaded {len(data)} CT images.")
    torch.save(ct_names, file_path + 'ct_names.pt')
    return data

def create_CT_pt(hospital_name, use_mask=False):
    file_path = f"data/{hospital_name}/"
    if not os.path.exists(file_path):
        print(f"Directory {file_path} does not exist.")
        return
    data = load_CT_data(file_path, use_mask)
    torch.save(data, f"data/{hospital_name}/ct_data.pt")
    print(f"Saved CT data for {hospital_name} to {file_path}ct_data.pt")

def create_dataloader(hospital_name, batch_size=16, shuffle=True, use_aligned=False, 
                      split=False, stage='arterial', side='Left', give_id=False):
    from sklearn.model_selection import train_test_split
    file_path = f"data_All/{hospital_name}/"
    if not os.path.exists(file_path):
        print(f"Directory {file_path} does not exist.")
        return
    data_a = torch.load(file_path + f'ct_arterial.pt')
    data_v = torch.load(file_path + f'ct_venous.pt')

    df = pd.read_csv(file_path + "filtered_data.csv")
    if hospital_name == 'gzyy':
        df['ID'] = [i.replace('gzyy', '') for i in df['ID']]
    if hospital_name == 'gdph':
        df['ID'] = [i.replace('gdph', '') for i in df['ID']]

    df_clin = pd.read_csv(f'data_All/临床和组学/组学/{hospital_name}.csv')
    df_clin['ID'] = [i.replace(hospital_name, '') for i in df_clin['ID']]
    df = df.merge(df_clin, on='ID', how='left')
    # 去除prediction_label_left、prediction_label_right列
    # df = df.drop(columns=['prediction_label_left', 'prediction_label_right'])
    colname = df.columns.tolist()
    # 去除ID, left_GFR, right_GFR
    colname.remove('ID')
    colname.remove('left_GFR')
    colname.remove('right_GFR')
    colname.remove('prediction_label_left')
    colname.remove('prediction_label_right')

    y1 = torch.tensor(df['left_GFR'].values, dtype=torch.float32)
    y2 = torch.tensor(df['right_GFR'].values, dtype=torch.float32)
    pre_1 = torch.tensor(df['prediction_label_left'].values, dtype=torch.float32)
    pre_2 = torch.tensor(df['prediction_label_right'].values, dtype=torch.float32)
    clin = torch.tensor(df[colname].values, dtype=torch.float32)
    print(f'Clinical Data Columns: {colname}, Shape: {clin.shape}')
    print(f'{y1[:5]}, {y2[:5]}')
    print(f'Preliminary Predictions: {pre_1[:5]}, {pre_2[:5]}')
    print(df.head())

    if split:
        # Split the data into training and validation sets
        df_idx = pd.read_csv('data_All/tr_val.csv')
        df_idx['ID'] = [i.replace('nfyy', '') for i in df_idx['ID']]
        df = df.merge(df_idx, on='ID', how='left')

        train_idx = df[df['tr'] == 1].index.tolist()
        val_idx = df[df['tr'] == 0].index.tolist()
        train_data_a = data_a[train_idx]
        train_data_v = data_v[train_idx]
        train_y1 = y1[train_idx]
        train_y2 = y2[train_idx]
        train_pre_1 = pre_1[train_idx]
        train_pre_2 = pre_2[train_idx]
        train_clin = clin[train_idx]
        val_data_a = data_a[val_idx]
        val_data_v = data_v[val_idx]
        val_y1 = y1[val_idx]
        val_y2 = y2[val_idx]
        val_pre_1 = pre_1[val_idx]
        val_pre_2 = pre_2[val_idx]
        val_clin = clin[val_idx]
        train_idx = torch.tensor(train_idx).reshape(-1, 1)
        val_idx = torch.tensor(val_idx).reshape(-1, 1)
        train_idx = torch.cat([train_idx, train_idx], dim=1)
        val_idx = torch.cat([val_idx, val_idx], dim=1)
        print(f'Clinical Data Shape: {train_clin.shape}, {val_clin.shape}')
        # Create DataLoader for training and validation sets
        if give_id:
            train_dataset = torch.utils.data.TensorDataset(train_data_a, train_data_v, train_y1, train_y2, train_idx, train_clin, train_pre_1, train_pre_2)
            val_dataset = torch.utils.data.TensorDataset(val_data_a, val_data_v, val_y1, val_y2, val_idx, val_clin, val_pre_1, val_pre_2)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            train_dataset = torch.utils.data.TensorDataset(train_data_a, train_data_v, train_y1, train_y2, train_idx, train_clin, train_pre_1, train_pre_2)
            val_dataset = torch.utils.data.TensorDataset(val_data_a, val_data_v, val_y1, val_y2, val_idx, val_clin, val_pre_1, val_pre_2)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    idx = torch.arange(data_a.shape[0]).reshape(-1, 1)
    idx = torch.cat([idx, idx], dim=1)
    dataset = torch.utils.data.TensorDataset(data_a, data_v, y1, y2, idx, clin, pre_1, pre_2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    setup_seed(42)
    # train_loader, val_loader = create_dataloader('nfyy', batch_size=8, shuffle=True, use_aligned=False, split=True, stage='arterial', side='All')
    print()
    test_loader = create_dataloader('gzyy', batch_size=8, shuffle=False, use_aligned=False, side='All', split=False, stage='arterial')