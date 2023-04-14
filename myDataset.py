from torch.utils.data import Dataset
import os
import numpy as np
import torch
import SimpleITK as itk
import scipy
import scipy.ndimage
import scipy.io
import nibabel as nib
#读取一对图像
class pair_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: str, 数据集所在路径
        transform: torch.transform，数据预处理
        """
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    '''def __getitem__(self, index):
        path_img1, path_img2 = self.data_info[index]
        np_var = 'vol_data'
        fixed_img = np.load(path_img1)[np_var]
        moving_img = np.load(path_img2)[np_var]

        if self.transform is not None:
            fixed_img = self.transform(fixed_img)
            moving_img = self.transform(moving_img)
        return fixed_img, moving_img
    '''

    def __getitem__(self, index):
        fixed_path, moving_path = self.data_info[index]

        fixed_sitk = itk.ReadImage(fixed_path)  # 读取nii或nii.gz
        fixed_img = itk.GetArrayFromImage(fixed_sitk)
        fix_img = torch.from_numpy(fixed_img).to("cuda").float()
        fix_img = fix_img.clone().detach().requires_grad_(requires_grad=False)

        moving_sitk = itk.ReadImage(moving_path)  # 添加一个channel维度
        moving_img = itk.GetArrayFromImage(moving_sitk)
        mov_img = torch.from_numpy(moving_img).to("cuda").float()
        mov_img = mov_img.clone().detach().requires_grad_(requires_grad=False)

        fix_img = fix_img.unsqueeze(0)
        mov_img = mov_img.unsqueeze(0)

        return fix_img, mov_img


    # 返回所有样本的数量
    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        # data_dir 是训练集、验证集或者测试集的路径
        for root_fix, _, files_fix in os.walk(data_dir[0]):
            img_names_fix = os.listdir(root_fix)
            img_names_fix = list(filter(lambda x: x.endswith('.nii'), img_names_fix))
            for i in range(len(img_names_fix)):
                img_name_fix = img_names_fix[i]
                path_img_fix = os.path.join(root_fix, img_name_fix)
        for root_moving, _, files_moving in os.walk(data_dir[1]):
            # 遍历类别
            img_names_moving = os.listdir(root_moving)
            img_names_moving = list(filter(lambda x: x.endswith('.nii'), img_names_moving))
            for i in range(len(img_names_moving)):
                #img_name_fix = img_names_fix[i]
                img_name_moving = img_names_moving[i]
                # 图片的绝对路径
                #path_img_fix = os.path.join(root_fix, img_name_fix)
                path_img_moving = os.path.join(root_moving, img_name_moving)
                # 保存在data_info变量中
                data_info.append((path_img_fix, path_img_moving))
        return data_info
class single_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: str, 数据集所在路径
        transform: torch.transform，数据预处理
        """
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img1 = self.data_info[index]
        np_var = 'vol_data'
        fixed_img = np.load(path_img1)[np_var]
        # fixed_img = fixed_img[np.newaxis, ...]
        if self.transform is not None:
            fixed_img = self.transform(fixed_img)
        return fixed_img

    # 返回所有样本的数量
    def __len__(self):
        # print(len(self.data_info))
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        # data_dir 是训练集、验证集或者测试集的路径
        for root_fix, _, files_fix in os.walk(data_dir):
            img_names_fix = os.listdir(root_fix)
            img_names_fix = list(filter(lambda x: x.endswith('.nii'), img_names_fix))
            for i in range(len(img_names_fix)):
                img_name_fix = img_names_fix[i]
                # 图片的绝对路径
                path_img_fix = os.path.join(root_fix, img_name_fix)
                data_info.append(path_img_fix)
        return data_info
class dice_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: str, 数据集所在路径
        transform: torch.transform，数据预处理
        """
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        fixed_path, moving_path = self.data_info[index]

        fixed_sitk = itk.ReadImage(fixed_path)
        fixed_img = itk.GetArrayFromImage(fixed_sitk)
        fix_img = torch.from_numpy(fixed_img).to("cuda").float()
        fix_img = fix_img.clone().detach().requires_grad_(requires_grad=False)

        moving_sitk = itk.ReadImage(moving_path)     #添加一个channel维度
        moving_img = itk.GetArrayFromImage(moving_sitk)
        mov_img = torch.from_numpy(moving_img).to("cuda").float()
        mov_img = mov_img.clone().detach().requires_grad_(requires_grad=False)

        fix_img = fix_img.unsqueeze(0)
        #mov_img = mov_img.unsqueeze(0)    #如果是test,则把此行注释
        return fix_img, mov_img
    # 返回所有样本的数量
    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        # data_dir 是训练集、验证集或者测试集的路径
        for root_fix, _, files_fix in os.walk(data_dir[0]):
            for root_moving, _, files_moving in os.walk(data_dir[1]):
            # 遍历类别
                img_names_fix = os.listdir(root_fix)
                img_names_fix = list(filter(lambda x: x.endswith('.nii'), img_names_fix))
                img_names_moving = os.listdir(root_moving)
                img_names_moving = list(filter(lambda x: x.endswith('.nii'), img_names_moving))
                for i in range(len(img_names_fix)):
                   img_name_fix = img_names_fix[i]
                   img_name_moving = img_names_moving[i]
                   # 图片的绝对路径
                   path_img_fix = os.path.join(root_fix, img_name_fix)
                   path_img_moving = os.path.join(root_moving, img_name_moving)
                   # 保存在data_info变量中
                   data_info.append((path_img_fix, path_img_moving))
        return data_info
