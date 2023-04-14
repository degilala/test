import os
import numpy as np
import SimpleITK as itk
import image_processing as ip
import scipy.io as io
import pandas as pd

def NpztoHdr_2D(data_dir,save_dir):
    if not os.path.exists(data_dir):
        print('输入数据的文件夹不存在！')
        return
    for root, dirs, files in os.walk(save_dir):
        for name in files:
            os.remove(os.path.join(root, name))
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.npz'):
                print('开始将' + filename + '转换为hdr格式')
                image_path = os.path.join(root, filename)
                data_vol = np.load(image_path)
                data = data_vol['vol_data']
                data = data.astype(np.float32)
                img = itk.GetImageFromArray(data)
                save_path = save_dir + '/' + filename.split('.')[0] + '.hdr'
                itk.WriteImage(img, save_path)

def DcmtoNpz_2D(data_dir,save_dir):
    if not os.path.exists(data_dir):
        print('输入数据的文件夹不存在！')
        return
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.dcm'):
                print('开始将' + filename + '转换为npz格式')
                image_path = os.path.join(root, filename)
                image = itk.ReadImage(image_path)
                data = itk.GetArrayFromImage(image)
                data = np.reshape(data, (512, 512))
                save_path = save_dir + '/' + filename.split('.')[0] + '.npz'
                np.savez(save_path, vol_data=data)

def NpztoNii(data_dir,save_dir):
    if not os.path.exists(data_dir):
        print('输入数据的文件夹不存在！')
        return
    for home, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.npz'):
                print('开始将' + filename + '转换为nii图片')
                image_path = os.path.join(home, filename)
                data = np.load(image_path)
                data_vol = data['vol_data']
                current_save_path = save_dir + '\\' + filename.split('.')[0] + '.nii'
                fix_image_sitk = ip.array_to_sitk_2D(data_vol, spacing=[1.5, 1.5], origin=[0, 0])                   #2D转换
                # fix_image_sitk = ip.array_to_sitk_3D(data_vol, spacing=[1.5, 1.5, 1.5, 1.5], origin=[0, 0, 0, 0]) #3D转换
                itk.WriteImage(fix_image_sitk, current_save_path)

def NpztoMat(data_dir,save_dir):
    if not os.path.exists(data_dir):
        print('输入数据的文件夹不存在！')
        return
    for home, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.npz'):
                print('开始将' + filename + '转换为mat图片')
                image_path = os.path.join(home, filename)
                data = np.load(image_path)
                data_vol = data['vol_data']
                save_path = save_dir + '\\' + filename.split('.')[0] + '.mat'
                io.savemat(save_path, {'data': data_vol})

def MattoNpz(data_dir,save_dir):
    if not os.path.exists(data_dir):
        print('输入数据的文件夹不存在！')
        return
    for home, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.mat'):
                print('开始将' + filename + '转换为npz图片')
                image_path = os.path.join(home, filename)
                data = io.loadmat(image_path)['data']
                save_path = save_dir + '\\' + filename.split('.')[0] + '.mat'
                np.savez(save_path, vol_data=data)

def NiitoHdr(niidir_moved_img, niidir_fix_img, hdrdir_moved_img, hdrdir_fix_img, hdrdir_diff2d_after, fix_range_csv, mov_range_csv):
    range_fix = pd.read_csv(fix_range_csv)
    range_mov = pd.read_csv(mov_range_csv)

    if not os.path.exists(niidir_moved_img):
        print('输入数据的文件夹不存在！')
        return
    if not os.path.exists(niidir_fix_img):
        print('输入数据的文件夹不存在！')
        return
    moved_img_list = []
    fix_img_list = []
    for home, dirs, files in os.walk(niidir_moved_img):
        for filename in files:
            if filename.endswith('.nii'):
                moved_img_list.append(os.path.join(home, filename))
    for home, dirs, files in os.walk(niidir_fix_img):
        for filename in files:
            if filename.endswith('.nii'):
                fix_img_list.append(os.path.join(home, filename))

    range_fix_list = []
    for i in range(0, len(range_fix)):
        name = os.path.join(niidir_fix_img, str(range_fix.iloc[i, 0]).split('.')[0] + '.nii')
        if name in fix_img_list:
            min_gray = int(range_fix.iloc[i, 1])
            max_gray = int(range_fix.iloc[i, 2])
            Fixrange = max_gray - min_gray
            range_fix_list.append([Fixrange, min_gray, name])
    range_mov_list = []
    for i in range(0, len(range_mov)):
        name = os.path.join(niidir_moved_img, str(range_mov.iloc[i, 0]).split('.')[0] + '.nii')
        if name in moved_img_list:
            min_gray = int(range_mov.iloc[i, 1])
            max_gray = int(range_mov.iloc[i, 2])
            Movrange = max_gray - min_gray
            range_mov_list.append([Movrange, min_gray, name])

    for i in range(len(moved_img_list)):
        moved_img = itk.GetArrayFromImage(itk.ReadImage(moved_img_list[i]))
        fix_img = itk.GetArrayFromImage(itk.ReadImage(fix_img_list[i]))
        moved_img = moved_img * range_mov_list[i][0] + range_mov_list[i][1]
        fix_img = fix_img * range_fix_list[i][0] + range_fix_list[i][1]
        diff_img = moved_img - fix_img
        np.savez(niidir_moved_img + '/' + moved_img_list[i].split('\\')[-1].split('.')[0] + '.npz', vol_data=moved_img)
        np.savez(niidir_fix_img + '/' + fix_img_list[i].split('\\')[-1].split('.')[0] + '.npz', vol_data=fix_img)
        itk.WriteImage(itk.GetImageFromArray(moved_img), hdrdir_moved_img + '/' + moved_img_list[i].split('\\')[-1].split('.')[0] + '.hdr')
        itk.WriteImage(itk.GetImageFromArray(fix_img), hdrdir_fix_img + '/' + fix_img_list[i].split('\\')[-1].split('.')[0] + '.hdr')
        itk.WriteImage(itk.GetImageFromArray(diff_img), hdrdir_diff2d_after + '/' + moved_img_list[i].split('\\')[-1].split('.')[0] + '.hdr')


def Nii2Npz(nii_dir, npz_dir):
    img_list = []
    for home, dirs, files in os.walk(nii_dir):
        for filename in files:
            if filename.endswith('.nii'):
                img_list.append(os.path.join(home, filename))
    for i in range(len(img_list)):
        img = itk.GetArrayFromImage(itk.ReadImage(img_list[i]))
        np.savez(npz_dir + '/' + img_list[i].split('\\')[-1].split('.')[0] + '.npz', vol_data=img)


def Npz2diffHdr(fix_dir, mov_dir, diff_dir):
    fix_img_list = []
    mov_img_list = []
    for home, dirs, files in os.walk(fix_dir):
        for filename in files:
            if filename.endswith('.npz'):
                fix_img_list.append(os.path.join(home, filename))
    for home, dirs, files in os.walk(mov_dir):
        for filename in files:
            if filename.endswith('.npz'):
                mov_img_list.append(os.path.join(home, filename))
    for i in range(len(fix_img_list)):
        fix = np.load(fix_img_list[i])['vol_data']
        mov = np.load(mov_img_list[i])['vol_data']
        img = fix - mov
        img = itk.GetImageFromArray(img)
        save_path = diff_dir + '/' + fix_img_list[i].split('\\')[-1].split('.')[0] + '.hdr'
        itk.WriteImage(img, save_path)

