import os
import SimpleITK as sitk
import numpy as np
import h5py

folder_names = ['F:\peizhun\SPAC-Deformable-Registration-main\mydata\mov_128', 'F:\peizhun\SPAC-Deformable-Registration-main\mydata\CSF_128',
                'F:\peizhun\SPAC-Deformable-Registration-main\mydata\GM_128', 'F:\peizhun\SPAC-Deformable-Registration-main\mydata\WM_128']
keys = ['moving', 'moving_seg_CSF', 'moving_seg_GM', 'moving_seg_WM']
data_dict = {}

# 遍历每个文件夹并读取其中所有nii文件
import numpy as np
import h5py
import SimpleITK as sitk

def load_data(folders):
    data={}
    for i, folder_name in enumerate(folders):
        imgs = []
        folder_path = os.path.join(os.getcwd(), folder_name)
        file_names=os.listdir(folder_path)
        file_names.sort()
        for file_name in file_names:
            if file_name.endswith('.nii'):
                file_path = os.path.join(folder_path, file_name)
                img = sitk.ReadImage(file_path)
                imgs.append(sitk.GetArrayFromImage(img))
        data[folder_name]=imgs

    return data
data = load_data(folder_names)
combined_data=[]
for i in range(310):
    combined_data.append({
        "moving": data["F:\peizhun\SPAC-Deformable-Registration-main\mydata\mov_128"][i],
        "moving_seg_pve0": data["F:\peizhun\SPAC-Deformable-Registration-main\mydata\CSF_128"][i],
        "moving_seg_pve1": data["F:\peizhun\SPAC-Deformable-Registration-main\mydata\GM_128"][i],
        "moving_seg_pve2": data["F:\peizhun\SPAC-Deformable-Registration-main\mydata\WM_128"][i]
    })

# Save combined data to .h5 file
with h5py.File("F:\peizhun\SPAC-Deformable-Registration-main\mydata/combined_data.h5", "w") as f:
    for i, combined_image in enumerate(combined_data):
        key = f"image_{i}"
        for folder, image in combined_image.items():
            f.create_dataset(f"{key}/{folder}", data=image)
