import h5py

with h5py.File('F:\peizhun\SPAC-Deformable-Registration-main\mydata/combined_data.h5', 'r') as hf:
    for key in hf.keys():
        print(hf[key])
        print([key for key in hf[key].keys()])
