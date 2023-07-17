import torch.utils.data as data
import torch.nn as nn
import os
import os.path
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader

ntu_skeleton_bone_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
)
index_start_point = [x[0] - 1 for x in ntu_skeleton_bone_pairs]
index_end_point = [x[1] - 1 for x in ntu_skeleton_bone_pairs]

class MSSTdata(data.Dataset):
    def __init__(self, dataroot, modality='joint', test_mode=False,):
        datapath = os.path.join(dataroot, 'train_dict.pkl')
        if test_mode:
            datapath = os.path.join(dataroot, 'val_dict.pkl')
        self.modality = modality
        self.test_mode = test_mode
        self.all_data = pickle.load(open(datapath, 'rb'))
        self.all_keys = list(self.all_data.keys())
        print('样本数量：', len(self.all_keys))
        # print(self.all_keys)

    def get_bone(self, sk_data):
        M, C, T, V = sk_data.shape
        bone_data = np.zeros([M, C, T, len(ntu_skeleton_bone_pairs)])
        # xy = sk_data[:, 0:2, :, index_start_point] - sk_data[:, 0:2, :, index_end_point]
        # print(xy.shape) #(1, 2, 36, 16)
        bone_data[:, 0:2, :, :] = sk_data[:, 0:2, :, index_start_point] - sk_data[:, 0:2, :, index_end_point]
        conf = sk_data[:, 2, :, index_start_point] + sk_data[:, 2, :, index_end_point]
        # print(conf.shape)#(16, 1, 27) v 到前面去了
        conf = np.transpose(conf, (1, 2, 0))  #
        bone_data[:, 2, :, :] = conf
        return bone_data

    def get_motion(self, sk_data):
        return sk_data[:, :, 1:, :] - sk_data[:, :, :-1, :]

    def trans(self, sk_data):
        if 'bone' in self.modality:
            pool2d = nn.AdaptiveAvgPool2d((100, 25))
        else:
            pool2d = nn.AdaptiveAvgPool2d((100, 25))
        tensor = torch.tensor(np.array(sk_data, dtype=np.float32))
        process_data = pool2d(tensor)
        return process_data

    def __getitem__(self, index):
        key = self.all_keys[index] #['S017C003P007R001A025.skeleton',
        sk_data = self.all_data[key]##(3, 117, 25, 2)
        M, T, V, C = sk_data.shape
        if M==1:
            sk_data = np.concatenate((sk_data, np.zeros_like(sk_data)), 0)

        sk_data = np.transpose(sk_data, (3, 0, 1, 2))#Mx3xTx25,为了适应后续的变化，
        if 'bone' in self.modality:
            sk_data = self.get_bone(sk_data)
        if 'motion' in self.modality:
            sk_data = self.get_motion(sk_data)
        sk_data = self.trans(sk_data)

        label = int(key.split('.skeleton')[0][-3:]) - 1

        return sk_data, label

    def __len__(self):
        return len(self.all_keys)

if __name__ == '__main__':

    datapath = r'G:\test_data_for_final\ntu_3dsk_dict\ntu60-xsub'
    train_dataloader = DataLoader(MSSTdata(datapath, test_mode=True, modality='bone_motion'), batch_size=4, shuffle=False, num_workers=1)
    print(len(train_dataloader))

    for step, (process_data, label) in enumerate(train_dataloader):
        print(process_data.size(), label)
        if step>5:
            break

