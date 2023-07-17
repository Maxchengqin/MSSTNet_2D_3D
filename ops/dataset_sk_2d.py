import torch.utils.data as data
import torch.nn as nn
import os
import os.path
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader

ntu_skeleton_bone_pairs = ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16))
index_start_point = [x[0] for x in ntu_skeleton_bone_pairs]
index_end_point = [x[1] for x in ntu_skeleton_bone_pairs]

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

    def get_bone(self, sk_data):
        M, C, T, V = sk_data.shape
        bone_data = np.zeros([M, C, T, len(ntu_skeleton_bone_pairs)])
        # xy = sk_data[:, 0:2, :, index_start_point] - sk_data[:, 0:2, :, index_end_point]
        # print(xy.shape) #(1, 2, 36, 16)
        bone_data[:, 0:2, :, :] = sk_data[:, 0:2, :, index_start_point] - sk_data[:, 0:2, :, index_end_point]
        conf = sk_data[:, 2, :, index_start_point] + sk_data[:, 2, :, index_end_point]#第三位是置信度，所以用加法。
        # print(conf.shape)#(16, 1, 27) v 到前面去了
        conf = np.transpose(conf, (1, 2, 0))  #
        bone_data[:, 2, :, :] = conf
        return bone_data

    def get_motion(self, sk_data):
        return sk_data[:, :, 1:, :] - sk_data[:, :, :-1, :]

    def trans(self, sk_data):
        if 'bone' in self.modality:
            pool2d = nn.AdaptiveAvgPool2d((200, 16))
        else:
            pool2d = nn.AdaptiveAvgPool2d((200, 17))
        tensor = torch.tensor(np.array(sk_data, dtype=np.float32))
        process_data = pool2d(tensor)
        return process_data

    def __getitem__(self, index):
        key = self.all_keys[index] #['S001C001P003R001A001', 'S001C001P003R001A002', 'S001C001P003R001A003', 'S001C001P003R001A004', 'S001C001P003R001A005',
        sk_data = self.all_data[key]##(1, 118, 17, 3) 有置信度
        M, T, V, C = sk_data.shape
        if M==1:
            sk_data = np.concatenate((sk_data, np.zeros_like(sk_data)), 0)

        sk_data = np.transpose(sk_data, (0, 3, 1, 2))#Mx3XTx17,为了适应后续的变化，
        if 'bone' in self.modality:
            sk_data = self.get_bone(sk_data)
        if 'motion' in self.modality:
            sk_data = self.get_motion(sk_data)
        sk_data = self.trans(sk_data)

        label = int(key[-3:]) - 1

        return sk_data, label

    def __len__(self):
        return len(self.all_keys)

if __name__ == '__main__':

    datapath = r'G:\test_data_for_final\ntu_2dsk_dict\ntu60-xsub'
    train_dataloader = DataLoader(MSSTdata(datapath, test_mode=True, modality='joint_motion'), batch_size=4, shuffle=False, num_workers=1)
    print(len(train_dataloader))

    for step, (process_data, label) in enumerate(train_dataloader):
        print(process_data.size(), label)
        if step>5:
            break

