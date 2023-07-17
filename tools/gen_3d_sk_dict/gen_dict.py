import pickle
import os
import numpy as np
data_split = ['train', 'val']
data_class = ['joint']
# root_path = '/data/cq/ntu_3dsk_pro/ntu120/xsub'
# save_root = '/data/cq/ntu_3dsk_pro_dict/ntu120-xsub'
# root_path = '/data/cq/ntu_3dsk_pro/ntu120/xset'
# save_root = '/data/cq/ntu_3dsk_pro_dict/ntu120-xset'
# root_path = '/data/cq/ntu_3dsk_pro/ntu60/xsub'
# save_root = '/data/cq/ntu_3dsk_pro_dict/ntu60-xsub'
root_path = '/data/cq/ntu_3dsk_pro/ntu60/xview'
save_root = '/data/cq/ntu_3dsk_pro_dict/ntu60-xview'
from tqdm import tqdm
def load_skeleton0(oridata):
    cut1 = 0
    cut2 = 299
    for j in range(300):
        if sum(oridata[0][j]) != 0:
            cut1 = j
            break
    for j in range(300):
        if sum(oridata[0][299 - j]) != 0 and sum(oridata[0][299 - j - 5]) != 0:
            cut2 = 299 - j + 1
            break
    # useful = oridata[:, cut1:cut2, :]
    return cut1, cut2

for split_name in data_split:
    print(split_name)
    label_file_path = os.path.join(root_path, split_name+'_label.pkl')
    labels = pickle.load(open(label_file_path, 'rb'))[0]
    data_dict = {}
    for class_name in data_class:
        print(class_name)
        data_file_path = os.path.join(root_path, split_name+'_'+class_name+'.npy')
        data = np.load(data_file_path)
        print('数据形状', data.shape)
        for i, sk_data in enumerate(tqdm(data, ncols=120)):
            sk0 = sk_data[:, :, :, 0]
            cut1, cut2 = load_skeleton0(sk0)
            data_dict[labels[i]]=sk_data[:, cut1:cut2, :, :]
        data_dict_file_path = os.path.join(save_root,  split_name+'_'+class_name+'.pkl')
        print(data_dict_file_path)
        data_dict_file = open(data_dict_file_path, 'wb')
        pickle.dump(data_dict, data_dict_file)
        print(data_dict_file_path, ' saved')

