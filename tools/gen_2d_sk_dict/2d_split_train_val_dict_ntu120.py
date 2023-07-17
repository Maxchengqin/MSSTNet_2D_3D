import pickle
import tqdm
import numpy as np
ori_data = pickle.load(open('ntu120_hrnet.pkl', 'rb'))
print(ori_data["split"].keys())
xsub_train_list, xsub_val_list, xset_train_list, xset_val_list = ori_data["split"].values()
print(len(xsub_train_list), len(xsub_val_list), len(xset_train_list), len(xset_val_list))


xsub_train_dict = {}
xsub_val_dict = {}
xset_train_dict = {}
xset_val_dict = {}
count = 0
# for i, sk in tqdm(enumerate(ori_data["annotations"]), ncols=120):
for sk in ori_data["annotations"]:
    count += 1
    print(count)
    if sk["frame_dir"] in xsub_train_list:
        xsub_train_dict[sk["frame_dir"]] = np.concatenate((sk["keypoint"], sk['keypoint_score'][:,:,:,None]), -1)
    if sk["frame_dir"] in xsub_val_list:
        xsub_val_dict[sk["frame_dir"]] = np.concatenate((sk["keypoint"], sk['keypoint_score'][:,:,:,None]), -1)
    if sk["frame_dir"] in xset_train_list:
        xset_train_dict[sk["frame_dir"]] = np.concatenate((sk["keypoint"], sk['keypoint_score'][:,:,:,None]), -1)
    if sk["frame_dir"] in xset_val_list:
        xset_val_dict[sk["frame_dir"]] = np.concatenate((sk["keypoint"], sk['keypoint_score'][:,:,:,None]), -1)
print(len(xsub_train_dict.keys()), len(xsub_val_dict.keys()), len(xset_train_dict.keys()), len(xset_val_dict.keys()))#63026 50919 54468 59477, 共 113945

xsub_train_dict_file = open("ntu120-xsub/train_dict.pkl", 'wb')
xsub_val_dict_file = open("ntu120-xsub/val_dict.pkl", 'wb')
xset_train_dict_file = open("ntu120-xset/train_dict.pkl", 'wb')
xset_val_dict_file = open("ntu120-xset/val_dict.pkl", 'wb')

pickle.dump(xsub_train_dict, xsub_train_dict_file)
xsub_train_dict_file.close()
pickle.dump(xsub_val_dict, xsub_val_dict_file)
xsub_val_dict_file.close()
pickle.dump(xset_train_dict, xset_train_dict_file)
xset_train_dict_file.close()
pickle.dump(xset_val_dict, xset_val_dict_file)
xset_val_dict_file.close()



