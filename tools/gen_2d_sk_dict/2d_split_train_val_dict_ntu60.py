import pickle
import numpy as np

ori_data = pickle.load(open('ntu60_hrnet.pkl', 'rb'))
print(ori_data["split"].keys())
xsub_train_list, xsub_val_list, xview_train_list, xview_val_list = ori_data["split"].values()
print(len(xsub_train_list), len(xsub_val_list), len(xview_train_list), len(xview_val_list))


xsub_train_dict = {}
xsub_val_dict = {}
xview_train_dict = {}
xview_val_dict = {}
count = 0
for sk in ori_data["annotations"]:
    count += 1
    print(count)
    if sk["frame_dir"] in xsub_train_list:
        xsub_train_dict[sk["frame_dir"]] = np.concatenate((sk["keypoint"], sk['keypoint_score'][:,:,:,None]), -1)
    if sk["frame_dir"] in xsub_val_list:
        xsub_val_dict[sk["frame_dir"]] = np.concatenate((sk["keypoint"], sk['keypoint_score'][:,:,:,None]), -1)
    if sk["frame_dir"] in xview_train_list:
        xview_train_dict[sk["frame_dir"]] = np.concatenate((sk["keypoint"], sk['keypoint_score'][:,:,:,None]), -1)
    if sk["frame_dir"] in xview_val_list:
        xview_val_dict[sk["frame_dir"]] = np.concatenate((sk["keypoint"], sk['keypoint_score'][:,:,:,None]), -1)
print(len(xsub_train_dict.keys()), len(xsub_val_dict.keys()), len(xview_train_dict.keys()), len(xview_val_dict.keys()))#  40091 16487 37646 18932 共56578,

xsub_train_dict_file = open("ntu60-xsub/train_dict.pkl", 'wb')
xsub_val_dict_file = open("ntu60-xsub/val_dict.pkl", 'wb')
xview_train_dict_file = open("ntu60-xview/train_dict.pkl", 'wb')
xview_val_dict_file = open("ntu60-xview/val_dict.pkl", 'wb')

pickle.dump(xsub_train_dict, xsub_train_dict_file)
xsub_train_dict_file.close()
pickle.dump(xsub_val_dict, xsub_val_dict_file)
xsub_val_dict_file.close()
pickle.dump(xview_train_dict, xview_train_dict_file)
xview_train_dict_file.close()
pickle.dump(xview_val_dict, xview_val_dict_file)
xview_val_dict_file.close()



