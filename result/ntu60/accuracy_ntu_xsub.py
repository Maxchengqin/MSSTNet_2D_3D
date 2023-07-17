import numpy as np
from sklearn.metrics import confusion_matrix
video_labels = np.load('ntu60_xsub_2dsk_msst_joint.npz', allow_pickle=True)['labels']#

joint = np.load('ntu60_xsub_2dsk_msst_joint.npz', allow_pickle=True)['scores']#
bone = np.load('ntu60_xsub_2dsk_msst_bone.npz', allow_pickle=True)['scores']#
joint_motion = np.load('ntu60_xsub_2dsk_msst_joint_motion.npz', allow_pickle=True)['scores']#
bone_motion = np.load('ntu60_xsub_2dsk_msst_joint_motion.npz', allow_pickle=True)['scores']#

video_pred = []
for i in range(len(joint)):
    pre = joint[i] * 0.6 + bone[i] * 0.6 + joint_motion[i] * 0.4 + bone_motion[i] * 0.4
    video_pred.extend([np.argmax(pre)])

cf = confusion_matrix(video_labels, video_pred).astype(float)
print('cf_shape', cf.shape)
cls_cnt = cf.sum(axis=1)  # 得到是每一类各自总评估次数.
cls_hit = np.diag(cf)  # 每一类总的评估对的次数.
cls_acc = cls_hit / cls_cnt
# print(video_labels)
print('各类正确率：\n', cls_acc)
print('各类平均精度 {:.04f}%，总数累加正确率{:.04f}%'.format(np.mean(cls_acc) * 100, (np.sum(cls_hit)) / (np.sum(cls_cnt)) * 100))



