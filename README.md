# MSSTNet_2D_3D
本代码是MSSTNet的实现方法，对应文章是“Multi-scale spatial–temporal convolutional neural network for skeleton-based action recognition”，文章链接是https://link.springer.com/article/10.1007/s10044-023-01156-w。文章中NTU60和NTU120数据集只使用了官方提供的3D骨架进行了实验，这里增加了基于2D骨架的实验。
2D骨架源自“Revisiting Skeleton-based Action Recognition”文章所使用的2D骨架。下载链接是https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md。

## 数据准备工作
训练之前先用tool里面的数据处理工具将数据整理成字典形式
对于2D骨架，下载好‘ntu60_hrnet.pkl’和‘ntu120_hrnet.pkl’后，在‘2d_split_train_val_dict_ntu60.py’和‘2d_split_train_val_dict_ntu120.py’代码里改成自己的路径，然后运行,程序会将数据分成两种测试协议下的训练集和测试集，并保存成字典格式。
对于3D骨架，以NTU60为例，下载好NTU官方的数据后，先运行‘ntu_gendata_pro.py’，生成.npy格式的数据，然后运行‘gen_dict.py’生成字典格式的数据。注意修改代码里面的路径！！！

## 训练
train_and_test_2d.txt 里面是训练命令。

## 3D骨架实验结果

NTU60-xusb：89.6%，NTU60-xview：95.3。NTU120-xsub：85.3，NTU120-xset：86.0。计算量：13.9GFLOPs。

## 2D骨架实验结果

NTU60-xusb：92.6%，NTU60-xview：97.8。NTU120-xsub：87.4，NTU120-xset：88.3。计算量：9.1GFLOPs。


