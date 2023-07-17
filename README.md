# MSSTNet_2D_3D
本代码是MSSTNet的实现方法，对应文章是“Multi-scale spatial–temporal convolutional neural network for skeleton-based action recognition”，文章链接是https://link.springer.com/article/10.1007/s10044-023-01156-w。文章中NTU60和NTU120数据集只使用了官方提供的3D骨架进行了实验，这里增加了基于2D骨架的实验。
2D骨架源自“Revisiting Skeleton-based Action Recognition”文章所使用的2D骨架。下载链接是https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md。
##3D骨架实验结果
NTU60-xusb：89.6%，NTU60-xview：95.3。NTU120-xsub：85.3，NTU120-xset：86.0。计算量：13.9GFLOPs。
##2D骨架实验结果
NTU60-xusb：92.6%，NTU60-xview：97.8。NTU120-xsub：87.4，NTU120-xset：88.3。计算量：9.1GFLOPs。


