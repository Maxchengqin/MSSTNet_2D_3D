from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.init import normal, constant
import os
import sys
import torchvision.models.resnet as resnet


class msstMoudel(nn.Module):
    def __init__(self, inchannel, outchannel1, outchannel3, outchannel5, outchannel7, outchannel11, stride=(1, 1)):
        super(msstMoudel, self).__init__()
        inplace = True

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel1, kernel_size=(1, 1), stride=stride,
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel1, affine=True),
            nn.ReLU(inplace))
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel3, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel3, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel3, out_channels=outchannel3, kernel_size=(3, 3), stride=stride,
                      padding=(1, 1)),
            nn.BatchNorm2d(outchannel3, affine=True),
            nn.ReLU(inplace))

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel5, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel5, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel5, out_channels=outchannel5, kernel_size=(5, 1), stride=stride,
                      padding=(2, 0)),
            nn.BatchNorm2d(outchannel5, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel5, out_channels=outchannel5, kernel_size=(1, 5), stride=(1, 1),
                      padding=(0, 2)),
            nn.BatchNorm2d(outchannel5, affine=True),
            nn.ReLU(inplace))

        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel7, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel7, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel7, out_channels=outchannel7, kernel_size=(7, 1), stride=stride,
                      padding=(3, 0)),
            nn.BatchNorm2d(outchannel7, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel7, out_channels=outchannel7, kernel_size=(1, 7), stride=(1, 1),
                      padding=(0, 3)),
            nn.BatchNorm2d(outchannel7, affine=True),
            nn.ReLU(inplace))

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel11, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel11, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel11, out_channels=outchannel11, kernel_size=(11, 1), stride=stride,
                      padding=(5, 0)),
            nn.BatchNorm2d(outchannel11, affine=True),
            nn.ReLU(inplace))
            # nn.Conv2d(in_channels=outchannel11, out_channels=outchannel11, kernel_size=(1, 11), stride=stride,
            #           padding=(0, 5)),
            # nn.BatchNorm2d(outchannel11, affine=True),
            # nn.ReLU(inplace))

    def forward(self, input):
        output1 = self.conv1x1(input)
        # print('o1',output1.size())
        output3 = self.conv3x3(input)
        # print('o3', output3.size())
        output5 = self.conv5x5(input)
        # print('o5', output5.size())
        output7 = self.conv7x7(input)

        # print('o7', output7.size())
        output11 = self.conv11(input)
        output = torch.cat([output1, output3, output5, output7, output11], 1)
        return output

class MSSTNet(nn.Module):

    def __init__(self, num_class=1000, dropout=0.8):
        super(MSSTNet, self).__init__()
        inplace = True
        self.dropout = dropout

        self.avgpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), ceil_mode=False, count_include_pad=True),
            nn.BatchNorm2d(416, affine=True),
            nn.ReLU(inplace))
        self.avgpool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), ceil_mode=False, count_include_pad=True),
            nn.BatchNorm2d(672, affine=True),
            nn.ReLU(inplace))

        self.m1 = msstMoudel(inchannel=3, outchannel1=32, outchannel3=48, outchannel5=48, outchannel7=48,
                              outchannel11=48, stride=(2, 1))
        self.m2 = msstMoudel(inchannel=224, outchannel1=32, outchannel3=64, outchannel5=64, outchannel7=64,
                              outchannel11=64, stride=(2, 1))
        self.m3 = msstMoudel(inchannel=288, outchannel1=32, outchannel3=96, outchannel5=96, outchannel7=96,
                              outchannel11=96, stride=(2, 1))
        self.m4 = msstMoudel(inchannel=416, outchannel1=128, outchannel3=128, outchannel5=128, outchannel7=128,
                              outchannel11=128, stride=(2, 1))
        self.m5 = msstMoudel(inchannel=640, outchannel1=32, outchannel3=160, outchannel5=160, outchannel7=160,
                              outchannel11=160, stride=(2, 1))
        self.m6 = msstMoudel(inchannel=672, outchannel1=192, outchannel3=192, outchannel5=192, outchannel7=192,
                              outchannel11=192, stride=(1, 1))
        self.m7 = msstMoudel(inchannel=960, outchannel1=256, outchannel3=256, outchannel5=256, outchannel7=256,
                              outchannel11=256, stride=(1, 1))
        self.dropout = nn.Dropout(p=self.dropout)
        self.last_linear = nn.Linear(1280, num_class)

    def features(self, input):
        m1out = self.m1(input)
        m2out = self.m2(m1out)
        m3out = self.m3(m2out)
        m3poolout = self.avgpool(m3out)
        m4out = self.m4(m3poolout)
        m5out = self.m5(m4out)
        m5poolout = self.avgpool2(m5out)
        m6out = self.m6(m5poolout)
        m7out = self.m7(m6out)
        # print(m7out.size())
        return m7out

    def logits(self, features):
        fea_base = features
        adaptiveAvgPoolWidth = (features.shape[2], features.shape[3])
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x_base = x.view(x.size(0), -1)
        x = self.dropout(x_base)
        x = self.last_linear(x)
        return fea_base, x

    def forward(self, input):
        x = self.features(input)
        x_base, x = self.logits(x)
        return x_base, x


class Net(nn.Module):
    def __init__(self, num_class=60, dropout=0.8):
        super(Net, self).__init__()
        self.dropout = dropout
        self.num_class = num_class
        self.model = MSSTNet(num_class=self.num_class, dropout=self.dropout)
    ##########################原处理方式#############################################
    # def forward(self, input):
    #     # print(self.model)
    #     output = self.model(input)
    #     # print(output.size())
    #     output = self.new_fc(output)
    #     # print('UYFUFKHFHGFYFGJHVJVHGC<JV<JCJGC<HV', output.size())
    #     return output

    #########################原处理方式#############################################

#####################################双图片改bachsize################################
    def forward(self, input):
        B, O, C, H, W = input.size()
        input = input.view(B*O,C,H,W)
        out_base, output = self.model(input)
        output = output.view(B,O,-1).mean(dim=1)
        out_base = out_base.view((B, O) + out_base.size()[1:]).mean(dim=1)
        return out_base, output
####################################################################################
if __name__=='__main__':
    from ptflops import get_model_complexity_info

    model = Net(num_class=60, dropout=0.8)
    input_sk = torch.rand([4, 2, 3, 200, 17])
    out = model(input_sk)
    print(out[1].size())
    print(out[0].size())
    flops, params = get_model_complexity_info(model, (2, 3, 200, 17), as_strings=True, print_per_layer_stat=False)  # as_strings=True,会用G或M为单位，反之精确到个位。
    print(flops, params)#(2, 3, 200, 17)需要2.39 GMac 9.9 M