from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from ops import msst_mini
from torch.nn.init import normal_, constant_
# weight_path = 'ops/imagenet_msst_mini_200x25.pth.tar'
weight_path = 'imagenet_msst_mini_200x25.pth.tar'#本地

class Net(nn.Module):
    def __init__(self, num_class=60, dropout=0.8):
        super(Net, self).__init__()
        self.dropout = dropout
        self.model = msst_mini.Net(num_class=1000, dropout=0.9).to(device)
        checkpoint = torch.load(weight_path)
        print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        self.model.load_state_dict(base_dict)
        self.model.model.last_linear = nn.Linear(640, num_class)#根据数据集对应的类别调整最后一层FC
        std = 0.001
        normal_(self.model.model.last_linear.weight, 0, std)
        constant_(self.model.model.last_linear.bias, 0)

    def forward(self, input):
        output = self.model(input)
        return output

if __name__ == '__main__':
    net = Net(num_class=60).cuda()
    input = torch.rand((8, 2, 3, 100, 17)).cuda()
    print(net)
    out_put = net(input)
    print(out_put.size())
