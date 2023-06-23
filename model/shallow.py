import numpy as np
import torch
from torch import nn
from torch.nn import init
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.util import np_to_var


class ShallowFBCSPNet(nn.Module):

    def __init__(self, in_chans, n_classes, input_time_length, final_conv_length='auto', args=None):
        super(ShallowFBCSPNet, self).__init__()
        self.final_conv_length = final_conv_length
        self.in_chans = in_chans
        self.input_time_length = input_time_length
        self.n_classes = n_classes
        self.batch_norm = True

        self.features = nn.Sequential()
        self.features.add_module('dimshuffle', Expression(_transpose_time_to_spat))
        self.features.add_module('conv_time', nn.Conv2d(1, 40, (25, 1), stride=(1, 1), ))
        self.features.add_module('conv_spat', nn.Conv2d(40, 40, (1, self.in_chans), stride=1, bias=not self.batch_norm))
        n_filters_conv = 40

        self.features.add_module('bnorm', nn.BatchNorm2d(n_filters_conv, momentum=0.1, affine=True), )

        self.features.add_module('conv_nonlin', Expression(square))
        self.features.add_module('pool', nn.AvgPool2d(kernel_size=(75, 1), stride=(15, 1)))
        self.features.add_module('pool_nonlin', Expression(safe_log))
        # self.features.add_module('drop', nn.Dropout(p=0.2))

        if self.final_conv_length == 'auto':
            out = self.features(np_to_var(np.ones((1, self.in_chans, self.input_time_length, 1), dtype=np.float32)))
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time

        self.classifier_1 = nn.Sequential()
        self.classifier_1.add_module('fc1',
                                     nn.Linear(n_filters_conv * self.final_conv_length, self.n_classes, bias=True))
        # self.classifier_1.add_module('fc2', nn.Linear(200, 2, bias=True))

        # self.classifier_1.add_module("softmax", nn.LogSoftmax(dim=1))
        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(self.features.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        init.constant_(self.features.conv_time.bias, 0)
        init.xavier_uniform_(self.features.conv_spat.weight, gain=1)
        if not self.batch_norm:
            init.constant_(self.features.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.features.bnorm.weight, 1)
            init.constant_(self.features.bnorm.bias, 0)

        init.xavier_uniform_(self.classifier_1.fc1.weight, gain=1)
        init.constant_(self.classifier_1.fc1.bias, 0)
        # init.xavier_uniform_(self.classifier_1.fc2.weight, gain=1)
        # init.constant_(self.classifier_1.fc2.bias, 0)

        # con = nn.Conv2d(1, 40, (25, 1), stride=(1, 1), )
        # register_buffer:是在内存中定一个常量，同时，模型保存和加载的时候可以写入和读出，下面的应该是可以加,requires_grad=False
        self.register_buffer('pre_features', torch.zeros(args.batch_size, n_filters_conv * self.final_conv_length))
        self.register_buffer('pre_weight1', torch.ones(args.batch_size, 1))
        if args.n_levels > 1:
            self.register_buffer('pre_features_2',
                                 torch.zeros(args.batch_size, n_filters_conv * self.final_conv_length))
            self.register_buffer('pre_weight1_2', torch.ones(args.batch_size, 1))
        if args.n_levels > 2:
            self.register_buffer('pre_features_3',
                                 torch.zeros(args.batch_size, n_filters_conv * self.final_conv_length))
            self.register_buffer('pre_weight1_3', torch.ones(args.batch_size, 1))
        if args.n_levels > 3:
            self.register_buffer('pre_features_4',
                                 torch.zeros(args.batch_size, n_filters_conv * self.final_conv_length))
            self.register_buffer('pre_weight1_4', torch.ones(args.batch_size, 1))
        if args.n_levels > 4:
            print('WARNING: THE NUMBER OF LEVELS CAN NOT BE BIGGER THAN 4')

    def _forward_impl(self, x):
        x = self.features(x)  # feature大小为[60, 40, 52, 1]
        x = x.view(x.shape[0], -1)  # 转换成了[60, 2080]
        flatten_features = x
        x = self.classifier_1(x)  # 进入分类器进行分类

        return x, flatten_features

    def forward(self, x):
        return self._forward_impl(x)


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)
