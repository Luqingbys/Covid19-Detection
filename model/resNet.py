from torch import nn
import torch
import torch.nn.functional as F

class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride) -> None:
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        # 为了统一残差结构下的shortcut与卷积层输出的张量尺寸，需要提供一个额外的卷积层
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    
    def forward(self, x: torch.Tensor):
        '''
        x (batch_size, c, h, w)
        '''
        # print('in resBlock:', x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn1(self.conv2(out)))
        # 加入shortcut，需要统一张量尺寸
        out = self.extra(x) + out
        return out


class ResNet(nn.Module):
    
    def __init__(self) -> None:
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # 接上4个block
        # (b, 64, h, w) => (b, 128, h, w)，以此类推
        self.blk1 = ResBlk(64, 128, stride=2)
        self.blk2 = ResBlk(128, 256, stride=2)
        self.blk3 = ResBlk(256, 512, stride=2)
        self.blk4 = ResBlk(512, 1024, stride=2)

        # 池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.outlayer = nn.Linear(1024, 2)


    def forward(self, x: torch.Tensor):
        # print('at first:', x.shape)
        x = F.relu(self.conv1(x))
        # print('before resBlock1:', x.shape)
        x = self.blk1(x)
        # print('before resBlock2:', x.shape)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # x = F.adaptive_avg_pool2d(x, output_size=[1, 1])
        # print('before pool: ', x.shape)
        # x = F.adaptive_avg_pool2d(x)
        x = self.avg_pool(x)
        # print('after pool: ', x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x