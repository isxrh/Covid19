import torch
import torch.nn as nn

class Resid_block(nn.Module):
    '''
    One simple residual block
    '''
    def __init__(self, inchannels, outchannels, padding=1, stride=1, downsample=None):
        super(Resid_block, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.bn2 = nn.BatchNorm2d(outchannels)
        self.downsample = downsample


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)

        return out


class ConvCore(nn.Module):
    def __init__(self, dropout=None, depth_scale=1, inchannels=3, fc_in_features=8000):
        super().__init__()
        self.d = depth_scale
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        self.conv1 = nn.Conv2d(inchannels,
                               int(32 * self.d),
                               kernel_size=(7, 3),
                               padding=1)
        self.resid1 = Resid_block(int(32 * self.d),
                                  int(64 * self.d),
                                  padding=(1, 1),
                                  stride=(2, 1),
                                  downsample=nn.Conv2d(int(32 * self.d),
                                                       int(64 * self.d),
                                                       kernel_size=1,
                                                       stride=(2, 1)))
        self.resid2 = Resid_block(int(64 * self.d),
                                  int(128 * self.d),
                                  padding=(1, 1),
                                  stride=(2, 1),
                                  downsample=nn.Conv2d(int(64 * self.d),
                                                       int(128 * self.d),
                                                       kernel_size=1,
                                                       stride=(2, 1)))
        self.resid3 = Resid_block(int(128 * self.d),
                                  int(256 * self.d),
                                  padding=(1, 1),
                                  stride=(2, 1),
                                  downsample=nn.Conv2d(int(128 * self.d),
                                                       int(256 * self.d),
                                                       kernel_size=1,
                                                       stride=(2, 1)))
        self.resid4 = Resid_block(int(256 * self.d),
                                  int(512 * self.d),
                                  padding=(1, 1),
                                  stride=(2, 1),
                                  downsample=nn.Conv2d(int(256 * self.d),
                                                       int(512 * self.d),
                                                       kernel_size=1,
                                                       stride=(2, 1)))
        self.resid5 = Resid_block(int(512 * self.d),
                                  int(128 * self.d),
                                  stride=2,
                                  downsample=nn.Conv2d(int(512 * self.d),
                                                       int(128 * self.d),
                                                       kernel_size=1,
                                                       stride=2))
        self.resid6 = Resid_block(int(128 * self.d),
                                  int(64 * self.d),
                                  stride=2,
                                  downsample=nn.Conv2d(int(128 * self.d),
                                                       int(64 * self.d),
                                                       kernel_size=1,
                                                       stride=2))
        self.resid7 = Resid_block(int(64 * self.d),
                                  int(32 * self.d),
                                  stride=2,
                                  downsample=nn.Conv2d(int(64 * self.d),
                                                       int(32 * self.d),
                                                       kernel_size=1,
                                                       stride=2))
        self.resid8 = Resid_block(int(32 * self.d),
                                  int(16 * self.d),
                                  downsample=nn.Conv2d(int(32 * self.d),
                                                       int(16 * self.d),
                                                       kernel_size=1,
                                                       stride=1))
        self.resid9 = Resid_block(int(16 * self.d),
                                  int(8 * self.d),
                                  downsample=nn.Conv2d(int(16 * self.d),
                                                       int(8 * self.d),
                                                       kernel_size=1,
                                                       stride=1))

        self.FC1 = nn.Linear(fc_in_features, 50)
        self.FC2 = nn.Linear(50,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x.float())
        if self.dropout != None:
            out = self.dropout(out)
        out = self.resid1(out)
        out = self.resid2(out)
        out = self.resid3(out)
        out = self.resid4(out)
        out = self.resid5(out)
        out = self.resid6(out)
        out = self.resid7(out)
        out = self.resid8(out)
        out = self.resid9(out)
        
        out = self.FC1(out.view(out.size()[0], -1))
        out = self.relu(out)
        out = self.FC2(out)
        out = torch.sigmoid(out)
        out = out.squeeze(-1)
        return out


def Cider(inputs_inchannel, fc_in_features):
    return ConvCore(inchannels=inputs_inchannel, fc_in_features=fc_in_features)
