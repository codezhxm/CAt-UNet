import torch
from torch import nn
import torch.nn.functional as F


# Taken from https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class SpaceToDepth(nn.Module):
    # Expects the following shape: Batch, Channel, Height, Width
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x

class DoubleDense(nn.Module):
    def __init__(self, in_channels, hidden_neurons, output_channels):
        super(DoubleDense, self).__init__()
        self.dense1 = nn.Linear(in_channels, out_features=hidden_neurons)
        self.dense2 = nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons // 2)
        self.dense3 = nn.Linear(in_features=hidden_neurons // 2, out_features=output_channels)

    def forward(self, x):
        out = F.relu(self.dense1(x.view(x.size(0), -1)))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out


class CSPABN(nn.Module):   ## Channel and space parallel attention
    def __init__(self, channel, reduction=16):
        super(CSPABN, self).__init__()
        #self.avg_pool = nn.Ad(1)
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        self.pool_c = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            #nn.Linear(channel, channel // reduction, bias=False),
            nn.Conv2d(channel, channel // reduction,kernel_size=1,stride=1, bias=False),
            # nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d( channel // reduction,channel, kernel_size=1, stride=1, bias=False),
            # nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        #b, c, _, _ = x.size()
        #y = self.avg_pool(x).view(b, c)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        #x_hw= x_w+x_h
        x_c = self.pool_c(x)
        x_h = self.fc(x_h)
        x_w = self.fc(x_w)
        x_c = self.fc(x_c)
        x=x * x_h * x_w * x_c
        # print(x.shape)
        return x  #x * x_h * x_w * x_c


class CSPA(nn.Module):   ## Channel and space parallel attention
    def __init__(self, channel, reduction=16):
        super(CSPA, self).__init__()
        #self.avg_pool = nn.Ad(1)
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        self.pool_c = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            #nn.Linear(channel, channel // reduction, bias=False),
            nn.Conv2d(channel, channel // reduction,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d( channel // reduction,channel, kernel_size=1, stride=1, bias=False),
            # nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        #b, c, _, _ = x.size()
        #y = self.avg_pool(x).view(b, c)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        #x_hw= x_w+x_h
        x_c = self.pool_c(x)
        x_h = self.fc(x_h)
        x_w = self.fc(x_w)
        x_c = self.fc(x_c)
        x=x * x_h * x_w * x_c
        # print(x.shape)
        return x  #x * x_h * x_w * x_c

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, input_size, kernel_size, bias=False):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = 1, 1
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, h_cur, c_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class RNN(nn.Module):
    def __init__(self, num_hidden, frame_channel, width):
        super(RNN, self).__init__()

        self.frame_channel = frame_channel
        self.num_layers = len(num_hidden)
        self.num_hidden = num_hidden
        self.device = 'cuda'
        self.height = width
        self.width = width
        cell_list = []

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], self.width, 3,
                                       1)
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, frames_tensor):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        # frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        # mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        frames = frames_tensor
        # frames = torch.unsqueeze(frames_tensor,dim=2)
        batch = frames.shape[0]
        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], self.height, self.width]).to(self.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], self.height, self.width]).to(self.device)

        for t in range(12):
            # reverse schedule sampling
            net = frames[:, t]
            h_t[0], c_t[0]= self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            # x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(h_t[self.num_layers - 1])

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        # next_frames = torch.stack(next_frames, dim=0)
        next_frames = torch.squeeze(next_frames,dim=1)
        return next_frames
