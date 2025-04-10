import torch
from torch import nn
from resnet import ResNet, init_pretrained_weights, model_urls
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """
    Self attention module for ResNet

    Implementation inspired by Non-Local Neural Networks (Wang et al.)
    https://arxiv.org/abs/1711.07971
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(SelfAttention, self).__init__()

        # Reduce channels for computation efficiency
        self.reduced_channels = in_channels // reduction_ratio

        self.query_conv = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.output_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.output_conv(out)

        return self.gamma * out + x


class SelfAttentionBasicBlock(nn.Module):
    """
    ResNet Basic Block with Self Attention
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SelfAttentionBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.self_attention = SelfAttention(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply self-attention
        out = self.self_attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet18_self_attention(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    """
    Create ResNet-18 model with self-attention blocks
    """
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=SelfAttentionBasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs,
    )

    if pretrained:
        # Load pretrained ResNet-18 weights
        init_pretrained_weights(model, model_urls["resnet18"])

    return model
