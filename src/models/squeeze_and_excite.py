from src.models.resnet import (
    ResNet,
    BasicBlock,
    Bottleneck,
    init_pretrained_weights,
    model_urls,
)
from torch import nn


class SEBasicBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__(inplanes, planes, stride, downsample)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, planes // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes // reduction, planes, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE attention
        out = out * self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__(inplanes, planes, stride, downsample)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                planes * self.expansion,
                planes * self.expansion // reduction,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                planes * self.expansion // reduction,
                planes * self.expansion,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply SE attention!
        out = out * self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet18_se(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=SEBasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs,
    )
    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])
    return model


def resnet18_se_fc512(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=SEBasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        **kwargs,
    )
    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])
    return model


def resnet50_se(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=SEBottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs,
    )
    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])
    return model


def resnet50_se_fc512(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=SEBottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        **kwargs,
    )
    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])
    return model
