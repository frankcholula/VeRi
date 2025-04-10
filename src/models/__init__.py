# Copyright (c) EEEM071, University of Surrey


from .self_attention import resnet18_self_attention
from .squeeze_and_excite import (
    resnet18_se,
    resnet50_se,
    resnet18_se_fc512,
    resnet50_se_fc512,
)
from .resnet import (
    resnet18,
    resnet18_fc512,
    resnet34,
    resnet34_fc512,
    resnet50,
    resnet50_fc512,
)
from .tvmodels import mobilenet_v3_small, vgg16, vit_b_16


__model_factory = {
    # image classification models
    "resnet18": resnet18,
    "resnet18_fc512": resnet18_fc512,
    "resnet34": resnet34,
    "resnet34_fc512": resnet34_fc512,
    "resnet50": resnet50,
    "resnet50_fc512": resnet50_fc512,
    "mobilenet_v3_small": mobilenet_v3_small,
    "vgg16": vgg16,
    # custom models
    "resnet18_se": resnet18_se,
    "resnet18_se_fc512": resnet18_se_fc512,
    "resnet18_self_attention": resnet18_self_attention,
    "resnet50_se": resnet50_se,
    "resnet50_se_fc512": resnet50_se_fc512,
    "vit_b_16": vit_b_16,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    return __model_factory[name](*args, **kwargs)
