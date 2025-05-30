# Copyright (c) EEEM071, University of Surrey

import os.path as osp
import shutil
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn

from .iotools import mkdir_if_missing


def save_checkpoint(state, save_dir, is_best=False, remove_module_from_keys=False):
    mkdir_if_missing(save_dir)
    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict
    # save
    epoch = state["epoch"]
    fpath = osp.join(save_dir, f"model-{str(epoch)}.pth.tar")
    wpath = osp.join(save_dir, f"model-{str(epoch)}.pth")
    torch.save(state, fpath)
    torch.save(state["state_dict"], wpath)
    print(f'Checkpoint saved to "{fpath}"')
    print(f"Weights saved to {wpath}")
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), "best_model.pth.tar"))
        shutil.copy(wpath, osp.join(osp.dirname(wpath), "best_model.pth"))


def resume_from_checkpoint(ckpt_path, model, optimizer=None):
    print(f'Loading checkpoint from "{ckpt_path}"')
    # force to load entire Python object
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print("Loaded model weights")
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("Loaded optimizer")
    start_epoch = ckpt["epoch"]
    print(
        "** previous epoch = {}\t previous rank1 = {:.1%}".format(
            start_epoch, ckpt["rank1"]
        )
    )
    return start_epoch


def adjust_learning_rate(
    optimizer,
    base_lr,
    epoch,
    stepsize=20,
    gamma=0.1,
    linear_decay=False,
    final_lr=0,
    max_epoch=100,
):
    if linear_decay:
        # linearly decay learning rate from base_lr to final_lr
        frac_done = epoch / max_epoch
        lr = frac_done * final_lr + (1.0 - frac_done) * base_lr
    else:
        # decay learning rate by gamma for every stepsize
        lr = base_lr * (gamma ** (epoch // stepsize))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def open_all_layers(model):
    """
    Open all layers in model for training.
    Args:
    - model (nn.Module): neural net model.
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    """
    Open specified layers in model for training while keeping
    other layers frozen.
    Args:
    - model (nn.Module): neural net model.
    - open_layers (list): list of layer names.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    for layer in open_layers:
        assert hasattr(
            model, layer
        ), '"{}" is not an attribute of the model, please provide the correct name'.format(
            layer
        )

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_num_param(model):
    num_param = sum(p.numel() for p in model.parameters()) / 1e06

    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e06
    return num_param


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if isinstance(output, (tuple, list)):
            output = output[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            res.append(acc.item())
        return res


def load_pretrained_weights(model, weight_path):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    checkpoint = torch.load(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, please check the key names manually (** ignored and continue **)'.format(
                weight_path
            )
        )
    else:
        print(f'Successfully loaded pretrained weights from "{weight_path}"')
        if len(discarded_layers) > 0:
            print(
                "** The following layers are discarded due to unmatched keys or layer size: {}".format(
                    discarded_layers
                )
            )
