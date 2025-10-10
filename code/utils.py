# coding=gbk
import os
import gc
import torch
import random
import nilearn
import numpy as np
from torch import nn
from scipy import io
from PIL import Image
from tqdm import tqdm
import cn_clip.clip as clip
from einops import rearrange
import torch.utils.data as data
import matplotlib.pyplot as plt
from timm.layers.mlp import Mlp
import torch.nn.functional as F
from einops import rearrange, repeat
from scipy.interpolate import griddata
from typing import Any, Dict, List, Optional, Tuple, Union
from cn_clip.clip import load_from_name, image_transform, load_from_name, _tokenizer
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, R2Score, MetricCollection


def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp

    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()

    loss = (loss1 + loss2) / 2
    return loss


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def cosine_sheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_epochs)
    iters = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def grid2fMRI(data, azimuth, elevation, img_size):
    data = data.reshape(-1)

    X, Y = np.meshgrid(np.linspace(-1, 1, img_size), np.linspace(-1, 1, img_size))

    elevation = np.sin(elevation)
    elevation = 2 * elevation / (np.max(elevation) - np.min(elevation))
    azimuth = 2 * azimuth / (np.max(azimuth) - np.min(azimuth))

    transformed_gridmap = griddata((azimuth, elevation), data, (X, Y), method='nearest')

    return transformed_gridmap


