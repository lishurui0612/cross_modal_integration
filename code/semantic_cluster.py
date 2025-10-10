# coding=gbk
import os
import gc
import torch
import random
import logging
import nilearn
import argparse
import numpy as np
import pandas as pd
from torch import nn
from scipy import io
from PIL import Image
from tqdm import tqdm
import cn_clip.clip as clip
from einops import rearrange
from nilearn import plotting
import torch.utils.data as data
import matplotlib.pyplot as plt
from timm.layers.mlp import Mlp
import torch.nn.functional as F
from sklearn.cluster import KMeans
from einops import rearrange, repeat
from info_nce import InfoNCE, info_nce
from timm.layers.norm import LayerNorm2d
from einops.layers.torch import Rearrange
from timm.models.convnext import ConvNeXtBlock
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Any, Dict, List, Optional, Tuple, Union
from cn_clip.clip import load_from_name, image_transform, load_from_name, _tokenizer
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, R2Score, MetricCollection

from dataset import prepare_pair_data, Caption_Image_dataset, Caption_Image_2d_dataset, natural_scene_dataset
from model_cae import BrainCLIP
from utils import get_parameter_number, cosine_sheduler, ExponentialMovingAverage

if __name__ == '__main__':
    Stimulus_index_root = '/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt'
    Image_Caption_pairs_root = '/public_bme/data/lishr/COCO_CN/Selected_Match_Image_Caption_pairs.txt'
    image_root = '/public_bme/data/lishr/COCO_CN/All_images_480'
    CLIP_model_root = '/public/home/lishr2022/Project/Cross-modal/encoding/cross_encoding/model_cache'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLIP_model, preprocess = load_from_name('ViT-H-14', device=device, download_root=CLIP_model_root)
    CLIP_model.eval()

    with torch.no_grad():
        CLIP_feature = []
        with open(Stimulus_index_root, 'r', encoding='gbk') as f:
            content = f.readlines()
            for line in tqdm(content):
                temp = line.split()
                if temp[1][-3:] == 'jpg':
                    image = preprocess(Image.open(os.path.join(image_root, temp[1]))).unsqueeze(0).to(device)
                    image_feature = CLIP_model.encode_image(image)
                    image_feature /= image_feature.norm(dim=-1, keepdim=True)
                    CLIP_feature.append(image_feature.detach().cpu().numpy())
                else:
                    text = clip.tokenize([temp[1]]).to(device)
                    text_feature = CLIP_model.encode_text(text)
                    text_feature /= text_feature.norm(dim=-1, keepdim=True)
                    CLIP_feature.append(text_feature.detach().cpu().numpy())
        f.close()
        CLIP_feature = np.concatenate(CLIP_feature, axis=0)

    kmeans = KMeans(n_clusters=10, random_state=1000)
    kmeans.fit(CLIP_feature)
    label = kmeans.predict(CLIP_feature)

    target_dir = '/public/home/lishr2022/Project/Cross-modal/beta_estimate/clip_cluster.npy'
    np.save(target_dir, label)

    target_dir = '/public/home/lishr2022/Project/Cross-modal/beta_estimate/clip_features.npy'
    np.save(target_dir, CLIP_feature)
