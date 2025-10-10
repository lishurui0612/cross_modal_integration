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


def add_argparse_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--Stimulus_index_root', type=str, default='/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt')
    parser.add_argument('--Image_Caption_pairs_root', type=str, default='/public_bme/data/lishr/COCO_CN/Selected_Match_Image_Caption_pairs.txt')
    parser.add_argument('--processed_root', type=str, default='/public_bme/data/lishr/Cross_modal/Processed_Data/')
    parser.add_argument('--image_root', type=str, default='/public_bme/data/lishr/COCO_CN/All_images_480')
    parser.add_argument('--CLIP_model_root', type=str, default='/public/home/lishr2022/Project/Cross-modal/encoding/cross_encoding/model_cache')
    parser.add_argument('--nsd_root', type=str, default='/public_bme/data/lishr/NSD/nsddata_betas/ppdata')
    parser.add_argument('--output_dir', type=str, default='/public/home/lishr2022/Project/Cross-modal/Predictive_coding/CLIP_3class')
    # encoding model
    parser.add_argument('--behavior_in', type=int, default=8)
    parser.add_argument('--behavior_hidden', type=int, default=16)
    parser.add_argument('--final_visual_emb_dim', type=int, default=64)
    parser.add_argument('--final_bert_emb_dim', type=int, default=1024)
    # contextual autoencoder
    parser.add_argument('--mask_proportion', type=float, default=0.75)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--type', type=str, default='vit')
    parser.add_argument('--data_type', type=str, default='beta_zscore')
    parser.add_argument('--use_ne', type=int, default=0)
    parser.add_argument('--post_type', type=str, default='linear')
    parser.add_argument('--Index_root', type=str, default=None)
    # head parameters
    parser.add_argument('--head_depth', type=int, default=1)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--head_align_weight', type=float, default=1.0)
    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--start_warmup_value', type=float, default=1e-5)
    parser.add_argument('--ema_interval', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--reg_weight', type=float, default=3e-5)
    parser.add_argument('--pearson_weight', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--subject', type=str)

    return parser


if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser = add_argparse_args(parser)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    caption_list = []
    image_list = []
    with open(args.Image_Caption_pairs_root, 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            image_list.append(temp[0])
            caption_list.append(temp[1])
    f.close()

    model, preprocess = load_from_name('ViT-H-14', device=device, download_root=args.CLIP_model_root)
    model.eval()

    results = [[], [], []]
    for k in range(10):
        normal = 0
        shuffle = 0
        unmatch = 0
        for i in tqdm(range(len(caption_list))):
            image = preprocess(Image.open(os.path.join(args.image_root, image_list[i]))).unsqueeze(0).to(device)

            # Normal
            normal_caption = caption_list[i]

            # Shuffle
            shuffled_caption = list(caption_list[i])
            random.shuffle(shuffled_caption)
            shuffled_caption = "".join(shuffled_caption)

            # Unmatch
            while True:
                temp = random.randint(0, len(caption_list)-1)
                if temp != i:
                    break
            unmatched_caption = caption_list[temp]

            text = [normal_caption, shuffled_caption, unmatched_caption]
            text = clip.tokenize(text).to(device)

            logits_per_image, _ = model.get_similarity(image, text)
            probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
            probs = probs[0]

            normal += probs[0]
            shuffle += probs[1]
            unmatch += probs[2]

        normal /= len(caption_list)
        shuffle /= len(caption_list)
        unmatch /= len(caption_list)

        results[0].append(normal)
        results[1].append(shuffle)
        results[2].append(unmatch)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i in range(3):
        results[i] = np.stack(results[i])
    np.savetxt(os.path.join(args.output_dir, 'results.csv'), results, delimiter=',')

    normal = np.mean(results[0])
    normal_std = np.std(results[0])

    shuffle = np.mean(results[1])
    shuffle_std = np.std(results[1])

    unmatch = np.mean(results[2])
    unmatch_std = np.std(results[2])

    with open(os.path.join(args.output_dir, 'results.txt'), 'w', encoding='gbk') as f:
        f.write('Average similarity in caption response is %.10f, std is %.10f\n' % (normal, normal_std))
        f.write('Average similarity in shuffled caption response is %.10f, std is %.10f\n' % (shuffle, shuffle_std))
        f.write('Average similarity in incorrected caption response is %.10f, std is %.10f\n\n' % (unmatch, unmatch_std))
    f.close()

