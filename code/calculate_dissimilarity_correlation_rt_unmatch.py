# coding=gbk
import os
import re
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
import multiprocessing
import cn_clip.clip as clip
import statsmodels.api as sm
from einops import rearrange
from nilearn import plotting
import torch.utils.data as data
import matplotlib.pyplot as plt
from timm.layers.mlp import Mlp
import torch.nn.functional as F
from scipy.stats import pearsonr
from scipy.stats import linregress
from sklearn.cluster import KMeans
from einops import rearrange, repeat
from joblib import Parallel, delayed
from scipy.stats import pearsonr as corr
from timm.layers.norm import LayerNorm2d
from timm.models.convnext import ConvNeXtBlock
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union
from statsmodels.stats.outliers_influence import variance_inflation_factor
from cn_clip.clip import load_from_name, image_transform, load_from_name, _tokenizer
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, R2Score, MetricCollection
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from dataset import prepare_pair_data, Caption_Image_dataset, SingleTrial_dataset
from dataset import unmatch_session_dataset as Unmatch_dataset
from model import EncodingModel
from utils import get_parameter_number, cosine_sheduler, ExponentialMovingAverage


def add_argparse_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--Stimulus_index_root', type=str, default='/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt')
    parser.add_argument('--Image_Caption_pairs_root', type=str, default='/public_bme/data/lishr/COCO_CN/Selected_Match_Image_Caption_pairs.txt')
    # parser.add_argument('--processed_root', type=str, default='/public_bme/data/lishr/Cross_modal/Processed_Data/')
    parser.add_argument('--root', type=str, default='/public_bme2/bme-liyuanning/lishr/Cross_modal/Data')
    parser.add_argument('--processed_root', type=str, default='/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data')
    parser.add_argument('--data_type', type=str, default='beta_zscore')
    parser.add_argument('--image_root', type=str, default='/public_bme/data/lishr/COCO_CN/All_images_480')
    parser.add_argument('--CLIP_model_root', type=str, default='/public/home/lishr2022/Project/Cross-modal/encoding/cross_encoding/model_cache')
    parser.add_argument('--output_dir', type=str, default='/public/home/lishr2022/Project/Cross-modal/reconstruction/alexnet')
    parser.add_argument('--vit_model_root', type=str, default='/public/home/lishr2022/Project/Cross-modal/encoding/visual_encoding/model_cache/models--google--vit-huge-patch14-224-in21k')
    parser.add_argument('--bert_model_root', type=str, default='/public/home/lishr2022/Project/Cross-modal/encoding/language_encoding/model_cache/models--hfl--chinese-roberta-wwm-ext-large')

    parser.add_argument('--Index_root', type=str, default=None)

    parser.add_argument('--behavior_in', type=int, default=8)
    parser.add_argument('--behavior_hidden', type=int, default=16)
    parser.add_argument('--final_visual_emb_dim', type=int, default=64)
    parser.add_argument('--final_bert_emb_dim', type=int, default=64)
    parser.add_argument('--encode_type', type=str, default='visual')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--start_warmup_value', type=float, default=1e-5)
    parser.add_argument('--ema_interval', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--reg_weight', type=float, default=3e-5)
    parser.add_argument('--pearson_weight', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--val_batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--subject', type=str)

    return parser


def count_layer(data, label, x_label, y_label, title, savedir=None,):
    fig = plt.figure(figsize=(12, 6))
    model_types = ['Bert', 'CLIP_Bert', 'CLIP_model', 'ViT', 'CLIP_ViT', 'Alexnet', 'Resnet18', 'Resnet50']
    colors = ['lightblue', 'lightcoral', 'cornflowerblue', 'plum', 'salmon', 'mediumpurple', 'darkseagreen', 'goldenrod']
    for i, model_type in enumerate(model_types):
        start = -1
        end = -1
        for index, layer in enumerate(label):
            if model_type in layer and start == -1:
                start = index
            if start != -1 and model_type not in layer and end == -1:
                end = index
        if end == -1:
            end = index + 1
        if start != -1:
            x = range(start, end)
            plt.bar(x, data[start: end], color=colors[i], label=model_type)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    if savedir is not None:
        fig.savefig(savedir)

    plt.close(fig)


def zscore(data):
    z_scores = (data - np.mean(data)) / np.std(data)

    return z_scores


if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser = add_argparse_args(parser)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 根据subject初始化
    if args.subject == 'S1':
        args.num_vertices = 300245
        args.lh_vertices = 149079
        args.rh_vertices = 151166
    elif args.subject == 'S2':
        args.num_vertices = 270826
        args.lh_vertices = 135103
        args.rh_vertices = 135723
    elif args.subject == 'S3':
        args.num_vertices = 306598
        args.lh_vertices = 155295
        args.rh_vertices = 151303
    elif args.subject == 'S4':
        args.num_vertices = 284718
        args.lh_vertices = 141922
        args.rh_vertices = 142796
    elif args.subject == 'S5':
        args.num_vertices = 280414
        args.lh_vertices = 141578
        args.rh_vertices = 138836
    elif args.subject == 'S6':
        args.num_vertices = 295579
        args.lh_vertices = 146440
        args.rh_vertices = 149139
    elif args.subject == 'S7':
        args.num_vertices = 290278
        args.lh_vertices = 145747
        args.rh_vertices = 144531
    elif args.subject == 'S8':
        args.num_vertices = 258073
        args.lh_vertices = 129958
        args.rh_vertices = 128115

    args.coords_root = os.path.join(args.processed_root, args.subject, 'coords.npy')
    hemi_vertices = [args.lh_vertices, args.rh_vertices]

    coords = np.load(args.coords_root)
    coords = torch.from_numpy(coords).to(device, dtype=torch.float)

    # args.output_dir = os.path.join(args.output_dir, args.subject + '_dissimilarity_2')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    Stimulus_index, Stimulus_pairs, stim_dict, train_list, val_list = prepare_pair_data(args.subject, args.Image_Caption_pairs_root, args.Stimulus_index_root, args.processed_root, session_type='Stimulus', data_type=args.data_type)

    subject_beta_root = os.path.join(args.processed_root, args.subject, 'unmatch', 'beta')
    FileList = sorted(os.listdir(subject_beta_root))
    unmatch_list = []
    for index, file in tqdm(enumerate(FileList)):
        if file[-5] != 'a':
            stim = int(file[-24:-19])
        else:
            stim = int(file[-22:-17])
        mat_root = os.path.join(subject_beta_root, file)
        temp = io.loadmat(mat_root)
        pair_stim = temp['pair_stim'][0][0]
        pair_stim = Stimulus_index[Stimulus_pairs[Stimulus_index[pair_stim]]]

        if stim != pair_stim and Stimulus_index[stim][-3:] == 'jpg':
            unmatch_list.append(mat_root)

    print('Total length of unmatch_list %d' % (len(unmatch_list)))

    # if not os.path.exists(os.path.join(args.output_dir, 'reaction_time.csv')):
    if 1:
        script_root = os.path.join(args.root, args.subject, 'behavior', 'unmatch')
        FileList = os.listdir(script_root)
        ReactionTime = []
        for file in tqdm(unmatch_list):
            stim = int(re.findall(r'stim_(\d{5})', file)[0])
            run = int(re.findall(r'Run_(\d{3})', file)[0])
            if file[-5] != 'a':
                c = 2
            else:
                c = 1

            for script in FileList:
                if '_run' + str(run) + '_' in script and '.mat' in script:
                    behavior_data = io.loadmat(os.path.join(script_root, script))
                    break
            rt = behavior_data['theData']['rt'][0][0][0]
            trials = behavior_data['theSubject']['trial'][0][0][0][0][4][0]
            unmatch = behavior_data['theSubject']['trial'][0][0][0][0][3][0]
            img = [trials[t][0] for t in range(len(trials))]

            trials = behavior_data['theSubject']['trial'][0][0][0][0][5][0]
            cap = [trials[t][0] for t in range(len(trials))]

            for i, t in enumerate(img):
                if t == 'None':
                    continue
                if Stimulus_pairs[cap[i]] != t:
                    if c == 1:
                        ReactionTime.append(rt[i])
                        c = c - 1
                    else:
                        c = c - 1

        ReactionTime = {'Reaction_Time': ReactionTime}
        savedir = os.path.join(args.output_dir, 'reaction_time.csv')
        df = pd.DataFrame(ReactionTime)
        df.to_csv(savedir, index=False)

    # if not os.path.exists(os.path.join(args.output_dir, 'rt_correlation.txt')):
    if 1:
        reactiontime = pd.read_csv(os.path.join(args.output_dir, 'reaction_time.csv'))
        unmatch_dissimilarity = pd.read_csv(os.path.join(args.output_dir, 'unmatch_dissimilarity.csv'))

        rt = reactiontime['Reaction_Time']
        label = unmatch_dissimilarity.keys()
        f = open(os.path.join(args.output_dir, 'rt_correlation.txt'), 'w')
        f.write('Layer\tCorrelation\tp-Value\n')
        for index, layer in enumerate(label):
            feature = unmatch_dissimilarity[layer]

            Y = []
            X = []
            for i in range(len(feature)):
                if rt[i] != 0:
                    Y.append(rt[i])
                    X.append(feature[i])
            Y = zscore(Y)
            r, p = pearsonr(X, Y)
            f.write('%s\t%f\t%f\n' % (layer, r, p))
        f.close()

    if not os.path.exists(os.path.join(args.output_dir, args.subject+'_reaction_time_corr_lh.csv')):
        reactiontime = pd.read_csv(os.path.join(args.output_dir, 'reaction_time.csv'))
        rt = reactiontime['Reaction_Time']

        beta = np.loadtxt(os.path.join(args.output_dir, 'beta.csv'), delimiter=',')
        for i in range(beta.shape[1]):
            beta[:, i] = zscore(beta[:, i])
        data = []
        Y = []
        for i in range(len(rt)):
            if rt[i] != 0:
                data.append(beta[i])
                Y.append(rt[i])
        data = np.array(data)
        Y = zscore(Y)

        corr = []
        p_value = []
        for i in tqdm(range(beta.shape[1]), total=beta.shape[1]):
            X = data[:, i]
            r, p = pearsonr(X, Y)
            corr.append(r)
            p_value.append(p)
        lh = {
            'Corr': corr[:hemi_vertices[0]],
            'p_value': p_value[:hemi_vertices[0]],
            'possibility': 1 - np.array(p_value[:hemi_vertices[0]])
        }
        savedir = os.path.join(args.output_dir, args.subject+'_reaction_time_corr_lh.csv')
        df = pd.DataFrame(lh)
        df.to_csv(savedir, index=False)

        rh = {
            'Corr': corr[hemi_vertices[0]:],
            'p_value': p_value[hemi_vertices[0]:],
            'possibility': 1 - np.array(p_value[hemi_vertices[0]:])
        }
        savedir = os.path.join(args.output_dir, args.subject+'_reaction_time_corr_rh.csv')
        df = pd.DataFrame(rh)
        df.to_csv(savedir, index=False)
