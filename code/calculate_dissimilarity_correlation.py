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
import multiprocessing
import cn_clip.clip as clip
import statsmodels.api as sm
from einops import rearrange
from nilearn import plotting
import torch.utils.data as data
import matplotlib.pyplot as plt
from timm.layers.mlp import Mlp
import torch.nn.functional as F
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

from dataset import prepare_pair_data, Caption_Image_dataset, SingleTrial_dataset, Unmatch_dataset
from model import EncodingModel
from utils import get_parameter_number, cosine_sheduler, ExponentialMovingAverage


def add_argparse_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--Stimulus_index_root', type=str, default='/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt')
    parser.add_argument('--Image_Caption_pairs_root', type=str, default='/public_bme/data/lishr/COCO_CN/Selected_Match_Image_Caption_pairs.txt')
    # parser.add_argument('--processed_root', type=str, default='/public_bme/data/lishr/Cross_modal/Processed_Data/')
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


def calculate_vif(data, i):
    return variance_inflation_factor(data.values, i)


def plot_weight(stat, subject, hemi_vertices, savedir, vmax=1.0):
    fig = plt.figure(figsize=(24, 8))
    axes = []
    for i in range(8):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        axes.append(ax)

    # vmax = np.max(abs(stat))
    # if vmax != 0.8:
    #     stat = stat / vmax

    view = ['lateral', 'ventral', 'medial', 'dorsal']

    left_surf_mesh = os.path.join('/public_bme/data/lishr/Cross_modal/subjects', subject[3:], 'surf', 'lh.inflated')
    right_surf_mesh = os.path.join('/public_bme/data/lishr/Cross_modal/subjects', subject[3:], 'surf', 'rh.inflated')

    left_stat = stat[:hemi_vertices[0]]
    right_stat = stat[hemi_vertices[0]:]
    # plot left hemisphere
    print('- - - plotting left hemisphere')
    for i in range(4):
        plotting.plot_surf_stat_map(
            surf_mesh = left_surf_mesh,
            stat_map = left_stat,
            hemi = 'left',
            view = view[i],
            vmax = vmax,
            axes = axes[i],
            title = 'Left - ' + view[i]
        )
    # plot right hemisphere
    print('- - - plotting right hemisphere')
    for i in range(4):
        plotting.plot_surf_stat_map(
            surf_mesh = right_surf_mesh,
            stat_map = right_stat,
            hemi = 'right',
            view = view[i],
            vmax = vmax,
            axes = axes[i+4],
            title = 'Right - ' + view[i]
        )

    fig.savefig(savedir)
    plt.close(fig)


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
    match_list = train_list + val_list

    if not os.path.exists(os.path.join(args.processed_root, args.subject, 'unmatched_image.txt')):
        subject_beta_root = os.path.join(args.processed_root, args.subject, 'Stimulus', args.data_type)
        FileList = sorted(os.listdir(subject_beta_root))
        unmatch_list = []
        for index, file in tqdm(enumerate(FileList)):
            mat_root = os.path.join(subject_beta_root, file)
            temp = io.loadmat(mat_root)
            image = temp['image'][0][0]
            if image == 0:
                continue
            pair_unmatch = temp['unmatch'][0][0]
            if pair_unmatch == 1:
                unmatch_list.append(mat_root)
        with open(os.path.join(args.processed_root, args.subject, 'unmatched_image.txt'), 'w', encoding='gbk') as f:
            f.write('\n'.join(unmatch_list))
    else:
        with open(os.path.join(args.processed_root, args.subject, 'unmatched_image.txt'), 'r', encoding='gbk') as f:
            lines = f.readlines()
        unmatch_list = [line.strip() for line in lines]

    print('Total length of train_list: %d, total length of val_list %d, total length of unmatch_list %d' % (len(train_list), len(val_list), len(unmatch_list)))

    '''
    根据ROI_index，提取beta数据
    '''
    if args.Index_root is None:
        temp = np.arange(args.num_vertices)
    else:
        temp = np.loadtxt(args.Index_root)
    ROI_index = temp[temp != -1].astype(int)
    ROI_index = torch.from_numpy(ROI_index).to(device, dtype=torch.long)

    if not os.path.exists(os.path.join(args.output_dir, 'beta.csv')):
        # Load dataset
        unmatch_dataset = Unmatch_dataset(unmatch_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False, encode_type='visual', type='visual')
        unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        beta = []
        for index, sample in tqdm(enumerate(unmatch_dataloader), total=len(unmatch_dataloader)):
            for i in range(len(sample)):
                if type(sample[i]) == torch.Tensor:
                    sample[i] = sample[i].to(device, dtype=torch.float)

            sample[4] = sample[4].permute(1, 0)
            temp = sample[4][ROI_index]
            temp = temp.permute(1, 0)
            beta.append(temp.cpu().detach().numpy())
        beta = np.vstack(beta)

        savedir = os.path.join(args.output_dir, 'beta.csv')
        np.savetxt(savedir, beta, delimiter=',')
        print('Beta array saved to CSV file successfully.')

        del beta

    # if not os.path.exists(os.path.join(args.output_dir, 'match_beta.csv')):
    #     # Load dataset
    #     unmatch_dataset = Unmatch_dataset(match_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False, encode_type='visual', type='visual')
    #     unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    #
    #     match_beta = []
    #     for index, sample in tqdm(enumerate(unmatch_dataloader), total=len(unmatch_dataloader)):
    #         for i in range(len(sample)):
    #             if type(sample[i]) == torch.Tensor:
    #                 sample[i] = sample[i].to(device, dtype=torch.float)
    #
    #         sample[4] = sample[4].permute(1, 0)
    #         temp = sample[4][ROI_index]
    #         temp = temp.permute(1, 0)
    #         match_beta.append(temp.cpu().detach().numpy())
    #     match_beta = np.vstack(match_beta)
    #
    #     savedir = os.path.join(args.output_dir, 'match_beta.csv')
    #     np.savetxt(savedir, match_beta, delimiter=',')
    #     print('Match Beta array saved to CSV file successfully.')
    #
    #     del match_beta

    if not os.path.exists(os.path.join(args.output_dir, 'unmatch_dissimilarity.csv')):
        unmatch_dissimilarity = {}

        '''
        1. Extract Bert semantic similarity
        '''
        print('Extracting Bert semantic similarity')
        # 初始化存储地址
        Bert_Layers = [1, 2, 4, 8, 12, 16, 20, 24]
        for layer in Bert_Layers:
            unmatch_dissimilarity['Bert_layer_' + str(layer)] = []

        # Load dataset
        unmatch_dataset = Unmatch_dataset(unmatch_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False, encode_type='caption', type='visual')
        unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        model = EncodingModel(
            num_voxels=args.num_vertices,
            coords=coords,
            behavior_in=args.behavior_in,
            behavior_hidden=args.behavior_hidden,
            final_visual_emb_dim=args.final_visual_emb_dim,
            final_bert_emb_dim=args.final_bert_emb_dim,
            CLIP_model_root=args.CLIP_model_root,
            encode_type='caption',
            vit_model_root=None,
            bert_model_root=args.bert_model_root
        )
        model = model.to(device, dtype=torch.float)

        model.eval()

        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
            # 针对于unmatch_list
            for index, sample in tqdm(enumerate(unmatch_dataloader), total=len(unmatch_dataloader)):
                for i in range(len(sample)):
                    if type(sample[i]) == torch.Tensor:
                        sample[i] = sample[i].to(device, dtype=torch.float)

                # 对于实际看到的caption
                condition = torch.zeros_like(sample[5])
                c = model.behavior_embedding(condition)

                word_emb_1, cls_dict_1, _ = model.bert_model.get_bert_intermediate_layers(sample[3], c=c)

                # 对于实际看到的图片，所对应的caption
                condition = torch.zeros_like(sample[5])
                c = model.behavior_embedding(condition)

                word_emb_2, cls_dict_2, _ = model.bert_model.get_bert_intermediate_layers(sample[2], c=c)

                # 针对于每一层计算dissimilarity
                for layer in Bert_Layers:
                    mean_word_emb_1 = torch.mean(word_emb_1[str(layer)], dim=1)
                    mean_word_emb_2 = torch.mean(word_emb_2[str(layer)], dim=1)

                    temp_1 = 1 - cos(mean_word_emb_1, mean_word_emb_2)

                    temp_2 = 1 - cos(cls_dict_1[str(layer)], cls_dict_2[str(layer)])

                    dissimilarity = (temp_1 + temp_2) / 2

                    unmatch_dissimilarity['Bert_layer_' + str(layer)].append(dissimilarity)

        # 整合成tensor
        for layer in Bert_Layers:
            unmatch_dissimilarity['Bert_layer_' + str(layer)] = torch.cat(unmatch_dissimilarity['Bert_layer_' + str(layer)], dim=0)
            unmatch_dissimilarity['Bert_layer_' + str(layer)] = unmatch_dissimilarity['Bert_layer_' + str(layer)].detach().cpu().numpy()
            unmatch_dissimilarity['Bert_layer_' + str(layer)] = zscore(unmatch_dissimilarity['Bert_layer_' + str(layer)])

        # 删除变量减少内存消耗
        del word_emb_1, word_emb_2, cls_dict_1, cls_dict_2

        '''
        2. Extract CLIP Bert semantic similarity
        '''
        print('Extracting CLIP Bert semantic similarity')
        # 初始化存储地址
        CLIP_Bert_Layers = [1, 2, 4, 8, 12, 16, 20, 24]
        for layer in CLIP_Bert_Layers:
            unmatch_dissimilarity['CLIP_Bert_layer_' + str(layer)] = []

        # Load dataset
        unmatch_dataset = Unmatch_dataset(unmatch_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False, encode_type='caption', type='visual')
        unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        model = EncodingModel(
            num_voxels=args.num_vertices,
            coords=coords,
            behavior_in=args.behavior_in,
            behavior_hidden=args.behavior_hidden,
            final_visual_emb_dim=args.final_visual_emb_dim,
            final_bert_emb_dim=args.final_bert_emb_dim,
            CLIP_model_root=args.CLIP_model_root,
            encode_type='caption',
            vit_model_root=None,
            bert_model_root=None
        )
        model = model.to(device, dtype=torch.float)

        model.eval()

        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
            # 针对于unmatch_list
            for index, sample in tqdm(enumerate(unmatch_dataloader), total=len(unmatch_dataloader)):
                for i in range(len(sample)):
                    if type(sample[i]) == torch.Tensor:
                        sample[i] = sample[i].to(device, dtype=torch.float)

                # 对于实际看到的caption
                inputs = clip.tokenize(sample[3]).to(device)
                condition = torch.zeros_like(sample[5])
                c = model.behavior_embedding(condition)

                word_emb_1, cls_dict_1, _ = model.CLIP_model.get_bert_intermediate_layers(inputs, c=c)

                # 对于实际看到的图片，所对应的caption
                inputs = clip.tokenize(sample[2]).to(device)
                condition = torch.zeros_like(sample[5])
                c = model.behavior_embedding(condition)

                word_emb_2, cls_dict_2, _ = model.CLIP_model.get_bert_intermediate_layers(inputs, c=c)

                # 针对于每一层计算dissimilarity
                for layer in CLIP_Bert_Layers:
                    mean_word_emb_1 = torch.mean(word_emb_1[str(layer)], dim=1)
                    mean_word_emb_2 = torch.mean(word_emb_2[str(layer)], dim=1)

                    temp_1 = 1 - cos(mean_word_emb_1, mean_word_emb_2)

                    temp_2 = 1 - cos(cls_dict_1[str(layer)], cls_dict_2[str(layer)])

                    dissimilarity = (temp_1 + temp_2) / 2

                    unmatch_dissimilarity['CLIP_Bert_layer_' + str(layer)].append(dissimilarity)

        # 整合成tensor
        for layer in CLIP_Bert_Layers:
            unmatch_dissimilarity['CLIP_Bert_layer_' + str(layer)] = torch.cat(unmatch_dissimilarity['CLIP_Bert_layer_' + str(layer)], dim=0)
            unmatch_dissimilarity['CLIP_Bert_layer_' + str(layer)] = unmatch_dissimilarity['CLIP_Bert_layer_' + str(layer)].detach().cpu().numpy()
            unmatch_dissimilarity['CLIP_Bert_layer_' + str(layer)] = zscore(unmatch_dissimilarity['CLIP_Bert_layer_' + str(layer)])

        # 删除变量减少内存消耗
        del word_emb_1, word_emb_2, cls_dict_1, cls_dict_2

        '''
        3. Extract CLIP semantic similarity
        '''
        print('Extracting CLIP semantic similarity')
        # 初始化存储地址
        unmatch_dissimilarity['CLIP_model'] = []

        # Load dataset
        unmatch_dataset = Unmatch_dataset(unmatch_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=True, encode_type='cross', type='visual')
        unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        model, _ = load_from_name('ViT-H-14', device=device, download_root=args.CLIP_model_root)
        model = model.to(device, dtype=torch.float)

        model.eval()

        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
            # 针对于unmatch_list
            for index, sample in tqdm(enumerate(unmatch_dataloader), total=len(unmatch_dataloader)):
                for i in range(len(sample)):
                    if type(sample[i]) == torch.Tensor:
                        sample[i] = sample[i].to(device, dtype=torch.float)

                # 对于实际看到的caption
                inputs = clip.tokenize(sample[3]).to(device)
                text_features = model.encode_text(inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # 对于实际看到的Image
                image_features = model.encode_image(sample[2])
                image_features /= image_features.norm(dim=-1, keepdim=True)

                dissimilarity = 1 - cos(text_features, image_features)

                unmatch_dissimilarity['CLIP_model'].append(dissimilarity)

        # 整合成tensor
        unmatch_dissimilarity['CLIP_model'] = torch.cat(unmatch_dissimilarity['CLIP_model'], dim=0)
        unmatch_dissimilarity['CLIP_model'] = unmatch_dissimilarity['CLIP_model'].detach().cpu().numpy()
        unmatch_dissimilarity['CLIP_model'] = zscore(unmatch_dissimilarity['CLIP_model'])

        # 删除变量减少内存消耗
        del text_features, image_features

        '''
        4. Extract CLIP ViT semantic similarity
        '''
        print('Extracting CLIP ViT semantic similarity')
        # 初始化存储地址
        CLIP_ViT_Layers = [1, 2, 4, 8, 14, 20, 26, 32]
        for layer in CLIP_ViT_Layers:
            unmatch_dissimilarity['CLIP_ViT_layer_' + str(layer)] = []

        # Load dataset
        unmatch_dataset = Unmatch_dataset(unmatch_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False, encode_type='visual', type='visual')
        unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        model = EncodingModel(
            num_voxels=args.num_vertices,
            coords=coords,
            behavior_in=args.behavior_in,
            behavior_hidden=args.behavior_hidden,
            final_visual_emb_dim=args.final_visual_emb_dim,
            final_bert_emb_dim=args.final_bert_emb_dim,
            CLIP_model_root=args.CLIP_model_root,
            encode_type='visual',
            vit_model_root=None,
            bert_model_root=None
        )
        model = model.to(device, dtype=torch.float)

        model.eval()

        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
            # 针对于unmatch_list
            for index, sample in tqdm(enumerate(unmatch_dataloader), total=len(unmatch_dataloader)):
                for i in range(len(sample)):
                    if type(sample[i]) == torch.Tensor:
                        sample[i] = sample[i].to(device, dtype=torch.float)

                # 对于实际看到的Caption, 所对应的Image
                image = model.norm(sample[3])
                condition = torch.zeros_like(sample[5])
                c = model.behavior_embedding(condition)

                retina_grid_1, cls_dict_1, _ = model.CLIP_model.get_visual_intermediate_layers(image, c=c)

                # 对于实际看到的Image
                image = model.norm(sample[2])
                condition = torch.zeros_like(sample[5])
                c = model.behavior_embedding(condition)

                retina_grid_2, cls_dict_2, _ = model.CLIP_model.get_visual_intermediate_layers(image, c=c)

                # 针对于每一层计算dissimilarity
                for layer in CLIP_ViT_Layers:
                    retina_dissimilarity = 1 - cos(retina_grid_1[str(layer)], retina_grid_2[str(layer)])

                    retina_dissimilarity = torch.mean(retina_dissimilarity, dim=[1, 2])

                    cls_dissimilarity = 1 - cos(cls_dict_1[str(layer)], cls_dict_2[str(layer)])

                    dissimilarity = (retina_dissimilarity + cls_dissimilarity) / 2

                    unmatch_dissimilarity['CLIP_ViT_layer_' + str(layer)].append(dissimilarity)


        # 整合成tensor
        for layer in CLIP_ViT_Layers:
            unmatch_dissimilarity['CLIP_ViT_layer_' + str(layer)] = torch.cat(unmatch_dissimilarity['CLIP_ViT_layer_' + str(layer)], dim=0)
            unmatch_dissimilarity['CLIP_ViT_layer_' + str(layer)] = unmatch_dissimilarity['CLIP_ViT_layer_' + str(layer)].detach().cpu().numpy()
            unmatch_dissimilarity['CLIP_ViT_layer_' + str(layer)] = zscore(unmatch_dissimilarity['CLIP_ViT_layer_' + str(layer)])

            # 删除变量减少内存消耗
        del retina_grid_1, retina_grid_2, cls_dict_1, cls_dict_2

        '''
        5. Extract ViT semantic similarity
        '''
        print('Extracting ViT semantic similarity')
        # 初始化存储地址
        ViT_Layers = [1, 2, 4, 8, 14, 20, 26, 32]
        for layer in ViT_Layers:
            unmatch_dissimilarity['ViT_layer_' + str(layer)] = []

        # Load dataset
        unmatch_dataset = Unmatch_dataset(unmatch_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False, encode_type='visual', type='visual')
        unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        model = EncodingModel(
            num_voxels=args.num_vertices,
            coords=coords,
            behavior_in=args.behavior_in,
            behavior_hidden=args.behavior_hidden,
            final_visual_emb_dim=args.final_visual_emb_dim,
            final_bert_emb_dim=args.final_bert_emb_dim,
            CLIP_model_root=args.CLIP_model_root,
            encode_type='visual',
            vit_model_root=args.vit_model_root,
            bert_model_root=None
        )
        model = model.to(device, dtype=torch.float)

        model.eval()

        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
            # 针对于unmatch_list
            for index, sample in tqdm(enumerate(unmatch_dataloader), total=len(unmatch_dataloader)):
                for i in range(len(sample)):
                    if type(sample[i]) == torch.Tensor:
                        sample[i] = sample[i].to(device, dtype=torch.float)

                # 对于实际看到的Caption, 所对应的Image
                image = model.norm(images=sample[3], return_tensors='pt')['pixel_values'].to(device)
                condition = torch.zeros_like(sample[5])
                c = model.behavior_embedding(condition)

                retina_grid_1, cls_dict_1, _ = model.vit_model.get_visual_intermediate_layers(image, c=c)

                # 对于实际看到的Image
                image = model.norm(images=sample[2], return_tensors='pt')['pixel_values'].to(device)
                condition = torch.zeros_like(sample[5])
                c = model.behavior_embedding(condition)

                retina_grid_2, cls_dict_2, _ = model.CLIP_model.get_visual_intermediate_layers(image, c=c)

                # 针对于每一层计算dissimilarity
                for layer in ViT_Layers:
                    retina_dissimilarity = 1 - cos(retina_grid_1[str(layer)], retina_grid_2[str(layer)])

                    retina_dissimilarity = torch.mean(retina_dissimilarity, dim=[1, 2])

                    cls_dissimilarity = 1 - cos(cls_dict_1[str(layer)], cls_dict_2[str(layer)])

                    dissimilarity = (retina_dissimilarity + cls_dissimilarity) / 2

                    unmatch_dissimilarity['ViT_layer_' + str(layer)].append(dissimilarity)

        # 整合成tensor
        for layer in ViT_Layers:
            unmatch_dissimilarity['ViT_layer_' + str(layer)] = torch.cat(unmatch_dissimilarity['ViT_layer_' + str(layer)], dim=0)
            unmatch_dissimilarity['ViT_layer_' + str(layer)] = unmatch_dissimilarity['ViT_layer_' + str(layer)].detach().cpu().numpy()
            unmatch_dissimilarity['ViT_layer_' + str(layer)] = zscore(unmatch_dissimilarity['ViT_layer_' + str(layer)])

        # 删除变量减少内存消耗
        del retina_grid_1, retina_grid_2, cls_dict_1, cls_dict_2

        '''
        6. Extract Alexnet semantic similarity
        '''
        print('Extracting Alexnet semantic similarity')
        # 初始化存储地址
        Alexnet_layers = ['features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6',
                        'features.7', 'features.8', 'features.9', 'features.10', 'features.11', 'features.12']
        for layer in Alexnet_layers:
            unmatch_dissimilarity['Alexnet_' + layer] = []

        # Load dataset
        unmatch_dataset = Unmatch_dataset(unmatch_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False, encode_type='visual', type='visual')
        unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        # 初始化norm
        norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)

        model = torch.hub.load('/public/home/lishr2022/Project/Cross-modal/reconstruction/pytorch_vision_v0.10.0', 'alexnet', source='local')
        model = model.to(device, dtype=torch.float)
        ckpt_dir = '/public/home/lishr2022/Project/Cross-modal/reconstruction/pytorch_vision_v0.10.0/checkpoint/alexnet-owt-7be5be79.pth'
        ckpt = torch.load(ckpt_dir, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
            # 针对于unmatch_list
            for layer in Alexnet_layers:
                feature_extractor = create_feature_extractor(model, return_nodes=[layer])

                for index, sample in tqdm(enumerate(unmatch_dataloader), total=len(unmatch_dataloader)):
                    for i in range(len(sample)):
                        if type(sample[i]) == torch.Tensor:
                            sample[i] = sample[i].to(device, dtype=torch.float)

                    # 对于实际看到的Caption, 所对应的Image
                    sample[3] = norm(sample[3])
                    ft1 = feature_extractor(sample[3])
                    ft1 = torch.hstack([torch.flatten(l, start_dim=1) for l in ft1.values()])

                    # 对于实际看到的Image
                    sample[2] = norm(sample[2])
                    ft2 = feature_extractor(sample[2])
                    ft2 = torch.hstack([torch.flatten(l, start_dim=1) for l in ft2.values()])

                    dissimilarity = 1 - cos(ft1, ft2)

                    unmatch_dissimilarity['Alexnet_' + layer].append(dissimilarity)

        for layer in Alexnet_layers:
            unmatch_dissimilarity['Alexnet_' + layer] = torch.cat(unmatch_dissimilarity['Alexnet_' + layer], dim=0)
            unmatch_dissimilarity['Alexnet_' + layer] = unmatch_dissimilarity['Alexnet_' + layer].detach().cpu().numpy()
            unmatch_dissimilarity['Alexnet_' + layer] = zscore(unmatch_dissimilarity['Alexnet_' + layer])

        '''
        7. Extract Resnet18 semantic similarity
        '''
        print('Extracting Resnet18 semantic similarity')
        # 初始化存储地址
        Alexnet_layers = ['layer1.0.relu_1', 'layer1.1.relu_1', 'layer2.0.relu_1', 'layer2.1.relu_1', 'layer3.0.relu_1', 'layer3.1.relu_1','layer4.0.relu_1', 'layer4.1.relu_1']
        for layer in Alexnet_layers:
            unmatch_dissimilarity['Resnet18_' + layer] = []

        # Load dataset
        unmatch_dataset = Unmatch_dataset(unmatch_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False, encode_type='visual', type='visual')
        unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        # 初始化norm
        norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)

        model = torch.hub.load('/public/home/lishr2022/Project/Cross-modal/reconstruction/pytorch_vision_v0.10.0', 'resnet18', source='local')
        model = model.to(device, dtype=torch.float)
        ckpt_dir = '/public/home/lishr2022/Project/Cross-modal/reconstruction/pytorch_vision_v0.10.0/checkpoint/resnet18-f37072fd.pth'
        ckpt = torch.load(ckpt_dir, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
            # 针对于unmatch_list
            for layer in Alexnet_layers:
                feature_extractor = create_feature_extractor(model, return_nodes=[layer])

                for index, sample in tqdm(enumerate(unmatch_dataloader), total=len(unmatch_dataloader)):
                    for i in range(len(sample)):
                        if type(sample[i]) == torch.Tensor:
                            sample[i] = sample[i].to(device, dtype=torch.float)

                    # 对于实际看到的Caption, 所对应的Image
                    sample[3] = norm(sample[3])
                    ft1 = feature_extractor(sample[3])
                    ft1 = torch.hstack([torch.flatten(l, start_dim=1) for l in ft1.values()])

                    # 对于实际看到的Image
                    sample[2] = norm(sample[2])
                    ft2 = feature_extractor(sample[2])
                    ft2 = torch.hstack([torch.flatten(l, start_dim=1) for l in ft2.values()])

                    dissimilarity = 1 - cos(ft1, ft2)

                    unmatch_dissimilarity['Resnet18_' + layer].append(dissimilarity)

        for layer in Alexnet_layers:
            unmatch_dissimilarity['Resnet18_' + layer] = torch.cat(unmatch_dissimilarity['Resnet18_' + layer], dim=0)
            unmatch_dissimilarity['Resnet18_' + layer] = unmatch_dissimilarity['Resnet18_' + layer].detach().cpu().numpy()
            unmatch_dissimilarity['Resnet18_' + layer] = zscore(unmatch_dissimilarity['Resnet18_' + layer])

        '''
        8. Extract Resnet50 semantic similarity
        '''
        print('Extracting Resnet50 semantic similarity')
        # 初始化存储地址
        Alexnet_layers = ['layer1.0.relu_2', 'layer1.1.relu_2', 'layer1.2.relu_2',
                          'layer2.0.relu_2', 'layer2.1.relu_2', 'layer2.2.relu_2',
                          'layer3.0.relu_2', 'layer3.1.relu_2', 'layer3.2.relu_2',
                          'layer4.0.relu_2', 'layer4.1.relu_2', 'layer4.2.relu_2']
        for layer in Alexnet_layers:
            unmatch_dissimilarity['Resnet50_' + layer] = []

        # Load dataset
        unmatch_dataset = Unmatch_dataset(unmatch_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False, encode_type='visual', type='visual')
        unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False,
                                             num_workers=args.num_workers, drop_last=False)

        # 初始化norm
        norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)

        model = torch.hub.load('/public/home/lishr2022/Project/Cross-modal/reconstruction/pytorch_vision_v0.10.0', 'resnet50', source='local')
        model = model.to(device, dtype=torch.float)
        ckpt_dir = '/public/home/lishr2022/Project/Cross-modal/reconstruction/pytorch_vision_v0.10.0/checkpoint/resnet50-0676ba61.pth'
        ckpt = torch.load(ckpt_dir, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
            # 针对于unmatch_list
            for layer in Alexnet_layers:
                feature_extractor = create_feature_extractor(model, return_nodes=[layer])

                for index, sample in tqdm(enumerate(unmatch_dataloader), total=len(unmatch_dataloader)):
                    for i in range(len(sample)):
                        if type(sample[i]) == torch.Tensor:
                            sample[i] = sample[i].to(device, dtype=torch.float)

                    # 对于实际看到的Caption, 所对应的Image
                    sample[3] = norm(sample[3])
                    ft1 = feature_extractor(sample[3])
                    ft1 = torch.hstack([torch.flatten(l, start_dim=1) for l in ft1.values()])

                    # 对于实际看到的Image
                    sample[2] = norm(sample[2])
                    ft2 = feature_extractor(sample[2])
                    ft2 = torch.hstack([torch.flatten(l, start_dim=1) for l in ft2.values()])

                    dissimilarity = 1 - cos(ft1, ft2)

                    unmatch_dissimilarity['Resnet50_' + layer].append(dissimilarity)

        for layer in Alexnet_layers:
            unmatch_dissimilarity['Resnet50_' + layer] = torch.cat(unmatch_dissimilarity['Resnet50_' + layer], dim=0)
            unmatch_dissimilarity['Resnet50_' + layer] = unmatch_dissimilarity['Resnet50_' + layer].detach().cpu().numpy()
            unmatch_dissimilarity['Resnet50_' + layer] = zscore(unmatch_dissimilarity['Resnet50_' + layer])

        df = pd.DataFrame(unmatch_dissimilarity)
        savedir = os.path.join(args.output_dir, 'unmatch_dissimilarity.csv')
        df.to_csv(savedir, index=False)
        print('Dictionary saved to CSV file successfully.')

        del unmatch_dissimilarity

    # if not os.path.exists(os.path.join(args.output_dir, 'match_dissimilarity.csv')):
    #     match_dissimilarity = {}
    #     '''
    #     1. Extract CLIP semantic similarity
    #     '''
    #     print('Extracting CLIP semantic similarity')
    #     match_dissimilarity['CLIP_model'] = []
    #
    #     # Load dataset
    #     unmatch_dataset = Unmatch_dataset(match_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=True, encode_type='cross', type='visual')
    #     unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    #
    #     model, _ = load_from_name('ViT-H-14', device=device, download_root=args.CLIP_model_root)
    #     model = model.to(device, dtype=torch.float)
    #     model.eval()
    #
    #     with torch.no_grad():
    #         cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    #         # 针对于unmatch_list
    #         for index, sample in tqdm(enumerate(unmatch_dataloader), total=len(unmatch_dataloader)):
    #             for i in range(len(sample)):
    #                 if type(sample[i]) == torch.Tensor:
    #                     sample[i] = sample[i].to(device, dtype=torch.float)
    #
    #             # 对于实际看到的caption
    #             inputs = clip.tokenize(sample[3]).to(device)
    #             text_features = model.encode_text(inputs)
    #             text_features /= text_features.norm(dim=-1, keepdim=True)
    #
    #             # 对于实际看到的Image
    #             image_features = model.encode_image(sample[2])
    #             image_features /= image_features.norm(dim=-1, keepdim=True)
    #
    #             dissimilarity = 1 - cos(text_features, image_features)
    #
    #             match_dissimilarity['CLIP_model'].append(dissimilarity)
    #
    #     # 整合成tensor
    #     match_dissimilarity['CLIP_model'] = torch.cat(match_dissimilarity['CLIP_model'], dim=0)
    #     match_dissimilarity['CLIP_model'] = match_dissimilarity['CLIP_model'].detach().cpu().numpy()
    #     match_dissimilarity['CLIP_model'] = zscore(match_dissimilarity['CLIP_model'])
    #
    #     # 删除变量减少内存消耗
    #     del text_features, image_features
    #
    #     df = pd.DataFrame(match_dissimilarity)
    #     savedir = os.path.join(args.output_dir, 'match_dissimilarity.csv')
    #     df.to_csv(savedir, index=False)
    #     print('Match Dictionary saved to CSV file successfully.')

    # if not os.path.exists(os.path.join(args.output_dir, args.subject + '_multivariate_dissimilarity_lh.csv')):
    #     '''
    #     多元线性回归
    #     '''
    #     beta = np.loadtxt(os.path.join(args.output_dir, 'beta.csv'), delimiter=',')
    #     for i in range(beta.shape[1]):
    #         beta[:, i] = zscore(beta[:, i])
    #     unmatch_dissimilarity = pd.read_csv(os.path.join(args.output_dir, 'unmatch_dissimilarity.csv'))
    #
    #     # 剔除一部分变量，避免多重共线性
    #     num_cores = multiprocessing.cpu_count()
    #     label = unmatch_dissimilarity.keys()
    #     vif_data = pd.DataFrame()
    #     vif_data["feature"] = unmatch_dissimilarity[label].columns
    #     vif_data["VIF"] = [variance_inflation_factor(unmatch_dissimilarity[label].values, i) for i in range(len(unmatch_dissimilarity[label].columns))]
    #     temp = []
    #     while max(vif_data['VIF']) >= 8:
    #         max_index = 0
    #         for index, layer in enumerate(label):
    #             if vif_data['VIF'][index] > vif_data['VIF'][max_index]:
    #                 max_index = index
    #         temp.append(label[max_index])
    #
    #         label = []
    #         for index, layer in enumerate(unmatch_dissimilarity.keys()):
    #             if layer not in temp:
    #                 label.append(layer)
    #
    #         vif_data = pd.DataFrame()
    #         vif_data["feature"] = unmatch_dissimilarity[label].columns
    #         vif_data["VIF"] = Parallel(n_jobs=num_cores)(delayed(calculate_vif)(unmatch_dissimilarity[label], i) for i in range(len(unmatch_dissimilarity[label].columns)))
    #
    #     VIF = vif_data['VIF']
    #     count_layer(VIF, label, 'Feature index', 'VIF', 'Multicollinearity analysis for '+args.subject[:2] , os.path.join(args.output_dir, 'VIF.png'))
    #
    #     # 计算多元线性回归
    #     beta_results = np.zeros((beta.shape[1], len(label), 4))
    #     model_performance = np.zeros((args.num_vertices, 3))
    #     for i in tqdm(range(beta.shape[1]), total=beta.shape[1]):
    #         # 全特征 多元线性回归
    #         X = unmatch_dissimilarity
    #         X = sm.add_constant(X)
    #         Y = pd.DataFrame(beta[:, i], columns=['Y'])
    #         model = sm.OLS(Y, X).fit()
    #         for index, layer in enumerate(label):
    #             coef = model.params[layer]
    #             std_err = model.bse[layer]
    #             t_stat = model.tvalues[layer]
    #             p_value = model.pvalues[layer]
    #
    #             beta_results[i, index, 0] = coef
    #             beta_results[i, index, 1] = std_err
    #             beta_results[i, index, 2] = t_stat
    #             beta_results[i, index, 3] = p_value
    #
    #         model_performance[ROI_index[i], 0] = model.rsquared
    #         model_performance[ROI_index[i], 1] = model.fvalue
    #         model_performance[ROI_index[i], 2] = model.f_pvalue
    #
    #     # 统计显著
    #     count = np.zeros(len(label))
    #     onehot_results = np.zeros((args.num_vertices, len(label)))
    #     for i in tqdm(range(beta.shape[1]), total=beta.shape[1]):
    #         for index, layer in enumerate(label):
    #             if beta_results[i, index, 3] <= 0.05:
    #                 count[index] += 1
    #                 if beta_results[i, index, 0] > 0:
    #                     onehot_results[ROI_index[i], index] = 1
    #                 else:
    #                     onehot_results[ROI_index[i], index] = -1
    #     count = count / beta.shape[1]
    #
    #     count_layer(count, label, 'Feature index', 'Proportion of significant voxels', 'Multivariate linear regression for '+args.subject[:2], os.path.join(args.output_dir, 'multiple_count.png'))
    #
    #     with open(os.path.join(args.output_dir, 'multiple_count.txt'), 'w', encoding='gbk') as f:
    #         for index, layer in enumerate(label):
    #             f.write('%s\t%d\n' % (layer, count[index]))
    #     f.close()
    #
    #     vis_lh = {}
    #     vis_rh = {}
    #
    #     # save model performance
    #     vis_lh['R2'] = model_performance[:hemi_vertices[0], 0]
    #     vis_lh['fvalue'] = model_performance[:hemi_vertices[0], 1]
    #     vis_lh['f_pvalue'] = model_performance[:hemi_vertices[0], 2]
    #
    #     vis_rh['R2'] = model_performance[hemi_vertices[0]:, 0]
    #     vis_rh['fvalue'] = model_performance[hemi_vertices[0]:, 1]
    #     vis_rh['f_pvalue'] = model_performance[hemi_vertices[0]:, 2]
    #
    #     # save one hot results
    #     for index, layer in enumerate(label):
    #         vis_lh[layer] = onehot_results[:hemi_vertices[0], index]
    #         vis_rh[layer] = onehot_results[hemi_vertices[0]:, index]
    #
    #     savedir = os.path.join(args.output_dir, args.subject + '_multivariate_dissimilarity_lh.csv')
    #     df = pd.DataFrame(vis_lh)
    #     df.to_csv(savedir, index=False)
    #
    #     savedir = os.path.join(args.output_dir, args.subject + '_multivariate_dissimilarity_rh.csv')
    #     df = pd.DataFrame(vis_rh)
    #     df.to_csv(savedir, index=False)

    if not os.path.exists(os.path.join(args.output_dir, args.subject + '_univariate_dissimilarity_R2_lh.csv')):
        '''
        每个Layer做单变量线性回归
        '''
        beta = np.loadtxt(os.path.join(args.output_dir, 'beta.csv'), delimiter=',')
        # for i in range(beta.shape[1]):
        #     beta[:, i] = zscore(beta[:, i])
        unmatch_dissimilarity = pd.read_csv(os.path.join(args.output_dir, 'unmatch_dissimilarity.csv'))

        label = unmatch_dissimilarity.keys()
        beta_results = np.zeros((beta.shape[1], len(label), 10))
        resid = np.zeros((betas.shape[1], len(label), betas.shape[0]))
        # slope slope_std_err t_stat p_value intercept intercept_std_err t_stat p_value
        for i in tqdm(range(beta.shape[1]), total=beta.shape[1]):
            Y = pd.DataFrame(beta[:, i], columns=['Y'])
            for index, layer in enumerate(label):
                X = unmatch_dissimilarity[layer]
                X = sm.add_constant(X)

                model = sm.OLS(Y, X).fit()

                beta_results[i, index, 0] = model.params[layer]
                beta_results[i, index, 1] = model.bse[layer]
                beta_results[i, index, 2] = model.tvalues[layer]
                beta_results[i, index, 3] = model.pvalues[layer]
                beta_results[i, index, 4] = model.params['const']
                beta_results[i, index, 5] = model.bse['const']
                beta_results[i, index, 6] = model.tvalues['const']
                beta_results[i, index, 7] = model.pvalues['const']
                beta_results[i, index, 8] = model.rsquared
                beta_results[i, index, 9] = model.f_pvalue

                resid[i, index, :] = model.resid

        np.save(os.path.join(args.output_dir, 'beta_results.npy'), beta_results)
        np.save(os.path.join(args.output_dir, 'resid.npy'), resid)

        # 统计显著
        count = np.zeros(len(label))
        onehot_results = np.zeros((args.num_vertices, len(label)))
        R2_results = np.zeros((args.num_vertices, len(label)))
        for i in tqdm(range(beta.shape[1]), total=beta.shape[1]):
            for index, layer in enumerate(label):
                if beta_results[i, index, 9] <= 0.05:
                    count[index] += 1
                    if beta_results[i, index, 0] > 0:
                        onehot_results[ROI_index[i], index] = 1
                        R2_results[ROI_index[i], index] = beta_results[i, index, 8]
                    else:
                        onehot_results[ROI_index[i], index] = -1
                        R2_results[ROI_index[i], index] = -beta_results[i, index, 8]
        count = count / beta.shape[1]

        count_layer(count, label, 'Feature index', 'Proportion of significant voxels', 'Univariate linear regression for '+args.subject[:2], os.path.join(args.output_dir, 'single_count.png'))

        with open(os.path.join(args.output_dir, 'single_count.txt'), 'w', encoding='gbk') as f:
            for index, layer in enumerate(label):
                f.write('%s\t%d\n' % (layer, count[index]))
        f.close()

        vis_lh = {}
        vis_rh = {}
        for index, layer in enumerate(label):
            vis_lh[layer] = onehot_results[:hemi_vertices[0], index]
            vis_rh[layer] = onehot_results[hemi_vertices[0]:, index]

        savedir = os.path.join(args.output_dir, args.subject + '_univariate_dissimilarity_lh.csv')
        df = pd.DataFrame(vis_lh)
        df.to_csv(savedir, index=False)

        savedir = os.path.join(args.output_dir, args.subject + '_univariate_dissimilarity_rh.csv')
        df = pd.DataFrame(vis_rh)
        df.to_csv(savedir, index=False)

        R2_lh = {}
        R2_rh = {}
        for index, layer in enumerate(label):
            R2_lh[layer] = R2_results[:hemi_vertices[0], index]
            R2_rh[layer] = R2_results[hemi_vertices[0]:, index]

        savedir = os.path.join(args.output_dir, args.subject + '_univariate_dissimilarity_R2_lh.csv')
        df = pd.DataFrame(R2_lh)
        df.to_csv(savedir, index=False)

        savedir = os.path.join(args.output_dir, args.subject + '_univariate_dissimilarity_R2_rh.csv')
        df = pd.DataFrame(R2_rh)
        df.to_csv(savedir, index=False)

    # if not os.path.exists(os.path.join(args.output_dir, args.subject + '_match_dissimilarity_lh.csv')):
    #     beta = np.loadtxt(os.path.join(args.output_dir, 'match_beta.csv'), delimiter=',')
    #     for i in range(beta.shape[1]):
    #         beta[:, i] = zscore(beta[:, i])
    #     match_dissimilarity = pd.read_csv(os.path.join(args.output_dir, 'match_dissimilarity.csv'))
    #
    #     label = match_dissimilarity.keys()
    #     beta_results = np.zeros((beta.shape[1], len(label), 8))
    #     # slope slope_std_err t_stat p_value intercept intercept_std_err t_stat p_value
    #     for i in tqdm(range(beta.shape[1]), total=beta.shape[1]):
    #         Y = pd.DataFrame(beta[:, i], columns=['Y'])
    #         for index, layer in enumerate(label):
    #             X = match_dissimilarity[layer]
    #             X = sm.add_constant(X)
    #
    #             model = sm.OLS(Y, X).fit()
    #
    #             beta_results[i, index, 0] = model.params[layer]
    #             beta_results[i, index, 1] = model.bse[layer]
    #             beta_results[i, index, 2] = model.tvalues[layer]
    #             beta_results[i, index, 3] = model.pvalues[layer]
    #             beta_results[i, index, 4] = model.params['const']
    #             beta_results[i, index, 5] = model.bse['const']
    #             beta_results[i, index, 6] = model.tvalues['const']
    #             beta_results[i, index, 7] = model.pvalues['const']
    #
    #     # 统计显著
    #     count = np.zeros(len(label))
    #     onehot_results = np.zeros((args.num_vertices, len(label)))
    #     for i in tqdm(range(beta.shape[1]), total=beta.shape[1]):
    #         for index, layer in enumerate(label):
    #             if beta_results[i, index, 3] <= 0.05:
    #                 count[index] += 1
    #                 if beta_results[i, index, 0] > 0:
    #                     onehot_results[ROI_index[i], index] = 1
    #                 else:
    #                     onehot_results[ROI_index[i], index] = -1
    #     print(count)
    #
    #     vis_lh = {}
    #     vis_rh = {}
    #     for index, layer in enumerate(label):
    #         vis_lh[layer] = onehot_results[:hemi_vertices[0], index]
    #         vis_rh[layer] = onehot_results[hemi_vertices[0]:, index]
    #
    #     savedir = os.path.join(args.output_dir, args.subject + '_match_dissimilarity_lh.csv')
    #     df = pd.DataFrame(vis_lh)
    #     df.to_csv(savedir, index=False)
    #
    #     savedir = os.path.join(args.output_dir, args.subject + '_match_dissimilarity_rh.csv')
    #     df = pd.DataFrame(vis_rh)
    #     df.to_csv(savedir, index=False)

