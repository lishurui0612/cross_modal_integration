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
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from einops import rearrange, repeat
from info_nce import InfoNCE, info_nce
from scipy.interpolate import griddata
from timm.layers.norm import LayerNorm2d
from einops.layers.torch import Rearrange
from timm.models.convnext import ConvNeXtBlock
from sklearn.metrics import calinski_harabasz_score
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Any, Dict, List, Optional, Tuple, Union
from cn_clip.clip import load_from_name, image_transform, load_from_name, _tokenizer
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, R2Score, MetricCollection

from dataset import prepare_pair_data, Caption_Image_dataset, Caption_Image_2d_dataset, natural_scene_dataset, \
    SingleTrial_dataset
from model_cae import BrainCLIP
from utils import get_parameter_number, cosine_sheduler, ExponentialMovingAverage


def add_argparse_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--Stimulus_index_root', type=str,
                        default='/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt')
    parser.add_argument('--Image_Caption_pairs_root', type=str,
                        default='/public_bme/data/lishr/COCO_CN/Selected_Match_Image_Caption_pairs.txt')
    # parser.add_argument('--processed_root', type=str, default='/public_bme/data/lishr/Cross_modal/Processed_Data/')
    parser.add_argument('--processed_root', type=str,
                        default='/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data')
    parser.add_argument('--image_root', type=str, default='/public_bme/data/lishr/COCO_CN/All_images_480')
    parser.add_argument('--CLIP_model_root', type=str,
                        default='/public/home/lishr2022/Project/Cross-modal/encoding/cross_encoding/model_cache')
    parser.add_argument('--nsd_root', type=str, default='/public_bme/data/lishr/NSD/nsddata_betas/ppdata')
    parser.add_argument('--output_dir', type=str,
                        default='/public/home/lishr2022/Project/Cross-modal/reconstruction/S1_LiJiawei_BrainCLIP_none_linear_1e-3_2stage')
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
    parser.add_argument('--type', type=str, default='linear')
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

    parser.add_argument('--subject', type=str, default='S1_LiJiawei')
    parser.add_argument('--times', type=int, default=10)

    return parser


def RSA(raw, feature, dim, args, raw2=None, feature2=None):
    assert len(raw) == len(feature), 'Something Wrong!'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    times = args.times
    num_points = len(feature)
    R_cos = np.zeros(args.num_vertices)
    R_dist = np.zeros(args.num_vertices)
    for t in tqdm(range(times)):
        raw_rsa = np.zeros((args.num_vertices, dim))
        feature_cos = np.zeros(dim)
        feature_dist = np.zeros(dim)
        for i in range(dim):
            # random choose 2 points
            id1 = random.randint(0, num_points - 1)
            id2 = random.randint(0, num_points - 1)

            if raw2 is None:
                raw_rsa[:, i] = raw[id1] - raw[id2]
            else:
                raw_rsa[:, i] = raw[id1] - raw2[id2]

            if feature2 is None:
                feature_cos[i] = -feature[id1] @ feature[id2].T
                feature_dist[i] = np.linalg.norm(feature[id1] - feature[id2])
            else:
                feature_cos[i] = -feature[id1] @ feature2[id2].T
                feature_dist[i] = np.linalg.norm(feature[id1] - feature2[id2])

        pearson = PearsonCorrCoef(num_outputs=args.num_vertices).to(device, dtype=torch.float)
        raw_rsa = torch.from_numpy(raw_rsa).to(device, dtype=torch.float)

        feature_cos = torch.unsqueeze(torch.from_numpy(feature_cos).to(device, dtype=torch.float), dim=0)
        feature_cos = repeat(feature_cos, '1 D -> N D', N=args.num_vertices)

        feature_dist = torch.unsqueeze(torch.from_numpy(feature_dist).to(device, dtype=torch.float), dim=0)
        feature_dist = repeat(feature_dist, '1 D -> N D', N=args.num_vertices)

        R_cos_temp = pearson(raw_rsa.T, feature_cos.T).detach().cpu().numpy()
        R_dist_temp = pearson(raw_rsa.T, feature_dist.T).detach().cpu().numpy()

        R_cos += R_cos_temp
        R_dist += R_dist_temp

    R_cos /= times
    R_dist /= times
    return R_cos, R_dist


def grid2fMRI(data, azimuth, elevation, img_size):
    data = data.reshape(-1)

    X, Y = np.meshgrid(np.linspace(-1, 1, img_size), np.linspace(-1, 1, img_size))
    X = X.reshape(-1)
    Y = Y.reshape(-1)

    elevation = np.sin(elevation)
    elevation = 2 * elevation / (np.max(elevation) - np.min(elevation))
    azimuth = 2 * azimuth / (np.max(azimuth) - np.min(azimuth))

    transformed_gridmap = griddata((X, Y), data, (azimuth, elevation), method='nearest')

    return transformed_gridmap


def plot_weight(stat, subject, hemi_vertices, savedir, norm=True):
    fig = plt.figure(figsize=(24, 8))
    axes = []
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        axes.append(ax)

    vmax = np.max(abs(stat))
    if norm:
        stat = stat / vmax
        vmax = 1

    view = ['lateral', 'ventral', 'medial', 'dorsal']

    left_surf_mesh = os.path.join('/public_bme/data/lishr/Cross_modal/subjects', subject[3:], 'surf', 'lh.inflated')
    right_surf_mesh = os.path.join('/public_bme/data/lishr/Cross_modal/subjects', subject[3:], 'surf', 'rh.inflated')

    left_stat = stat[:hemi_vertices[0]]
    right_stat = stat[hemi_vertices[0]:]
    # plot left hemisphere
    print('- - - plotting left hemisphere')
    for i in range(4):
        plotting.plot_surf_stat_map(
            surf_mesh=left_surf_mesh,
            stat_map=left_stat,
            hemi='left',
            view=view[i],
            vmax=vmax,
            axes=axes[i],
            title='Left - ' + view[i]
        )
    # plot right hemisphere
    print('- - - plotting right hemisphere')
    for i in range(4):
        plotting.plot_surf_stat_map(
            surf_mesh=right_surf_mesh,
            stat_map=right_stat,
            hemi='right',
            view=view[i],
            vmax=vmax,
            axes=axes[i + 4],
            title='Right - ' + view[i]
        )

    fig.savefig(savedir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_argparse_args(parser)
    args = parser.parse_args()

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

    args.coords_root = os.path.join(args.processed_root, args.subject, 'sph_coords.npy')
    sph_coords = np.load(args.coords_root)
    hemi_vertices = [args.lh_vertices, args.rh_vertices]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load CLIP model
    CLIP_model, _ = load_from_name("ViT-H-14", device=device, download_root=args.CLIP_model_root)
    # load pretrain model
    model_root = os.path.join(args.output_dir, 'opt_model.pth')
    ckpt = torch.load(model_root, map_location=device)
    model = BrainCLIP(
        vertices=hemi_vertices,
        sph_coords=sph_coords,
        img_size=args.image_size,
        depth=args.depth,
        type=args.type,
        embed_dim=args.embed_dim,
        post_type=args.post_type,
        Index_root=args.Index_root
    )
    model.load_state_dict(ckpt)
    model = model.to(device, dtype=torch.float)

    torch.cuda.empty_cache()
    model.eval()
    CLIP_model.eval()

    transform = image_transform()
    Stimulus_index, Stimulus_pairs, stim_dict, train, val = prepare_pair_data(args.subject,
                                                                              args.Image_Caption_pairs_root,
                                                                              args.Stimulus_index_root,
                                                                              args.processed_root,
                                                                              data_type=args.data_type)
    datadir = os.path.join(args.processed_root, args.subject, 'Stimulus', args.data_type)
    FileList = sorted(os.listdir(datadir))

    image_list = []
    caption_list = []
    for file in FileList:
        if file[-5] != 'a':
            stim = int(file[-24:-19])
        else:
            stim = int(file[-22:-17])
        if Stimulus_index[stim][-3:] == 'jpg':
            image_list.append(os.path.join(datadir, file))
        else:
            caption_list.append(os.path.join(datadir, file))
    image_dataset = SingleTrial_dataset(image_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root)
    caption_dataset = SingleTrial_dataset(caption_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root)
    image_dataloader = data.DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    caption_dataloader = data.DataLoader(caption_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    train_list = []
    val_list = []
    for file in train:
        if file[-5] != 'a':
            stim = int(file[-24:-19])
        else:
            stim = int(file[-22:-17])
        if Stimulus_index[stim][-3:] == 'jpg':
            train_list.append(stim)
        else:
            train_list.append(Stimulus_index[Stimulus_pairs[Stimulus_index[stim]]])
    for file in val:
        if file[-5] != 'a':
            stim = int(file[-24:-19])
        else:
            stim = int(file[-22:-17])
        if Stimulus_index[stim][-3:] == 'jpg':
            val_list.append(stim)
        else:
            val_list.append(Stimulus_index[Stimulus_pairs[Stimulus_index[stim]]])
    
    raw = []
    brain_feature = []
    clip_feature = []

    caption_raw = []
    image_raw = []
    brain_caption_feature = []
    brain_caption_label = []
    brain_image_feature = []
    brain_image_label = []
    with torch.no_grad():
        for index, sample in tqdm(enumerate(image_dataloader), total=len(image_dataloader)):
            for i in range(1, len(sample)):
                sample[i] = sample[i].to(device, dtype=torch.float)
                
            image_feature, image_feature_clip = model(sample[2])
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            image_feature = image_feature.detach().cpu().numpy()
            brain_feature.append(image_feature)

            raw.append(sample[2].detach().cpu().numpy())
            image_raw.append(sample[2].detach().cpu().numpy())
            brain_image_feature.append(image_feature)
            brain_image_label.append(sample[0])

            feature = CLIP_model.encode_image(sample[1]).to(torch.float)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            feature = feature.detach().cpu().numpy()
            clip_feature.append(feature)

        for index, sample in tqdm(enumerate(caption_dataloader), total=len(caption_dataloader)):
            for i in range(2, len(sample)):
                sample[i] = sample[i].to(device, dtype=torch.float)

            caption_feature, caption_feature_clip = model(sample[2])
            caption_feature = caption_feature / caption_feature.norm(dim=-1, keepdim=True)
            caption_feature = caption_feature.detach().cpu().numpy()
            brain_feature.append(caption_feature)

            raw.append(sample[2].detach().cpu().numpy())
            caption_raw.append(sample[2].detach().cpu().numpy())
            brain_caption_feature.append(caption_feature)
            brain_caption_label.append(sample[0])

            inputs = clip.tokenize(sample[1]).to(device)
            feature = CLIP_model.encode_text(inputs).to(torch.float)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            feature = feature.detach().cpu().numpy()
            clip_feature.append(feature)

    raw = np.concatenate(raw, axis=0)
    caption_raw = np.concatenate(caption_raw, axis=0)
    image_raw = np.concatenate(image_raw, axis=0)

    brain_feature = np.concatenate(brain_feature, axis=0)
    clip_feature = np.concatenate(clip_feature, axis=0)

    brain_caption_feature = np.concatenate(brain_caption_feature, axis=0)
    brain_image_feature = np.concatenate(brain_image_feature, axis=0)

    brain_image_label = np.concatenate(brain_image_label, axis=0)
    brain_caption_label = np.concatenate(brain_caption_label, axis=0)

    # Cluster
    kmeans = KMeans(n_clusters=10, random_state=1234)
    kmeans.fit(clip_feature)
    label = kmeans.predict(clip_feature)

    colors = ['#f06868', '#fab57a', '#edf798', '#80d6ff', '#0f1021', '#d01257', '#fb90b7', '#ffcee4', '#5588a3',
              '#00c9b1']
    # Visualization CLIP semantic space
    if not os.path.exists(os.path.join(args.output_dir, 'fig1.png')):
        tsne_clip = TSNE(n_components=2, random_state=1234)
        clip_feature_tsne = tsne_clip.fit_transform(clip_feature)
        fig, ax = plt.subplots(figsize=(1.875, 1.5))
        for c in range(10):
            ax.scatter(clip_feature_tsne[label == c, 0],
                        clip_feature_tsne[label == c, 1],
                        c=colors[c], s=0.1, label='CLIP cluster' + str(c))
        # plt.title('CLIP embedding clusters in CLIP semantic space')
        # plt.legend()
        plt.tick_params(axis='x', which='both', bottom='off', length=3, width=1)
        plt.tick_params(axis='y', which='both', left='off', length=3, width=1)
        plt.xlabel('Component 1', fontsize=7, fontproperties='Arial')
        plt.ylabel('Component 2', fontsize=7, fontproperties='Arial')
        plt.tick_params(axis='x', labelsize=7, labelcolor='black', pad=2)
        plt.tick_params(axis='y', labelsize=7, labelcolor='black', pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(1)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['bottom'].set_color('black')
        fig.savefig(os.path.join(args.output_dir, 'fig1.png'), format="png", dpi=500, bbox_inches="tight")
        plt.close(fig)

    # Visualization Brain semantic space
    if not os.path.exists(os.path.join(args.output_dir, 'fig2.png')):
        tsne_brain = TSNE(n_components=2, random_state=1234)
        brain_feature_tsne = tsne_brain.fit_transform(brain_feature)
        fig, ax = plt.subplots(figsize=(1.875, 1.5))
        for c in range(10):
            ax.scatter(brain_feature_tsne[label == c, 0],
                        brain_feature_tsne[label == c, 1],
                        c=colors[c], s=0.1, label='CLIP cluster' + str(c))
        # plt.title('CLIP embedding clusters in Brain semantic space')
        # plt.legend()
        plt.tick_params(axis='x', which='both', bottom='off', length=3, width=1)
        plt.tick_params(axis='y', which='both', left='off', length=3, width=1)
        plt.xlabel('Component 1', fontsize=7, fontproperties='Arial')
        plt.ylabel('Component 2', fontsize=7, fontproperties='Arial')
        plt.tick_params(axis='x', labelsize=7, labelcolor='black', pad=2)
        plt.tick_params(axis='y', labelsize=7, labelcolor='black', pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(1)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['bottom'].set_color('black')
        fig.savefig(os.path.join(args.output_dir, 'fig2.png'), format="png", dpi=500, bbox_inches="tight")
        plt.close(fig)

    # Visualization Brain raw space
    if not os.path.exists(os.path.join(args.output_dir, 'fig3.png')):
        tsne_raw = TSNE(n_components=2, random_state=1234)
        raw_tsne = tsne_raw.fit_transform(raw)
        fig, ax = plt.subplots(figsize=(1.875, 1.5))
        for c in range(10):
            ax.scatter(raw_tsne[label == c, 0],
                        raw_tsne[label == c, 1],
                        c=colors[c], s=0.1, label='CLIP cluster' + str(c))
        # plt.title('CLIP embedding clusters in Brain raw space')
        # plt.legend()
        plt.tick_params(axis='x', which='both', bottom='off', length=3, width=1)
        plt.tick_params(axis='y', which='both', left='off', length=3, width=1)
        plt.xlabel('Component 1', fontsize=7, fontproperties='Arial')
        plt.ylabel('Component 2', fontsize=7, fontproperties='Arial')
        plt.tick_params(axis='x', labelsize=7, labelcolor='black', pad=2)
        plt.tick_params(axis='y', labelsize=7, labelcolor='black', pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(1)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['bottom'].set_color('black')
        fig.savefig(os.path.join(args.output_dir, 'fig3.png'), format="png", dpi=500, bbox_inches="tight")
        plt.close(fig)

    # Calculate Brain semantic space CH-Score
    ch_score_raw = calinski_harabasz_score(raw, label)
    ch_score = calinski_harabasz_score(brain_feature, label)
    f = open(os.path.join(args.output_dir, 'results.txt'), 'a', encoding='gbk')
    f.write('\nFor total stimulus, CH-Score-raw = %.5f\n' % ch_score_raw)
    f.write('For total stimulus, CH-Score = %.5f\n' % ch_score)
    f.close()

    # # Visualization retrieval performance
    # if not os.path.exists(os.path.join(args.output_dir, 'retrieval_performance_srl.txt')):
    #     top_1 = 0
    #     top_5 = 0
    #     top_10 = 0
    #     count = 0
    #     similarity = brain_image_feature @ brain_caption_feature.T
    #     index = np.argsort(-similarity, axis=1)
    #     f = open(os.path.join(args.output_dir, 'retrieval_performance_srl.txt'), 'w', encoding='gbk')
    #     for i in range(len(brain_image_feature)):
    #         if int(brain_image_label[i]) not in val_list:
    #             continue
    #         count += 1
    #         f.write('%s ' % Stimulus_index[int(brain_image_label[i])])
    #         f.write('%s ' % Stimulus_pairs[Stimulus_index[int(brain_image_label[i])]])
    #         for j in range(10):
    #             f.write('%s ' % Stimulus_index[int(brain_caption_label[index[i, j]])])
    #             if brain_image_label[i] == Stimulus_index[Stimulus_pairs[Stimulus_index[int(brain_caption_label[index[i, j]])]]]:
    #                 if j < 1:
    #                     top_1 += 1
    #                     top_5 += 1
    #                     top_10 += 1
    #                 elif j < 5:
    #                     top_5 += 1
    #                     top_10 += 1
    #                 else:
    #                     top_10 += 1
    #                 break
    #         f.write('\n')
    #     f.close()
    #     top_1 /= count
    #     top_5 /= count
    #     top_10 /= count
    #     print(top_1, top_5, top_10)

    # Representation similarity analysis - Image
    # R_cos, R_dist = RSA(image_raw, brain_image_feature, 1000, args)
    # savedir = os.path.join(args.output_dir, 'fig4_' + str(args.times).zfill(4) + '.png')
    # plot_weight(R_cos, args.subject, hemi_vertices, savedir, norm=False)
    # savedir = os.path.join(args.output_dir, 'fig5_' + str(args.times).zfill(4) + '.png')
    # plot_weight(R_dist, args.subject, hemi_vertices, savedir, norm=False)

    # Representation similarity analysis - Caption
    # R_cos, R_dist = RSA(caption_raw, brain_caption_feature, 1000, args)
    # savedir = os.path.join(args.output_dir, 'fig6_' + str(args.times).zfill(4) + '.png')
    # plot_weight(R_cos, args.subject, hemi_vertices, savedir, norm=False)
    # savedir = os.path.join(args.output_dir, 'fig7_' + str(args.times).zfill(4) + '.png')
    # plot_weight(R_dist, args.subject, hemi_vertices, savedir, norm=False)

    # Representation similarity analysis - Image & Caption
    # R_cos, R_dist = RSA(image_raw, brain_image_feature, 1000, args, raw2=caption_raw, feature2=brain_caption_feature)
    # savedir = os.path.join(args.output_dir, 'fig8_' + str(args.times).zfill(4) + '.png')
    # plot_weight(R_cos, args.subject, hemi_vertices, savedir, norm=False)
    # savedir = os.path.join(args.output_dir, 'fig9_' + str(args.times).zfill(4) + '.png')
    # plot_weight(R_dist, args.subject, hemi_vertices, savedir, norm=False)


    # # Test each voxel performance
    # if not os.path.exists(os.path.join(args.output_dir, 'test_performance.npy')):
    #     performance = np.zeros((28, 14, 3))
    #     for i_index in range(28):
    #         for j_index in tqdm(range(14)):
    #             brain_caption_feature = []
    #             brain_caption_label = []
    #             brain_image_feature = []
    #             brain_image_label = []
    #             with torch.no_grad():
    #                 for index, sample in enumerate(image_dataloader):
    #                     for i in range(1, len(sample)):
    #                         sample[i] = sample[i].to(device, dtype=torch.float)
    #
    #                     image_feature, image_feature_clip = model.forward_test(sample[3], i_index, j_index)
    #                     image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    #                     image_feature = image_feature.detach().cpu().numpy()
    #
    #                     brain_image_feature.append(image_feature)
    #                     brain_image_label.append(sample[0].detach().cpu().numpy())
    #
    #                 for index, sample in enumerate(caption_dataloader):
    #                     for i in range(2, len(sample)):
    #                         sample[i] = sample[i].to(device, dtype=torch.float)
    #
    #                     caption_feature, caption_feature_clip = model.forward_test(sample[3], i_index, j_index)
    #                     caption_feature = caption_feature / caption_feature.norm(dim=-1, keepdim=True)
    #                     caption_feature = caption_feature.detach().cpu().numpy()
    #
    #                     brain_caption_feature.append(caption_feature)
    #                     brain_caption_label.append(sample[0].detach().cpu().numpy())
    #
    #                 brain_image_feature = np.concatenate(brain_image_feature, axis=0)
    #                 brain_image_label = np.concatenate(brain_image_label, axis=0)
    #                 brain_caption_feature = np.concatenate(brain_caption_feature, axis=0)
    #                 brain_caption_label = np.concatenate(brain_caption_label, axis=0)
    #                 # calculate performance
    #                 # image beta <-> caption beta
    #                 similarity = brain_image_feature @ brain_caption_feature.T
    #                 index = np.argsort(-similarity, axis=1)
    #                 top_1 = 0
    #                 top_5 = 0
    #                 top_10 = 0
    #                 count = 0
    #                 print(brain_image_label)
    #                 for i in range(len(brain_image_feature)):
    #                     if brain_image_label[i] not in val_list:
    #                         continue
    #                     count += 1
    #                     for j in range(10):
    #                         if brain_image_label[i] == brain_caption_label[index[i, j]]:
    #                             if j < 1:
    #                                 top_1 += 1
    #                                 top_5 += 1
    #                                 top_10 += 1
    #                             elif j < 5:
    #                                 top_5 += 1
    #                                 top_10 += 1
    #                             else:
    #                                 top_10 += 1
    #                             break
    #                 top_1 /= count
    #                 top_5 /= count
    #                 top_10 /= count
    #
    #                 performance[i_index, j_index, 0] = top_1
    #                 performance[i_index, j_index, 1] = top_5
    #                 performance[i_index, j_index, 2] = top_10
    #
    #             print('For i = %d, j = %d, Top_1: %.5f, Top_5: %.5f, Top_10: %.5f' %(i_index, j_index, top_1, top_5, top_10))
    #
    #
    #     savedir = os.path.join(args.output_dir, 'test_performance.npy')
    #     np.save(savedir, performance)

    # 区域对semantic alignment的影响
    # if not os.path.exists(os.path.join(args.output_dir, 'fig10.png')):
    #     performance = np.load(os.path.join(args.output_dir, 'test_performance.npy'))
    #     performance[:, :, 0] = performance[:, :, 0] - 0.54768
    #     performance[:, :, 1] = performance[:, :, 1] - 0.74083
    #     performance[:, :, 2] = performance[:, :, 2] - 0.79707
    #
    #     for i in tqdm(range(3)):
    #         transformed_brain = np.zeros(args.num_vertices)
    #         transformed_brain[:args.lh_vertices] = grid2fMRI(performance[:14, :, i], sph_coords[:args.lh_vertices, 0], sph_coords[:args.lh_vertices, 1], 14)
    #         transformed_brain[args.lh_vertices:] = grid2fMRI(performance[14:, :, i], sph_coords[args.lh_vertices:, 0], sph_coords[args.lh_vertices:, 1], 14)
    #
    #         savedir = os.path.join(args.output_dir, 'fig' + str(i + 10) + '.png')
    #         plot_weight(transformed_brain, args.subject, hemi_vertices, savedir, norm=False)

