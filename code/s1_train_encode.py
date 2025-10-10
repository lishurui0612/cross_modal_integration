# coding=gbk
import os
import gc
import copy
import torch
import random
import logging
import nilearn
import argparse
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from tqdm import tqdm
from scipy import io, stats
import cn_clip.clip as clip
from einops import rearrange
from nilearn import plotting
import torch.utils.data as data
import matplotlib.pyplot as plt
from timm.layers.mlp import Mlp
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers.norm import LayerNorm2d
from timm.models.convnext import ConvNeXtBlock
from typing import Any, Dict, List, Optional, Tuple, Union
from cn_clip.clip import load_from_name, image_transform, load_from_name, _tokenizer
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, R2Score, MetricCollection

from dataset import prepare_pair_data, Caption_Image_dataset, Unmatch_dataset, Predictive_coding_dataset
from model import EncodingModel, PredictiveEncodingModel
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
    parser.add_argument('--output_dir', type=str, default='/public/home/lishr2022/Project/Cross-modal/reconstruction/encoding/test_001')
    parser.add_argument('--vit_model_root', type=str, default=None)
    parser.add_argument('--bert_model_root', type=str, default=None)
    parser.add_argument('--bert_ckpt_dir', type=str, default='/public/home/lishr2022/Project/Cross-modal/reconstruction/S1_LiJiawei_predictive_encoding_vis_ensemble_0_0_64_v4_08/opt_cap_model.pth')
    parser.add_argument('--vit_ckpt_dir', type=str, default='/public/home/lishr2022/Project/Cross-modal/reconstruction/S1_LiJiawei_predictive_encoding_vis_ensemble_0_0_64_v4_12/opt_img_model.pth')
    parser.add_argument('--trial_type', type=str, default='match')

    parser.add_argument('--behavior_in', type=int, default=8)
    parser.add_argument('--behavior_hidden', type=int, default=16)
    parser.add_argument('--final_visual_emb_dim', type=int, default=64)
    parser.add_argument('--final_bert_emb_dim', type=int, default=1024)
    parser.add_argument('--encode_type', type=str, default='visual')
    parser.add_argument('--index', type=int, default=3)
    parser.add_argument('--condition', type=int, default=1)
    parser.add_argument('--roi_type', type=int, default=0)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--start_warmup_value', type=float, default=1e-5)
    parser.add_argument('--ema_interval', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--reg_weight', type=float, default=3e-5)
    parser.add_argument('--pearson_weight', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--subject', type=str)

    return parser


def validation(model, val_dataloader, pearsonr, device, args):
    model.eval()
    val_img_loss = torch.zeros(1).to(device)
    val_caption_loss = torch.zeros(1).to(device)
    with torch.no_grad():
        for step, sample in enumerate(val_dataloader):
            for i in range(len(sample)):
                if type(sample[i]) == torch.Tensor:
                    sample[i] = sample[i].to(device, dtype=torch.float)

            # image encode
            out_img = model.VisualEncode(sample)
            img_mae = 1 - (torch.mean(pearsonr(out_img, sample[3][:, :args.num_vertices])) + 1) / 2
            val_img_loss += img_mae

            # caption encode
            # out_caption = model.BertEncode(sample)
            # caption_mae = mae(out_caption, sample[2][:, :args.num_vertices])
            # val_caption_loss += caption_mae
    val_img_loss /= step
    val_caption_loss /= step
    model.train()
    return val_img_loss, val_caption_loss


def calculateR(model, dataloader, device, args, target_index=None, shuffle=0):
    if target_index is None:
        target_index = args.index

    if args.roi_type == 0:
        # 全脑encoding
        ROI_index = np.arange(args.num_vertices).astype(int)
    elif args.roi_type == 1:
        # V1-V3参与encoding
        ROI_index = np.loadtxt(os.path.join(args.processed_root, args.subject, 'ROI', 'EVC.txt')).astype(int)
    elif args.roi_type == 2:
        ROI_index = np.loadtxt(os.path.join(args.processed_root, args.subject, 'ROI', 'IFS.txt')).astype(int)
    ROI_index = torch.squeeze(torch.from_numpy(ROI_index).to(device, dtype=torch.long))

    pearsonr = PearsonCorrCoef(num_outputs=len(ROI_index)).to(device)
    model.eval()
    out = []
    target = []
    with torch.no_grad():
        for step, sample in enumerate(dataloader):
            for i in range(len(sample)):
                if type(sample[i]) == torch.Tensor:
                    sample[i] = sample[i].to(device, dtype=torch.float)

            if args.condition == 0:
                sample[4] = torch.zeros_like(sample[4]).to(device, dtype=torch.float)
                sample[5] = torch.zeros_like(sample[5]).to(device, dtype=torch.float)
            if shuffle == 1:
                sample[0] = list(sample[0])
                random.shuffle(sample[0])
                sample[0] = tuple(sample[0])

                random_indices = torch.randperm(args.val_batch_size)
                sample[1] = sample[1][random_indices]
            elif shuffle == 2:
                sample[0] = list(sample[0])
                for i in range(len(sample[0])):
                    sample[0][i] = list(sample[0][i])
                    random.shuffle(sample[0][i])
                    sample[0][i] = ''.join(sample[0][i])
                sample[0] = tuple(sample[0])

                random_indices = torch.randperm(args.val_batch_size)
                sample[1] = sample[1][random_indices]

            if args.encode_type == 'visual':
                temp = model.VisualEncode(sample, args.index)
                # out.append(temp)
                # target.append(sample[3])
                temp = temp.permute(1, 0)
                temp = temp[ROI_index]
                temp = temp.permute(1, 0)

                out.append(temp)

                sample[target_index] = sample[target_index].permute(1, 0)
                sample[target_index] = sample[target_index][ROI_index]
                sample[target_index] = sample[target_index].permute(1, 0)
                target.append(sample[target_index])
            elif args.encode_type == 'caption':
                temp = model.BertEncode(sample, args.index)
                # out.append(temp)
                # target.append(sample[3])
                temp = temp.permute(1, 0)
                temp = temp[ROI_index]
                temp = temp.permute(1, 0)

                out.append(temp)

                sample[target_index] = sample[target_index].permute(1, 0)
                sample[target_index] = sample[target_index][ROI_index]
                sample[target_index] = sample[target_index].permute(1, 0)
                target.append(sample[target_index])
            else:
                temp = model.predictive_coding(sample)
                temp = temp.permute(1, 0)
                temp = temp[ROI_index]
                temp = temp.permute(1, 0)

                out.append(temp)

                sample[target_index] = sample[target_index].permute(1, 0)
                sample[target_index] = sample[target_index][ROI_index]
                sample[target_index] = sample[target_index].permute(1, 0)
                target.append(sample[target_index])

    out = torch.cat(out, dim=0)
    target = torch.cat(target, dim=0)
    pearson = pearsonr(out, target)

    model.train()
    return pearson


def calculateR_unmatch(model, dataloader, device, args, shuffle=0):
    target_index = 4
    if args.roi_type == 0:
        # 全脑encoding
        ROI_index = np.arange(args.num_vertices).astype(int)
    elif args.roi_type == 1:
        # V1-V3参与encoding
        ROI_index = np.loadtxt(os.path.join(args.processed_root, args.subject, 'ROI', 'EVC.txt')).astype(int)
    elif args.roi_type == 2:
        ROI_index = np.loadtxt(os.path.join(args.processed_root, args.subject, 'ROI', 'IFS.txt')).astype(int)
    ROI_index = torch.squeeze(torch.from_numpy(ROI_index).to(device, dtype=torch.long))

    pearsonr = PearsonCorrCoef(num_outputs=len(ROI_index)).to(device)
    model.eval()
    out_match = []
    out_unmatch = []
    target = []
    with torch.no_grad():
        for step, sample in enumerate(dataloader):
            for i in range(len(sample)):
                if type(sample[i]) == torch.Tensor:
                    sample[i] = sample[i].to(device, dtype=torch.float)

            if args.condition == 0:
                sample[5] = torch.zeros_like(sample[5]).to(device, dtype=torch.float)
            if shuffle == 1:
                if args.encode_type == 'caption':
                    sample[2] = list(sample[2])
                    random.shuffle(sample[2])
                    sample[2] = tuple(sample[2])

                    sample[3] = list(sample[3])
                    random.shuffle(sample[3])
                    sample[3] = tuple(sample[3])
                else:
                    random_indices = torch.randperm(args.val_batch_size)
                    sample[2] = sample[2][random_indices]

                    random_indices = torch.randperm(args.val_batch_size)
                    sample[3] = sample[3][random_indices]
            elif shuffle == 2:
                if args.encode_type == 'caption':
                    sample[2] = list(sample[2])
                    for i in range(len(sample[2])):
                        sample[2][i] = list(sample[2][i])
                        random.shuffle(sample[2][i])
                        sample[2][i] = ''.join(sample[2][i])
                    sample[2] = tuple(sample[2])

                    sample[3] = list(sample[3])
                    for i in range(len(sample[3])):
                        sample[3][i] = list(sample[3][i])
                        random.shuffle(sample[3][i])
                        sample[3][i] = ''.join(sample[3][i])
                    sample[3] = tuple(sample[3])

            if args.encode_type == 'visual':
                inputs = [None, sample[2], None, None, None, sample[5]]
                temp = model.VisualEncode(inputs, 3)
                temp = temp.permute(1, 0)
                temp = temp[ROI_index]
                temp = temp.permute(1, 0)
                out_match.append(temp)

                inputs = [None, sample[3], None, None, None, sample[5]]
                temp = model.VisualEncode(inputs, 3)
                temp = temp.permute(1, 0)
                temp = temp[ROI_index]
                temp = temp.permute(1, 0)
                out_unmatch.append(temp)

                sample[target_index] = sample[target_index].permute(1, 0)
                sample[target_index] = sample[target_index][ROI_index]
                sample[target_index] = sample[target_index].permute(1, 0)
                target.append(sample[target_index])
            elif args.encode_type == 'caption':
                inputs = [sample[2], None, None, None, sample[5], None]
                temp = model.BertEncode(inputs, 2)
                temp = temp.permute(1, 0)
                temp = temp[ROI_index]
                temp = temp.permute(1, 0)
                out_match.append(temp)

                inputs = [sample[3], None, None, None, sample[5], None]
                temp = model.BertEncode(inputs, 2)
                temp = temp.permute(1, 0)
                temp = temp[ROI_index]
                temp = temp.permute(1, 0)
                out_unmatch.append(temp)

                sample[target_index] = sample[target_index].permute(1, 0)
                sample[target_index] = sample[target_index][ROI_index]
                sample[target_index] = sample[target_index].permute(1, 0)
                target.append(sample[target_index])
    out_match = torch.cat(out_match, dim=0)
    out_unmatch = torch.cat(out_unmatch, dim=0)
    target = torch.cat(target, dim=0)
    pearson_match = pearsonr(out_match, target)
    pearson_unmatch = pearsonr(out_unmatch, target)

    out_match = out_match.detach().cpu().numpy()
    out_unmatch = out_unmatch.detach().cpu().numpy()
    np.savez(os.path.join(args.output_dir, 'match.npz'), out_match, out_unmatch)

    return pearson_match, pearson_unmatch


def test_performance(model, dataloader, device, args, target_index=None):
    if target_index is None:
        target_index = args.index

    if args.roi_type == 0:
        # 全脑encoding
        ROI_index = np.arange(args.num_vertices).astype(int)
    elif args.roi_type == 1:
        # V1-V3参与encoding
        ROI_index = np.loadtxt(os.path.join(args.processed_root, args.subject, 'ROI', 'EVC.txt')).astype(int)
    elif args.roi_type == 2:
        ROI_index = np.loadtxt(os.path.join(args.processed_root, args.subject, 'ROI', 'IFS.txt')).astype(int)
    ROI_index = torch.squeeze(torch.from_numpy(ROI_index).to(device, dtype=torch.long))

    pearsonr = PearsonCorrCoef(num_outputs=len(ROI_index)).to(device)
    model.eval()

    results = [[], [], []]
    output_pearson = []
    with torch.no_grad():
        for z in range(100):
            out = [[], [], []]
            target = []
            for step, sample in enumerate(dataloader):
                for i in range(len(sample)):
                    if type(sample[i]) == torch.Tensor:
                        sample[i] = sample[i].to(device, dtype=torch.float)

                if args.condition == 0:
                    sample[4] = torch.zeros_like(sample[4]).to(device, dtype=torch.float)
                    sample[5] = torch.zeros_like(sample[5]).to(device, dtype=torch.float)

                original_sample = copy.copy(sample)
                for shuffle in range(0, 3):

                    sample = copy.copy(original_sample)
                    if shuffle == 2:
                        sample[0] = list(sample[0])
                        random.shuffle(sample[0])
                        sample[0] = tuple(sample[0])

                        random_indices = torch.randperm(args.val_batch_size)
                        sample[1] = sample[1][random_indices]
                    elif shuffle == 1:
                        sample[0] = list(sample[0])
                        for i in range(len(sample[0])):
                            sample[0][i] = list(sample[0][i])
                            random.shuffle(sample[0][i])
                            sample[0][i] = ''.join(sample[0][i])
                        sample[0] = tuple(sample[0])

                        random_indices = torch.randperm(args.val_batch_size)
                        sample[1] = sample[1][random_indices]

                    if args.encode_type == 'visual':
                        temp = model.VisualEncode(sample, args.index)
                        # out.append(temp)
                        # target.append(sample[3])
                        temp = temp.permute(1, 0)
                        temp = temp[ROI_index]
                        temp = temp.permute(1, 0)

                        out[shuffle].append(temp)
                    elif args.encode_type == 'caption':
                        temp = model.BertEncode(sample, args.index)
                        # out.append(temp)
                        # target.append(sample[3])
                        temp = temp.permute(1, 0)
                        temp = temp[ROI_index]
                        temp = temp.permute(1, 0)

                        out[shuffle].append(temp)
                    else:
                        temp = model.predictive_coding(sample)
                        temp = temp.permute(1, 0)
                        temp = temp[ROI_index]
                        temp = temp.permute(1, 0)

                        out[shuffle].append(temp)
                    if shuffle == 0:
                        sample[target_index] = sample[target_index].permute(1, 0)
                        sample[target_index] = sample[target_index][ROI_index]
                        sample[target_index] = sample[target_index].permute(1, 0)
                        target.append(sample[target_index])

            target = torch.cat(target, dim=0)
            for shuffle in range(0, 3):
                out[shuffle] = torch.cat(out[shuffle], dim=0)
                pearson = pearsonr(out[shuffle], target)
                R2 = torch.mean(torch.square(pearson)).detach().cpu().numpy()
                results[shuffle].append(R2)
                if shuffle == 0:
                    output_pearson.append(pearson.detach().cpu().numpy())

    for shuffle in range(0, 3):
        results[shuffle] = np.stack(results[shuffle])
    output_pearson = np.vstack(output_pearson)
    output_pearson = np.mean(output_pearson, axis=0)

    return results, output_pearson


def plot_weight(stat, subject, hemi_vertices, savedir):
    fig = plt.figure(figsize=(24, 8))
    axes = []
    for i in range(8):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        axes.append(ax)

    # vmax = np.max(abs(stat))
    # stat = stat / vmax
    vmax = 0.4

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


if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)

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

    args.coords_root = os.path.join(args.processed_root, args.subject, 'coords.npy')
    hemi_vertices = [args.lh_vertices, args.rh_vertices]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'img_log')):
        os.makedirs(os.path.join(args.output_dir, 'img_log'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handle = logging.FileHandler(os.path.join(args.output_dir, 'record.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)

    if args.trial_type == 'match':
        Stimulus_index, Stimulus_pairs, stim_dict, train_list, val_list = prepare_pair_data(args.subject, args.Image_Caption_pairs_root, args.Stimulus_index_root, args.processed_root, data_type=args.data_type)

        train_dataset = Caption_Image_dataset(train_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False)
        val_dataset = Caption_Image_dataset(val_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False)

        train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_dataloader = data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    elif args.trial_type == 'unmatch':
        Stimulus_index, Stimulus_pairs, stim_dict, train_list, val_list = prepare_pair_data(args.subject, args.Image_Caption_pairs_root, args.Stimulus_index_root, args.processed_root, data_type=args.data_type, trial_type='unmatch')

        train_dataset = Predictive_coding_dataset(train_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False)
        val_dataset = Predictive_coding_dataset(val_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False)

        train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_dataloader = data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    coords = np.load(args.coords_root)
    coords = torch.from_numpy(coords).to(device, dtype=torch.float)

    if args.roi_type == 0:
        # 全脑encoding
        ROI_index = np.arange(args.num_vertices).astype(int)
    elif args.roi_type == 1:
        # V1-V3参与encoding
        ROI_index = np.loadtxt(os.path.join(args.processed_root, args.subject, 'ROI', 'EVC.txt')).astype(int)
    elif args.roi_type == 2:
        ROI_index = np.loadtxt(os.path.join(args.processed_root, args.subject, 'ROI', 'IFS.txt')).astype(int)
    ROI_index = torch.squeeze(torch.from_numpy(ROI_index).to(device, dtype=torch.long))

    if args.trial_type == 'match':
        model = EncodingModel(
            num_voxels=args.num_vertices,
            coords=coords,
            behavior_in=args.behavior_in,
            behavior_hidden=args.behavior_hidden,
            final_visual_emb_dim=args.final_visual_emb_dim,
            final_bert_emb_dim=args.final_bert_emb_dim,
            CLIP_model_root=args.CLIP_model_root,
            encode_type=args.encode_type,
            vit_model_root=args.vit_model_root,
            bert_model_root=args.bert_model_root
        )
    elif args.trial_type == 'unmatch':
        model = PredictiveEncodingModel(
            # num_voxels=args.num_vertices,
            coords=coords,
            behavior_in=args.behavior_in,
            behavior_hidden=args.behavior_hidden,
            final_visual_emb_dim=args.final_visual_emb_dim,
            final_bert_emb_dim=args.final_bert_emb_dim,
            CLIP_model_root=args.CLIP_model_root,
            encode_type=args.encode_type,
            vit_model_root=args.vit_model_root,
            bert_model_root=args.bert_model_root
        )
        # model = EncodingModel(
        #     num_voxels=args.num_vertices,
        #     coords=coords,
        #     behavior_in=args.behavior_in,
        #     behavior_hidden=args.behavior_hidden,
        #     final_visual_emb_dim=args.final_visual_emb_dim,
        #     final_bert_emb_dim=args.final_bert_emb_dim,
        #     CLIP_model_root=args.CLIP_model_root,
        #     encode_type=args.encode_type,
        #     vit_model_root=args.vit_model_root,
        #     bert_model_root=args.bert_model_root
        # )
    if args.encode_type == 'all':
        vit_ckpt = torch.load(args.vit_ckpt_dir, map_location=device)
        model.load_state_dict(vit_ckpt, strict=False)
        bert_ckpt = torch.load(args.bert_ckpt_dir, map_location=device)
        model.load_state_dict(bert_ckpt, strict=False)

        model.retinomapper.requires_grad_(False)
        model.visuallayerselector.requires_grad_(False)
        model.wordmapper.requires_grad_(False)
        model.bertlayerselector.requires_grad_(False)

    model = model.to(device, dtype=torch.float)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_schedule = cosine_sheduler(args.lr, args.min_lr, args.epochs, args.warmup_epochs, args.start_warmup_value)
    assert len(lr_schedule) == args.epochs, 'Length of lr_schedule must be equal to epochs'

    mae = MeanAbsoluteError().to(device)
    # pearsonr = PearsonCorrCoef(num_outputs=args.num_vertices).to(device)
    pearsonr = PearsonCorrCoef(num_outputs=len(ROI_index)).to(device)

    max_val = -torch.inf

    patience = 20
    count = 0

    logger.info('[Subject: %s] [behavior_hidden: %d] [final_visual_emb_dim: %d] [lr: %f] [epochs: %d] [weight_decay: %.5f] [pearson_weight: %d] [reg_weight: %.5f] [warmup_epochs: %d]'
                % (args.subject, args.behavior_hidden, args.final_visual_emb_dim, args.lr, args.epochs, args.weight_decay, args.pearson_weight, args.reg_weight, args.warmup_epochs))

    if not os.path.exists(os.path.join(args.output_dir, 'opt_img_model.pth')) and not os.path.exists(os.path.join(args.output_dir, 'opt_cap_model.pth')):
        logger.info('---------------Start Training---------------')
        for epoch in range(args.epochs):
            # 调整 learning rate
            for id, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_schedule[epoch]

            torch.cuda.empty_cache()
            for index, sample in enumerate(train_dataloader):
                for i in range(len(sample)):
                    if type(sample[i]) == torch.Tensor:
                        sample[i] = sample[i].to(device, dtype=torch.float)

                if args.condition == 0:
                    sample[4] = torch.zeros_like(sample[4]).to(device, dtype=torch.float)
                    sample[5] = torch.zeros_like(sample[5]).to(device, dtype=torch.float)
                # else:
                #     a = np.random.rand()
                #     if a > 0.95:
                #         sample[4][:, 0] = torch.randint(1, 200, [args.batch_size]).to(device, dtype=torch.float)
                #         sample[4][:, 1] = torch.randint(1, 52, [args.batch_size]).to(device, dtype=torch.float)
                #
                #         sample[5][:, 0] = torch.randint(1, 200, [args.batch_size]).to(device, dtype=torch.float)
                #         sample[5][:, 1] = torch.randint(1, 52, [args.batch_size]).to(device, dtype=torch.float)

                # image encode
                optimizer.zero_grad()
                if args.encode_type == 'visual':
                    out, reg = model.VisualEncode(sample, args.index)
                    target = sample[args.index]
                elif args.encode_type == 'caption':
                    out, reg = model.BertEncode(sample, args.index)
                    target = sample[args.index]
                else:
                    out, reg = model.predictive_coding(sample)
                    target = sample[args.index]

                target = target.permute(1, 0)
                target = target[ROI_index]
                target = target.permute(1, 0)

                out = out.permute(1, 0)
                out = out[ROI_index]
                out = out.permute(1, 0)

                mae = F.smooth_l1_loss(out, target)
                pearson = 1 - (torch.mean(pearsonr(out, target)) + 1) / 2
                # if epoch < 10:
                #     loss = mae + reg * args.reg_weight + pearson * args.pearson_weight
                # else:
                #     loss = mae - reg * args.reg_weight + pearson * args.pearson_weight
                #     # loss = mae + pearson * args.pearson_weight
                loss = mae + reg * args.reg_weight + pearson * args.pearson_weight
                loss.backward()
                optimizer.step()

                if index % 100 == 0:
                    logger.info('[Epoch %d/%d] [Step %d/%d] [MAE: %.5f] [Pearson: %.5f] [Reg: %.5f] [Loss: %.5f] [Learning_rate: %f]'
                                % (epoch, args.epochs, index, len(train_dataloader), mae.item(), pearson.item(), reg.item(), loss.item(), optimizer.param_groups[0]['lr']))

            pearson = calculateR(model, val_dataloader, device, args)
            R2 = torch.mean(torch.square(pearson))
            logger.info('In validation, [Epoch %d/%d] [R2: %.5f] [Learning_rate: %f]'
                        % (epoch, args.epochs, R2.item(), optimizer.param_groups[0]['lr']))

            pearson = pearson.detach().cpu().numpy()
            if args.encode_type == 'visual':
                savedir = os.path.join(args.output_dir, 'img_log', 'Epoch_' + str(epoch).zfill(3) + '_R_img.png')
            else:
                savedir = os.path.join(args.output_dir, 'img_log', 'Epoch_' + str(epoch).zfill(3) + '_R_cap.png')
            if len(pearson) != args.num_vertices:
                temp = np.zeros(args.num_vertices)
                for i in range(len(ROI_index)):
                    temp[ROI_index[i]] = pearson[i]
                pearson = temp
            plot_weight(pearson, args.subject, hemi_vertices, savedir)

            if R2 > max_val:
                if args.encode_type == 'visual':
                    savedir = os.path.join(args.output_dir, 'opt_img_model.pth')
                else:
                    savedir = os.path.join(args.output_dir, 'opt_cap_model.pth')
                max_val = R2
                torch.save(model.state_dict(), savedir)
                count = 0
            else:
                count += 1

            if count >= patience and epoch > patience + 5:
                logger.info('----------------Early Stopping!----------------')
                break

            gc.collect()
            torch.cuda.empty_cache()

        logger.info('----------------Finish training----------------')

    logger.info('----------------Test model performance----------------')
    if args.encode_type == 'visual':
        ckpt_dir = os.path.join(args.output_dir, 'opt_img_model.pth')
    else:
        ckpt_dir = os.path.join(args.output_dir, 'opt_cap_model.pth')
    ckpt = torch.load(ckpt_dir, map_location=device)
    model.load_state_dict(ckpt, strict=False)

    if args.trial_type == 'match':
        cap_results, cap_pearson = test_performance(model, val_dataloader, device, args, target_index=2)
        np.savetxt(os.path.join(args.output_dir, 'cap_results.csv'), cap_results, delimiter=',')
        cap_R2 = np.mean(cap_results[0])
        cap_R2_std = np.std(cap_results[0])
        cap_R2_shuffle = np.mean(cap_results[1])
        cap_R2_shuffle_std = np.std(cap_results[1])
        cap_R2_shuffle2 = np.mean(cap_results[2])
        cap_R2_shuffle2_std = np.std(cap_results[2])

        img_results, img_pearson = test_performance(model, val_dataloader, device, args, target_index=3)
        np.savetxt(os.path.join(args.output_dir, 'img_results.csv'), img_results, delimiter=',')
        img_R2 = np.mean(img_results[0])
        img_R2_std = np.std(img_results[0])
        img_R2_shuffle = np.mean(img_results[1])
        img_R2_shuffle_std = np.std(img_results[1])
        img_R2_shuffle2 = np.mean(img_results[2])
        img_R2_shuffle2_std = np.std(img_results[2])

        # test_data = {}
        # # Encode Caption beta
        # data_list = []
        # for i in range(20):
        #     cap_pearson = calculateR(model, val_dataloader, device, args, target_index=2)
        #     cap_pearson = cap_pearson.detach().cpu().numpy()
        #     temp = np.mean(np.square(cap_pearson))
        #     data_list.append(temp)
        # data_list = np.stack(data_list)
        # cap_R2 = np.mean(data_list)
        # cap_R2_std = np.std(data_list)
        # test_data['Caption'] = data_list
        #
        # if len(cap_pearson) != args.num_vertices:
        #     temp = np.zeros(args.num_vertices)
        #     for i in range(len(ROI_index)):
        #         temp[ROI_index[i]] = cap_pearson[i]
        #     cap_pearson = temp
        # savedir = os.path.join(args.output_dir, 'R_cap.png')
        # plot_weight(cap_pearson, args.subject, hemi_vertices, savedir)
        #
        # # Encode Shuffled Caption beta
        # data_list = []
        # for i in range(20):
        #     cap_pearson_shuffle = calculateR(model, val_dataloader, device, args, target_index=2, shuffle=1)
        #     cap_pearson_shuffle = cap_pearson_shuffle.detach().cpu().numpy()
        #     temp = np.mean(np.square(cap_pearson_shuffle))
        #     data_list.append(temp)
        # data_list = np.stack(data_list)
        # cap_R2_shuffle = np.mean(data_list)
        # cap_R2_shuffle_std = np.std(data_list)
        # test_data['Caption_shuffle_1'] = data_list
        #
        # # Encode Shuffled 2 Caption beta
        # data_list = []
        # for i in range(20):
        #     cap_pearson_shuffle2 = calculateR(model, val_dataloader, device, args, target_index=2, shuffle=2)
        #     cap_pearson_shuffle2 = cap_pearson_shuffle2.detach().cpu().numpy()
        #     temp = np.mean(np.square(cap_pearson_shuffle2))
        #     data_list.append(temp)
        # data_list = np.stack(data_list)
        # cap_R2_shuffle2 = np.mean(data_list)
        # cap_R2_shuffle2_std = np.std(data_list)
        # test_data['Caption_shuffle_2'] = data_list
        #
        # # Encode Image beta
        # data_list = []
        # for i in range(20):
        #     img_pearson = calculateR(model, val_dataloader, device, args, target_index=3)
        #     img_pearson = img_pearson.detach().cpu().numpy()
        #     temp = np.mean(np.square(img_pearson))
        #     data_list.append(temp)
        # data_list = np.stack(data_list)
        # img_R2 = np.mean(data_list)
        # img_R2_std = np.std(data_list)
        # test_data['Image'] = data_list
        #
        # if len(img_pearson) != args.num_vertices:
        #     temp = np.zeros(args.num_vertices)
        #     for i in range(len(ROI_index)):
        #         temp[ROI_index[i]] = img_pearson[i]
        #     img_pearson = temp
        # savedir = os.path.join(args.output_dir, 'R_img.png')
        # plot_weight(img_pearson, args.subject, hemi_vertices, savedir)
        #
        # # Encode Shuffled Image beta
        # data_list = []
        # for i in range(20):
        #     img_pearson_shuffle = calculateR(model, val_dataloader, device, args, target_index=3, shuffle=1)
        #     img_pearson_shuffle = img_pearson_shuffle.detach().cpu().numpy()
        #     temp = np.mean(np.square(img_pearson_shuffle))
        #     data_list.append(temp)
        # data_list = np.stack(data_list)
        # img_R2_shuffle = np.mean(data_list)
        # img_R2_shuffle_std = np.std(data_list)
        # test_data['Image_shuffle_1'] = data_list
        #
        # # Encode Shuffled 2 Image beta
        # data_list = []
        # for i in range(20):
        #     img_pearson_shuffle2 = calculateR(model, val_dataloader, device, args, target_index=3, shuffle=2)
        #     img_pearson_shuffle2 = img_pearson_shuffle2.detach().cpu().numpy()
        #     temp = np.mean(np.square(img_pearson_shuffle2))
        #     data_list.append(temp)
        # data_list = np.stack(data_list)
        # img_R2_shuffle2 = np.mean(data_list)
        # img_R2_shuffle2_std = np.std(data_list)
        # test_data['Image_shuffle_2'] = data_list
        #
        # savedir = os.path.join(args.output_dir, 'test_results.csv')
        # df = pd.DataFrame(test_data)
        # df.to_csv(savedir, index=False)

        # Prepare unmatch caption data
        if not os.path.exists(os.path.join(args.processed_root, args.subject, 'unmatched_caption.txt')):
            subject_beta_root = os.path.join(args.processed_root, args.subject, 'Stimulus', args.data_type)
            FileList = sorted(os.listdir(subject_beta_root))
            unmatch_list = []
            for index, file in tqdm(enumerate(FileList)):
                mat_root = os.path.join(subject_beta_root, file)
                temp = io.loadmat(mat_root)
                caption = temp['caption'][0][0]
                if caption == 0:
                    continue
                pair_unmatch = temp['pair_unmatch'][0][0]
                if pair_unmatch == 1:
                    unmatch_list.append(mat_root)
            with open(os.path.join(args.processed_root, args.subject, 'unmatched_caption.txt'), 'w', encoding='gbk') as f:
                f.write('\n'.join(unmatch_list))
        else:
            with open(os.path.join(args.processed_root, args.subject, 'unmatched_caption.txt'), 'r', encoding='gbk') as f:
                lines = f.readlines()
            unmatch_list = [line.strip() for line in lines]

        unmatch_list = unmatch_list[:len(val_list)]
        unmatch_dataset = Unmatch_dataset(unmatch_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False, encode_type=args.encode_type)
        unmatch_dataloader = data.DataLoader(unmatch_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

        # Encode unmatch caption beta
        pearson_match, pearson_unmatch = calculateR_unmatch(model, unmatch_dataloader, device, args)
        pearson_match = pearson_match.detach().cpu().numpy()
        match_R2 = np.mean(np.square(pearson_match))
        if len(pearson_match) != args.num_vertices:
            temp = np.zeros(args.num_vertices)
            for i in range(len(ROI_index)):
                temp[ROI_index[i]] = pearson_match[i]
            pearson_match = temp
        savedir = os.path.join(args.output_dir, 'R_match.png')
        plot_weight(pearson_match, args.subject, hemi_vertices, savedir)

        pearson_unmatch = pearson_unmatch.detach().cpu().numpy()
        unmatch_R2 = np.mean(np.square(pearson_unmatch))
        if len(pearson_unmatch) != args.num_vertices:
            temp = np.zeros(args.num_vertices)
            for i in range(len(ROI_index)):
                temp[ROI_index[i]] = pearson_unmatch[i]
            pearson_unmatch = temp
        savedir = os.path.join(args.output_dir, 'R_unmatch.png')
        plot_weight(pearson_unmatch, args.subject, hemi_vertices, savedir)

        # Encode shuffled unmatch caption beta
        pearson_match_shuffle, pearson_unmatch_shuffle = calculateR_unmatch(model, unmatch_dataloader, device, args, shuffle=1)
        pearson_match_shuffle = pearson_match_shuffle.detach().cpu().numpy()
        match_R2_shuffle = np.mean(np.square(pearson_match_shuffle))

        pearson_unmatch_shuffle = pearson_unmatch_shuffle.detach().cpu().numpy()
        unmatch_R2_shuffle = np.mean(np.square(pearson_unmatch_shuffle))

        # Encode shuffled 2 caption beta
        if args.encode_type == 'caption':
            pearson_match_shuffle2, pearson_unmatch_shuffle2 = calculateR_unmatch(model, unmatch_dataloader, device, args, shuffle=2)
            pearson_match_shuffle2 = pearson_match_shuffle2.detach().cpu().numpy()
            match_R2_shuffle2 = np.mean(np.square(pearson_match_shuffle2))

            pearson_unmatch_shuffle2 = pearson_unmatch_shuffle2.detach().cpu().numpy()
            unmatch_R2_shuffle2 = np.mean(np.square(pearson_unmatch_shuffle2))

        with open(os.path.join(args.output_dir, 'results.txt'), 'w', encoding='gbk') as f:
            f.write('Average R2 in caption response is %.5f, std is %.5f\n' % (cap_R2, cap_R2_std))
            f.write('Average R2 in shuffled caption response is %.5f, std is %.5f\n' % (cap_R2_shuffle, cap_R2_shuffle_std))
            f.write('Average R2 in incorrected caption response is %.5f, std is %.5f\n\n' %(cap_R2_shuffle2, cap_R2_shuffle2_std))

            f.write('Average R2 in image response is %.5f, std is %.5f\n' % (img_R2, img_R2_std))
            f.write('Average R2 in shuffled image response is %.5f, std is %.5f\n' % (img_R2_shuffle, img_R2_shuffle_std))
            f.write('Average R2 in incorrected image response is %.5f, std is %.5f\n\n' % (img_R2_shuffle2, img_R2_shuffle2_std))

            f.write('Average R2 in match caption response is %.5f\n' % match_R2)
            f.write('Average R2 in shuffled match caption response is %.5f\n' % match_R2_shuffle)
            if args.encode_type == 'caption':
                f.write('Average R2 in incorrect match caption response is %.5f\n' % match_R2_shuffle2)

            f.write('\nAverage R2 in unmatch caption response is %.5f\n' % unmatch_R2)
            f.write('Average R2 in shuffled unmatch caption response is %.5f\n' % unmatch_R2_shuffle)
            if args.encode_type == 'caption':
                f.write('Average R2 in incorrect unmatch caption response is %.5f\n' % unmatch_R2_shuffle2)
        f.close()

        # Save caption encode performance
        if len(cap_pearson) != args.num_vertices:
            temp = np.zeros(args.num_vertices)
            for i in range(len(ROI_index)):
                temp[ROI_index[i]] = cap_pearson[i]
            cap_pearson = temp
        savedir = os.path.join(args.output_dir, 'R_cap.png')
        plot_weight(cap_pearson, args.subject, hemi_vertices, savedir)

        savedir = os.path.join(args.output_dir, 'cap_encode_performance_lh.csv')
        results = {
            'R': cap_pearson[:args.lh_vertices],
            'R2': np.square(cap_pearson)[:args.lh_vertices]
        }
        df = pd.DataFrame(results)
        df.to_csv(savedir, index=False)

        savedir = os.path.join(args.output_dir, 'cap_encode_performance_rh.csv')
        results = {
            'R': cap_pearson[args.lh_vertices:],
            'R2': np.square(cap_pearson)[args.lh_vertices:]
        }
        df = pd.DataFrame(results)
        df.to_csv(savedir, index=False)

        # Save image encode performance
        if len(img_pearson) != args.num_vertices:
            temp = np.zeros(args.num_vertices)
            for i in range(len(ROI_index)):
                temp[ROI_index[i]] = img_pearson[i]
            img_pearson = temp
        savedir = os.path.join(args.output_dir, 'R_img.png')
        plot_weight(img_pearson, args.subject, hemi_vertices, savedir)

        savedir = os.path.join(args.output_dir, 'img_encode_performance_lh.csv')
        results = {
            'R': img_pearson[:args.lh_vertices],
            'R2': np.square(img_pearson)[:args.lh_vertices]
        }
        df = pd.DataFrame(results)
        df.to_csv(savedir, index=False)

        savedir = os.path.join(args.output_dir, 'img_encode_performance_rh.csv')
        results = {
            'R': img_pearson[args.lh_vertices:],
            'R2': np.square(img_pearson)[args.lh_vertices:]
        }
        df = pd.DataFrame(results)
        df.to_csv(savedir, index=False)
    elif args.trial_type == 'unmatch':
        data_list = []
        img_pearson = []
        for i in range(20):
            pearson = calculateR(model, val_dataloader, device, args, target_index=3)
            pearson = pearson.detach().cpu().numpy()
            temp = np.mean(np.square(pearson))
            data_list.append(temp)
            img_pearson.append(pearson)
        img_pearson = np.vstack(img_pearson)
        img_pearson = np.mean(img_pearson, axis=0)
        data_list = np.stack(data_list)
        img_R2 = np.mean(data_list)
        img_R2_std = np.std(data_list)

        if len(img_pearson) != args.num_vertices:
            temp = np.zeros(args.num_vertices)
            for i in range(len(ROI_index)):
                temp[ROI_index[i]] = img_pearson[i]
            img_pearson = temp
        savedir = os.path.join(args.output_dir, 'R_img.png')
        plot_weight(img_pearson, args.subject, hemi_vertices, savedir)

        savedir = os.path.join(args.output_dir, 'img_encode_performance_lh.csv')
        results = {
            'R': img_pearson[:args.lh_vertices],
            'R2': np.square(img_pearson)[:args.lh_vertices]
        }
        df = pd.DataFrame(results)
        df.to_csv(savedir, index=False)

        savedir = os.path.join(args.output_dir, 'img_encode_performance_rh.csv')
        results = {
            'R': img_pearson[args.lh_vertices:],
            'R2': np.square(img_pearson)[args.lh_vertices:]
        }
        df = pd.DataFrame(results)
        df.to_csv(savedir, index=False)

        with open(os.path.join(args.output_dir, 'results.txt'), 'w', encoding='gbk') as f:
            f.write('Average R2 in image response is %.5f, std is %.5f\n' % (img_R2, img_R2_std))
        f.close()

    # Test Layer Selector performance
    if args.roi_type == 0:
        # 全脑encoding
        ROI_index = np.arange(args.num_vertices).astype(int)
    elif args.roi_type == 1:
        # V1-V3参与encoding
        ROI_index = np.loadtxt(os.path.join(args.processed_root, args.subject, 'ROI', 'EVC.txt')).astype(int)
    elif args.roi_type == 2:
        ROI_index = np.loadtxt(os.path.join(args.processed_root, args.subject, 'ROI', 'IFS.txt')).astype(int)
    ROI_index = torch.squeeze(torch.from_numpy(ROI_index).to(device, dtype=torch.long))

    if args.encode_type == 'visual':
        # Calculate
        w_layer = model.visuallayerselector(model.coords, ROI_index)
        w_layer = w_layer.detach().cpu().numpy()
        savedir = os.path.join(args.output_dir, 'visual_layer_selector.npy')
        np.save(savedir, w_layer)
        # Base stats
        w_layer = np.mean(w_layer, axis=0)
        categoris = ['Layer 1', 'Layer 2', 'Layer 4', 'Layer 8', 'Layer 14', 'Layer 20', 'Layer 26', 'Layer 32']
        savedir = os.path.join(args.output_dir, 'visual_layer_selector.png')
        fig = plt.figure(figsize=(15, 5))
        plt.bar(categoris, w_layer, color='skyblue')
        fig.savefig(savedir)
        plt.close(fig)
    else:
        # Calculate
        w_layer = model.bertlayerselector(model.coords, ROI_index)
        w_layer = w_layer.detach().cpu().numpy()
        savedir = os.path.join(args.output_dir, 'bert_layer_selector.npy')
        np.save(savedir, w_layer)
        # Base stats
        w_layer = np.mean(w_layer, axis=0)
        categoris = ['Layer 1', 'Layer 2', 'Layer 4', 'Layer 8', 'Layer 12', 'Layer 16', 'Layer 20', 'Layer 24']
        savedir = os.path.join(args.output_dir, 'bert_layer_selector.png')
        fig = plt.figure(figsize=(15, 5))
        plt.bar(categoris, w_layer, color='skyblue')
        fig.savefig(savedir)
        plt.close(fig)

        # Test Mapper performance
        word_weight = model.wordmapper(model.coords, ROI_index)
        word_weight = word_weight.detach().cpu().numpy()
        savedir = os.path.join(args.output_dir, 'word_mapper.npy')
        np.save(savedir, word_weight)
        # Base stats
        word_weight = np.mean(word_weight, axis=0)
        categoris = range(51)
        savedir = os.path.join(args.output_dir, 'word_mapper.png')
        fig = plt.figure(figsize=(15, 5))
        plt.bar(categoris, word_weight, color='skyblue')
        fig.savefig(savedir)
        plt.close(fig)

    logger.info('----------------Complete testing----------------')