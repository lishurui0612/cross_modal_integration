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
from tensorboardX import SummaryWriter
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


def apply_global_weighting_and_noise(data, weight_range=(0.8, 1.2), noise_std=0.01, threshold=0.5):
    if torch.rand(1).item() > threshold:
        # 全局加权
        weight = random.uniform(*weight_range)
        data = data * weight
        # 添加高斯噪声
        noise = torch.randn_like(data) * noise_std
        return data + noise
    else:
        return data


def apply_random_mask(sample, mask_ratio=0.1, threshold=0.5):
    if torch.rand(1).item() > threshold:
        """应用随机mask到样本的部分特征"""
        mask = torch.rand_like(sample) < mask_ratio  # 生成随机mask，控制mask比例
        masked_sample = sample * mask  # 按照mask去除特征
        return masked_sample
    else:
        return sample


def mixco(caption_voxels, image_voxels, beta=0.15, s_thresh=0.5):
    perm = torch.randperm(caption_voxels.shape[0])
    betas = torch.distributions.Beta(beta, beta).sample([caption_voxels.shape[0]]).to(caption_voxels.device,dtype=caption_voxels.dtype)
    select = (torch.rand(caption_voxels.shape[0]) <= s_thresh).to(caption_voxels.device)
    betas_shape = [-1] + [1]*(len(caption_voxels.shape)-1)

    caption_voxels_shuffle = caption_voxels[perm].to(caption_voxels.device,dtype=caption_voxels.dtype)
    image_voxels_shuffle = image_voxels[perm].to(image_voxels.device,dtype=image_voxels.dtype)

    caption_voxels[select] = caption_voxels[select] * betas[select].reshape(*betas_shape) + \
                             caption_voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    image_voxels[select] = image_voxels[select] * betas[select].reshape(*betas_shape) + \
                           image_voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return caption_voxels, image_voxels, perm, betas, select


def cross_condition_resample(sample, exchange_ratio=0.1, threshold=0.3):
    """实现交叉条件重采样"""
    # 从 dataloader 中随机采样一个 batch，用于交叉条件重采样
    if torch.rand(1).item() > threshold:
        condition_indices = torch.randperm(len(sample))
        # 替换部分 beta 数据
        resampled_sample = sample[condition_indices]

        num_features = sample.shape[1]
        num_exchange = int(num_features * exchange_ratio)

        exchange_indices = torch.randperm(num_features)[:num_exchange]
        exchanged_sample = sample.clone()
        exchanged_sample[:, exchange_indices] = resampled_sample[:, exchange_indices]

        return exchanged_sample
    else:
        return sample


def mixco_nce(preds, targs, temp=0.006, perm=None, betas=None, select=None, distributed=False,
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp

    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss


def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp

    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()

    loss = (loss1 + loss2) / 2
    return loss


def add_argparse_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--Stimulus_index_root', type=str, default='/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt')
    parser.add_argument('--Image_Caption_pairs_root', type=str, default='/public_bme/data/lishr/COCO_CN/Selected_Match_Image_Caption_pairs.txt')
    # parser.add_argument('--processed_root', type=str, default='/public_bme/data/lishr/Cross_modal/Processed_Data/')
    parser.add_argument('--processed_root', type=str, default='/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data')
    parser.add_argument('--image_root', type=str, default='/public_bme/data/lishr/COCO_CN/All_images_480')
    parser.add_argument('--CLIP_model_root', type=str, default='/public/home/lishr2022/Project/Cross-modal/encoding/cross_encoding/model_cache')
    parser.add_argument('--nsd_root', type=str, default='/public_bme/data/lishr/NSD/nsddata_betas/ppdata')
    parser.add_argument('--output_dir', type=str, default='/public/home/lishr2022/Project/Cross-modal/reconstruction/encoding/test_001')
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
    parser.add_argument('--use_mix', type=int, default=0)
    parser.add_argument('--use_noise', type=int, default=0)
    parser.add_argument('--use_mask', type=int, default=0)
    parser.add_argument('--use_cross_condition', type=int, default=0)
    parser.add_argument('--mix_epochs', type=int, default=50)
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


def validation(model, CLIP_model, val_dataloader, device, args):
    model.eval()
    NCE = InfoNCE(temperature=0.07).to(device)
    val_beta_loss = torch.zeros([]).to(device, dtype=torch.float)
    val_image_loss = torch.zeros([]).to(device, dtype=torch.float)
    val_caption_loss = torch.zeros([]).to(device, dtype=torch.float)
    val_loss = torch.zeros([]).to(device, dtype=torch.float)
    with torch.no_grad():
        for step, sample in enumerate(val_dataloader):

            for i in range(1, len(sample)):
                sample[i] = sample[i].to(device, dtype=torch.float)

            # beta semantic embeddings
            caption_feature, caption_feature_clip = model(sample[2])
            image_feature, image_feature_clip = model(sample[3])

            # CLIP embeddings
            inputs = clip.tokenize(sample[0]).to(device)
            clip_caption_feature = CLIP_model.encode_text(inputs).to(torch.float)
            clip_image_feature = CLIP_model.encode_image(sample[1]).to(torch.float)

            caption_feature /= caption_feature.norm(dim=-1, keepdim=True)
            caption_feature_clip /= caption_feature_clip.norm(dim=-1, keepdim=True)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            image_feature_clip /= image_feature_clip.norm(dim=-1, keepdim=True)
            clip_caption_feature /= clip_caption_feature.norm(dim=-1, keepdim=True)
            clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)

            beta_loss = soft_clip_loss(caption_feature, image_feature)
            image_loss = soft_clip_loss(image_feature_clip, clip_image_feature)
            caption_loss = soft_clip_loss(caption_feature_clip, clip_caption_feature)

            # loss = beta_loss + image_loss + caption_loss
            loss = beta_loss

            val_beta_loss += beta_loss
            val_image_loss += image_loss
            val_caption_loss += caption_loss
            val_loss += loss

    val_beta_loss /= step
    val_image_loss /= step
    val_caption_loss /= step
    val_loss /= step
    model.train()
    return val_beta_loss, val_image_loss, val_caption_loss, val_loss


if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser = add_argparse_args(parser)
    args = parser.parse_args()

    if args.use_ne == 1:
        args.lr /= 2

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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handle = logging.FileHandler(os.path.join(args.output_dir, 'record' + str(args.num) + '.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)

    writer = SummaryWriter(args.output_dir)

    Stimulus_index, Stimulus_pairs, stim_dict, train_list, val_list = prepare_pair_data(args.subject, args.Image_Caption_pairs_root, args.Stimulus_index_root, args.processed_root, args.data_type)

    train_dataset = Caption_Image_dataset(train_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=True, ne=False)
    val_dataset = Caption_Image_dataset(val_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=True, ne=False)

    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    CLIP_model, _ = load_from_name('ViT-H-14', device=device, download_root=args.CLIP_model_root)

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

    model = model.to(device, dtype=torch.float)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_schedule = cosine_sheduler(args.lr, args.min_lr, args.epochs, args.warmup_epochs, args.start_warmup_value)
    assert len(lr_schedule) == args.epochs, 'Length of lr_schedule must be equal to epochs'

    min_val = torch.inf

    patience = 20
    count = 0

    logger.info('[Encoder dim: %d]' % len(model.index))
    logger.info(
        '[Subject: %s] [Image_size: %d] [Type: %s] [Encoder depth: %d] [lr: %f] [epochs: %d] [weight_decay: %f] [warmup_epochs: %d]'
        % (args.subject, args.image_size, args.type, args.depth, args.lr, args.epochs, args.weight_decay,
           args.warmup_epochs))

    if not os.path.exists(os.path.join(args.output_dir, 'opt_model.pth')):
        logger.info('---------------Start Training---------------')
        NCE = InfoNCE(temperature=0.07).to(device)
        for epoch in range(args.epochs):
            # 调整 learning rate
            for id, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_schedule[epoch]

            torch.cuda.empty_cache()

            for index, sample in enumerate(train_dataloader):
                for i in range(1, len(sample)):
                    sample[i] = sample[i].to(device, dtype=torch.float)

                if args.use_noise:
                    sample[2] = apply_global_weighting_and_noise(sample[2], weight_range=(0.8, 1.2), noise_std=0.01)
                    sample[3] = apply_global_weighting_and_noise(sample[3], weight_range=(0.8, 1.2), noise_std=0.01)

                # 随机mask
                if args.use_mask:
                    sample[2] = apply_random_mask(sample[2])
                    sample[3] = apply_random_mask(sample[3])
                    
                if args.use_mix == 1 and epoch < args.mix_epochs:
                    original_caption_voxels = sample[2]
                    original_image_voxels = sample[3]
                    sample[2], sample[3], perm, betas, select = mixco(sample[2], sample[3])
                elif args.use_cross_condition == 1:
                    sample[2] = cross_condition_resample(sample[2])
                    sample[3] = cross_condition_resample(sample[3])

                optimizer.zero_grad()

                caption_feature, caption_feature_clip = model(sample[2])
                image_feature, image_feature_clip = model(sample[3])

                inputs = clip.tokenize(sample[0]).to(device)
                clip_caption_feature = CLIP_model.encode_text(inputs).to(torch.float)
                clip_image_feature = CLIP_model.encode_image(sample[1]).to(torch.float)

                caption_feature_norm = caption_feature / caption_feature.norm(dim=-1, keepdim=True)
                caption_feature_clip_norm = caption_feature_clip / caption_feature_clip.norm(dim=-1, keepdim=True)
                image_feature_norm = image_feature / image_feature.norm(dim=-1, keepdim=True)
                image_feature_clip_norm = image_feature_clip / image_feature_clip.norm(dim=-1, keepdim=True)
                clip_caption_feature_norm = clip_caption_feature / clip_caption_feature.norm(dim=-1, keepdim=True)
                clip_image_feature_norm = clip_image_feature / clip_image_feature.norm(dim=-1, keepdim=True)

                if args.use_mix == 1 and epoch < args.mix_epochs:
                    original_caption_feature, _ = model(original_caption_voxels)
                    original_image_feature, _ = model(original_image_voxels)

                    original_caption_feature_norm = original_caption_feature / original_caption_feature.norm(dim=-1, keepdim=True)
                    original_image_feature_norm = original_image_feature / original_image_feature.norm(dim=-1, keepdim=True)

                if args.use_mix == 1 and epoch < args.mix_epochs:
                    beta_loss = 0.5 * (mixco_nce(
                        caption_feature_norm,
                        original_image_feature_norm,
                        temp=0.006,
                        perm=perm, betas=betas, select=select
                    ) + mixco_nce(
                        image_feature_norm,
                        original_caption_feature_norm,
                        temp=0.006,
                        perm=perm, betas=betas, select=select
                    ))
                    image_loss = mixco_nce(
                        image_feature_clip_norm,
                        clip_image_feature_norm,
                        temp=0.006,
                        perm=perm, betas=betas, select=select
                    )
                    caption_loss = mixco_nce(
                        caption_feature_clip_norm,
                        clip_caption_feature_norm,
                        temp=0.006,
                        perm=perm, betas=betas, select=select
                    )
                else:
                    beta_loss = soft_clip_loss(caption_feature_norm, image_feature_norm)
                    image_loss = soft_clip_loss(image_feature_clip_norm, clip_image_feature_norm)
                    caption_loss = soft_clip_loss(caption_feature_clip_norm, clip_caption_feature_norm)

                loss = beta_loss + image_loss + caption_loss
                # loss = beta_loss

                if index % 100 == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f"grad/{name}", param.grad, epoch * len(train_dataloader) + index)

                    writer.add_histogram('Activation/caption_feature', caption_feature, epoch * len(train_dataloader) + index)
                    writer.add_histogram('Activation/caption_feature_clip', caption_feature_clip_norm, epoch * len(train_dataloader) + index)
                    writer.add_histogram('Activation/image_feature', image_feature, epoch * len(train_dataloader) + index)
                    writer.add_histogram('Activation/image_feature_clip', image_feature_clip_norm, epoch * len(train_dataloader) + index)
                    writer.add_histogram('Activation/clip_image_feature', clip_image_feature_norm, epoch * len(train_dataloader) + index)
                    writer.add_histogram('Activation/clip_caption_feature', clip_caption_feature_norm, epoch * len(train_dataloader) + index)


                loss.backward()
                optimizer.step()

                if index % 100 == 0:
                    logger.info(
                        '[Epoch %d/%d] [Step %d/%d] [Beta_loss: %.5f] [Image_loss: %.5f] [Caption_loss: %.5f] [Loss: %.5f] [Learning_rate: %f]'
                        % (epoch, args.epochs, index, len(train_dataloader), beta_loss.item(), image_loss.item(),
                           caption_loss.item(), loss.item(), optimizer.param_groups[0]['lr']))

                    writer.add_scalar('Train_loss/Beta_loss', beta_loss.item(), epoch * len(train_dataloader) + index)
                    writer.add_scalar('Train_loss/Image_loss', image_loss.item(), epoch * len(train_dataloader) + index)
                    writer.add_scalar('Train_loss/Caption_loss', caption_loss.item(), epoch * len(train_dataloader) + index)
                    writer.add_scalar('Train_loss/Total_loss', loss.item(), epoch * len(train_dataloader) + index)

            val_beta_loss, val_image_loss, val_caption_loss, val_loss = validation(model, CLIP_model, val_dataloader, device, args)
            logger.info(
                'In validation, [Epoch %d/%d] [Val_beta_loss: %.5f] [Val_image_loss: %.5f] [val_caption_loss: %.5f] [Val_loss: %.5f] [Learning_rate: %f]'
                % (epoch, args.epochs, val_beta_loss.item(), val_image_loss.item(), val_caption_loss.item(),
                   val_loss.item(), optimizer.param_groups[0]['lr']))

            writer.add_scalar('Val_loss/Beta_loss', val_beta_loss.item(), epoch)
            writer.add_scalar('Val_loss/Image_loss', val_image_loss.item(), epoch)
            writer.add_scalar('Val_loss/Caption_loss', val_caption_loss.item(), epoch)
            writer.add_scalar('Val_loss/Total_loss', val_loss.item(), epoch)

            if val_loss < min_val:
                min_val = val_loss
                savedir = os.path.join(args.output_dir, 'opt_model.pth')
                torch.save(model.state_dict(), savedir)
                count = 0
            else:
                count += 1

            if count > patience:
                logger.info('----------------Early Stopping!----------------')
                break

        logger.info('----------------Finish Training Attentive Probing Head!----------------')
        writer.close()

    '''
    Test retrival performance
    1 image beta             <->   caption beta
    2 image beta             <->   image
    3 caption beta           <->   caption
    4 image beta             <->   caption neural encoding
    5 image neural encoding  <->   caption neural encoding
    '''

    if os.path.exists(os.path.join(args.output_dir, 'opt_model.pth')):
        f = open(os.path.join(args.output_dir, 'results.txt'), 'w', encoding='gbk')

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

        logger.info('- - calculating features')
        # load data
        transform = image_transform()
        Stimulus_index, Stimulus_pairs, stim_dict, train, val = prepare_pair_data(args.subject, args.Image_Caption_pairs_root, args.Stimulus_index_root, args.processed_root, data_type=args.data_type)
        datadir = os.path.join(args.processed_root, args.subject, 'Stimulus', args.data_type)
        FileList = sorted(os.listdir(datadir))

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

        for type in range(2):
            count = 0
            mean_image_beta = torch.zeros((1, args.num_vertices)).to(device, dtype=torch.float)

            brain_caption_feature = []
            brain_clip_caption_feature = []
            # brain_caption_ne_feature = []
            brain_caption_label = []
            brain_image_feature = []
            brain_clip_image_feature = []
            # brain_image_ne_feature = []
            brain_image_label = []
            clip_caption_feature = []
            clip_caption_label = []
            clip_image_feature = []
            clip_image_label = []

            clip_feature_used_for_label = []
            with torch.no_grad():
                for step, file in enumerate(tqdm(FileList)):

                    if file[-5] != 'a':
                        stim = int(file[-24:-19])
                    else:
                        stim = int(file[-22:-17])

                    if type == 0:
                        if stim not in val_list and Stimulus_index[Stimulus_pairs[Stimulus_index[stim]]] not in val_list:
                            continue

                    if Stimulus_index[stim][-3:] == 'jpg':
                        image_dir = os.path.join(args.image_root, Stimulus_index[stim])
                        image = transform(Image.open(image_dir))
                        image = torch.unsqueeze(image.to(device, dtype=torch.float), dim=0)

                        mat_dir = os.path.join(datadir, file)
                        image_mat = io.loadmat(mat_dir)
                        image_beta = torch.unsqueeze(
                            torch.from_numpy(np.squeeze(image_mat['beta'])).to(device, dtype=torch.float), dim=0)
                        image_condition = torch.unsqueeze(torch.from_numpy(np.array([
                            image_mat['run'][0][0],
                            image_mat['timestamp'][0][0], image_mat['behavior'][0][0], image_mat['behavior_rt'][0][0],
                            image_mat['caption'][0][0], image_mat['image'][0][0], image_mat['unmatch'][0][0],
                            image_mat['repeat'][0][0]
                        ])).to(device, dtype=torch.float), dim=0)
                        # image_ne = torch.unsqueeze(torch.from_numpy(np.squeeze(image_mat['neural_encoding'])).to(device, dtype=torch.float), dim=0)

                        count += 1
                        mean_image_beta += image_beta

                        image_feature, image_feature_clip = model(image_beta)

                        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
                        image_feature = image_feature.detach().cpu().numpy()
                        brain_image_feature.append(image_feature)

                        image_feature_clip = image_feature_clip / image_feature_clip.norm(dim=-1, keepdim=True)
                        image_feature_clip = image_feature_clip.detach().cpu().numpy()
                        brain_clip_image_feature.append(image_feature_clip)

                        # image_ne_feature, image_ne_feature_clip = model(image_ne)
                        # image_ne_feature = image_ne_feature / image_ne_feature.norm(dim=-1, keepdim=True)
                        # image_ne_feature = image_ne_feature.detach().cpu().numpy()
                        # brain_image_ne_feature.append(image_ne_feature)

                        brain_image_label.append(stim)

                        if stim not in clip_image_label:
                            feature = CLIP_model.encode_image(image).to(torch.float)
                            feature = feature / feature.norm(dim=-1, keepdim=True)
                            feature = feature.detach().cpu().numpy()

                            clip_image_feature.append(feature)
                            clip_image_label.append(stim)

                    else:
                        caption = [Stimulus_index[stim]]

                        mat_dir = os.path.join(datadir, file)
                        caption_mat = io.loadmat(mat_dir)
                        caption_beta = torch.unsqueeze(
                            torch.from_numpy(np.squeeze(caption_mat['beta'])).to(device, dtype=torch.float), dim=0)
                        caption_condition = torch.unsqueeze(torch.from_numpy(np.array([
                            caption_mat['run'][0][0],
                            caption_mat['timestamp'][0][0], caption_mat['behavior'][0][0], caption_mat['behavior_rt'][0][0],
                            caption_mat['caption'][0][0], caption_mat['image'][0][0], caption_mat['unmatch'][0][0],
                            caption_mat['repeat'][0][0]
                        ])).to(device, dtype=torch.float), dim=0)
                        # caption_ne = torch.unsqueeze(torch.from_numpy(np.squeeze(caption_mat['neural_encoding'])).to(device, dtype=torch.float), dim=0)

                        caption_feature, caption_feature_clip = model(caption_beta)

                        caption_feature = caption_feature / caption_feature.norm(dim=-1, keepdim=True)
                        caption_feature = caption_feature.detach().cpu().numpy()
                        brain_caption_feature.append(caption_feature)

                        caption_feature_clip = caption_feature_clip / caption_feature_clip.norm(dim=-1, keepdim=True)
                        caption_feature_clip = caption_feature_clip.detach().cpu().numpy()
                        brain_clip_caption_feature.append(caption_feature_clip)

                        # caption_ne_feature, caption_ne_feature_clip = model(caption_ne)
                        # caption_ne_feature = caption_ne_feature / caption_ne_feature.norm(dim=-1, keepdim=True)
                        # caption_ne_feature = caption_ne_feature.detach().cpu().numpy()
                        # brain_caption_ne_feature.append(caption_ne_feature)

                        brain_caption_label.append(Stimulus_index[Stimulus_pairs[Stimulus_index[stim]]])

                        if Stimulus_index[Stimulus_pairs[Stimulus_index[stim]]] not in clip_caption_label:
                            inputs = clip.tokenize(caption).to(device)
                            feature = CLIP_model.encode_text(inputs).to(torch.float)
                            feature = feature / feature.norm(dim=-1, keepdim=True)
                            feature = feature.detach().cpu().numpy()

                            clip_caption_feature.append(feature)
                            clip_caption_label.append(Stimulus_index[Stimulus_pairs[Stimulus_index[stim]]])

            brain_caption_feature = np.concatenate(brain_caption_feature, axis=0)
            brain_clip_caption_feature = np.concatenate(brain_clip_caption_feature, axis=0)
            # brain_caption_ne_feature = np.concatenate(brain_caption_ne_feature, axis=0)
            brain_image_feature = np.concatenate(brain_image_feature, axis=0)
            brain_clip_image_feature = np.concatenate(brain_clip_image_feature, axis=0)
            # brain_image_ne_feature = np.concatenate(brain_image_ne_feature, axis=0)
            clip_caption_feature = np.concatenate(clip_caption_feature, axis=0)
            clip_image_feature = np.concatenate(clip_image_feature, axis=0)

            mean_image_beta /= count
            mean_brain_image_feature, _ = model(mean_image_beta)
            mean_brain_image_feature = mean_brain_image_feature / mean_brain_image_feature.norm(dim=-1, keepdim=True)
            mean_brain_image_feature = mean_brain_image_feature.detach().cpu().numpy()
            savedir = os.path.join(args.output_dir, 'uncondition_embedding.npy')
            np.save(savedir, mean_brain_image_feature)

            label = np.load('/public/home/lishr2022/Project/Cross-modal/beta_estimate/clip_cluster.npy')

            if not os.path.exists(os.path.join(args.output_dir, 'retrieval_performance.txt')):
                similarity = brain_image_feature @ brain_caption_feature.T
                index = np.argsort(-similarity, axis=1)
                z = open(os.path.join(args.output_dir, 'retrieval_performance.txt'), 'w', encoding='gbk')
                for i in range(len(brain_image_feature)):
                    if brain_image_label[i] not in val_list:
                        continue
                    z.write('%s ' % Stimulus_index[brain_image_label[i]])
                    z.write('%s ' % Stimulus_pairs[Stimulus_index[brain_image_label[i]]])
                    for j in range(10):
                        z.write('%s ' % Stimulus_pairs[Stimulus_index[brain_caption_label[index[i, j]]]])
                    z.write('\n')
                z.close()

            # image beta <-> caption beta
            similarity = brain_image_feature @ brain_caption_feature.T
            index = np.argsort(-similarity, axis=1)
            top_1 = 0
            top_5 = 0
            top_10 = 0
            count = 0
            label_performance = np.zeros((10, 4))
            for i in range(len(brain_image_feature)):
                if brain_image_label[i] not in val_list:
                    continue
                count += 1
                label_performance[label[brain_image_label[i]], 0] += 1
                for j in range(10):
                    if brain_image_label[i] == brain_caption_label[index[i, j]]:
                        if j < 1:
                            top_1 += 1
                            top_5 += 1
                            top_10 += 1
                            label_performance[label[brain_image_label[i]], 1:] += 1
                        elif j < 5:
                            top_5 += 1
                            top_10 += 1
                            label_performance[label[brain_image_label[i]], 2:] += 1
                        else:
                            top_10 += 1
                            label_performance[label[brain_image_label[i]], 3:] += 1
                        break
            top_1 /= count
            top_5 /= count
            top_10 /= count
            label_performance = label_performance[:, 1:] / label_performance[:, 0:1]
            plt.figure()
            plt.bar(range(10), label_performance[:, 0])
            plt.ylim(0, 1.0)
            plt.title('CLIP cluster retrieval performance')
            plt.savefig(os.path.join(args.output_dir, 'cluster_retrieval_'+str(type)+'.png'))
            plt.close()
            f.write('For image beta <-> caption beta, top_1 acc = %.5f, top_5 acc = %.5f, top_10 acc = %.5f\n' % (top_1, top_5, top_10))

            # image beta     <->   image
            similarity = brain_clip_image_feature @ clip_image_feature.T
            index = np.argsort(-similarity, axis=1)
            top_1 = 0
            top_5 = 0
            top_10 = 0
            count = 0
            for i in range(len(brain_clip_image_feature)):
                if brain_image_label[i] not in val_list:
                    continue
                count += 1
                for j in range(10):
                    if brain_image_label[i] == clip_image_label[index[i, j]]:
                        if j < 1:
                            top_1 += 1
                            top_5 += 1
                            top_10 += 1
                        elif j < 5:
                            top_5 += 1
                            top_10 += 1
                        else:
                            top_10 += 1
                        break
            top_1 /= count
            top_5 /= count
            top_10 /= count
            f.write('For image beta <-> image, top_1 acc = %.5f, top_5 acc = %.5f, top_10 acc = %.5f\n' % (top_1, top_5, top_10))

            # caption beta    <->   caption
            similarity = brain_clip_caption_feature @ clip_caption_feature.T
            index = np.argsort(-similarity, axis=1)
            top_1 = 0
            top_5 = 0
            top_10 = 0
            count = 0
            for i in range(len(brain_clip_caption_feature)):
                if brain_caption_label[i] not in val_list:
                    continue
                count += 1
                for j in range(10):
                    if brain_caption_label[i] == clip_caption_label[index[i, j]]:
                        if j < 1:
                            top_1 += 1
                            top_5 += 1
                            top_10 += 1
                        elif j < 5:
                            top_5 += 1
                            top_10 += 1
                        else:
                            top_10 += 1
                        break
            top_1 /= count
            top_5 /= count
            top_10 /= count
            f.write('For caption beta <-> caption, top_1 acc = %.5f, top_5 acc = %.5f, top_10 acc = %.5f\n' % (top_1, top_5, top_10))

            # # image beta     <->   caption neural encoding
            # similarity = brain_image_feature @ brain_caption_ne_feature.T
            # index = np.argsort(-similarity, axis=1)
            # top_1 = 0
            # top_5 = 0
            # top_10 = 0
            # count = 0
            # for i in range(len(brain_image_feature)):
            #     if brain_image_label[i] not in val_list:
            #         continue
            #     count += 1
            #     for j in range(10):
            #         if brain_image_label[i] == brain_caption_label[index[i, j]]:
            #             if j < 1:
            #                 top_1 += 1
            #                 top_5 += 1
            #                 top_10 += 1
            #             elif j < 5:
            #                 top_5 += 1
            #                 top_10 += 1
            #             else:
            #                 top_10 += 1
            #             break
            # top_1 /= count
            # top_5 /= count
            # top_10 /= count
            # f.write('For image beta <-> caption neural encoding, top_1 acc = %.5f, top_5 acc = %.5f, top_10 acc = %.5f\n' % (top_1, top_5, top_10))
            #
            # # image neural encoding   <->   caption neural encoding
            # similarity = brain_image_ne_feature @ brain_caption_ne_feature.T
            # index = np.argsort(-similarity, axis=1)
            # top_1 = 0
            # top_5 = 0
            # top_10 = 0
            # count = 0
            # for i in range(len(brain_image_ne_feature)):
            #     if brain_image_label[i] not in val_list:
            #         continue
            #     count += 1
            #     for j in range(10):
            #         if brain_image_label[i] == brain_caption_label[index[i, j]]:
            #             if j < 1:
            #                 top_1 += 1
            #                 top_5 += 1
            #                 top_10 += 1
            #             elif j < 5:
            #                 top_5 += 1
            #                 top_10 += 1
            #             else:
            #                 top_10 += 1
            #             break
            # top_1 /= count
            # top_5 /= count
            # top_10 /= count
            # f.write('For image neural encoding <-> caption neural encoding, top_1 acc = %.5f, top_5 acc = %.5f, top_10 acc = %.5f\n' % (top_1, top_5, top_10))

        f.close()
