# coding=gbk
import os
import gc
import math
import torch
import random
import nilearn
import numpy as np
from torch import nn
from scipy import io
from PIL import Image
from tqdm import tqdm
from nilearn import surface
import cn_clip.clip as clip
from einops import rearrange
import torch.utils.data as data
import matplotlib.pyplot as plt
from timm.layers.mlp import Mlp
import torch.nn.functional as F
from collections import OrderedDict
from scipy.sparse import csr_matrix
from einops import rearrange, repeat
from scipy.interpolate import griddata
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from typing import Any, Dict, List, Optional, Tuple, Union
from cn_clip.clip import load_from_name, image_transform, load_from_name, _tokenizer
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, R2Score, MetricCollection
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode

from model import EncodingModel
from dataset import fMRI2grid
from imagenet_autoencoder.models.resnet import ResNetAutoEncoder, get_configs


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class SphereGridSpatialRelationEncoder(nn.Module):
    def __init__(self, frequency_num=16, max_radius=1, min_radius=1e-6):
        super(SphereGridSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius

        self.cal_freq_list()
        self.cal_freq_mat()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def cal_freq_list(self):
        log_timescale_increment = (math.log(float(self.max_radius) / float(self.min_radius)) / (self.frequency_num*1.0 - 1))
        timescales = self.min_radius * np.exp(np.arange(self.frequency_num).astype(float) * log_timescale_increment)
        self.freq_list = 1.0 / timescales

    def cal_freq_mat(self):
        self.freq_mat = np.expand_dims(self.freq_list, axis=1)

    def forward(self, coords):
        B, H, W, _ = coords.shape # B H W 2
        coords = np.expand_dims(coords, axis=-1) # B H W 2 1
        coords = np.expand_dims(coords, axis=-1) # B H W 2 1 1
        coords = np.repeat(coords, self.frequency_num, axis=-2) # B H W 2 FN 1

        lon_single = coords[:, :, :, :1, :, :]
        lat_single = coords[:, :, :, 1:, :, :]

        lon_single_sin = np.sin(lon_single)
        lon_single_cos = np.cos(lon_single)

        lat_single_sin = np.sin(lat_single)
        lat_single_cos = np.cos(lat_single)

        spr_embeds = coords * self.freq_mat # B H W 2 FN 1

        lon = spr_embeds[:, :, :, :1, :, :]
        lat = spr_embeds[:, :, :, 1:, :, :]

        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)

        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)

        spr_embeds_ = np.concatenate([lat_sin, lat_cos, lon_sin, lon_cos, lat_cos * lon_single_cos, lat_single_cos * lon_cos, lat_cos * lon_single_sin, lat_single_cos * lon_sin], axis = -1)

        spr_embeds = np.reshape(spr_embeds_, (B, H, W, -1))
        return torch.from_numpy(spr_embeds).to(self.device, dtype=torch.float)


class BrainSpherePositionEncoder(nn.Module):
    def __init__(self, embed_dim, max_radius=1, min_radius=1e-6):
        super(BrainSpherePositionEncoder, self).__init__()
        assert embed_dim % 8 == 0, 'Embedding dim must be divided by 8.'

        self.embed_dim = embed_dim

        sphere_freauency_num = self.embed_dim / 8
        self.sphere_PE = SphereGridSpatialRelationEncoder(frequency_num=sphere_freauency_num, max_radius=max_radius, min_radius=min_radius)

        self.scale = nn.Parameter(torch.ones([]))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def hemi_PE(self, hemi):
        B, H, W, _ = hemi.shape

        if np.mean(hemi[:, :H//2]) == 0:
            t = np.zeros((B, H//2, W, self.embed_dim // 2))
            pe = np.concatenate([np.sin(t), np.cos(t)], axis=-1)
        else:
            t = np.zeros((B, H//2, W, self.embed_dim // 2))
            pe = np.concatenate([np.cos(t), np.sin(t)], axis=-1)
        rest = 1 - pe
        outputs = np.concatenate([pe, rest], axis=-3)

        return torch.from_numpy(outputs).to(self.device, dtype=torch.float)

    def forward(self, coords):
        B, H, W, _ = coords.shape

        hemi = coords[:, :, :, :1]
        sphere_coords = coords[:, :, :, 1:]
        hemi_pe = self.hemi_PE(hemi)
        sphere_PE = self.sphere_PE(sphere_coords)

        pe = sphere_PE * self.scale + hemi_pe

        return pe


def positional_encoding(max_len, d_model):
    """
    生成位置编码矩阵

    参数：
    max_len：序列的最大长度
    d_model：词向量的维度

    返回值：
    position_encoding：位置编码矩阵，shape为(max_len, d_model)
    """
    position_encoding = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            position_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
            position_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
    return position_encoding


# B 1 H W -> B Num_Patches D (在H上叠加)
class PatchEmbed(nn.Module):
    def __init__(self, H, W, patch_size=16, in_chans=1, embed_dim=768):
        super(PatchEmbed, self).__init__()
        patch_size = pair(patch_size)
        num_patches = (H // patch_size[0]) * (W // patch_size[1])
        self.patch_shape = (H // patch_size[0], W // patch_size[1])
        self.img_h = H
        self.img_w = W
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_h and W == self.img_w, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_h}*{self.img_w})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))    # (B, N_head, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0.,  attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)    # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))      # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                  attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,  attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class AttentiveBlock(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()

        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.cross_attn =  CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k):
        x_q = self.norm_q(x_q + pos_q)
        x_k = self.norm_k(x_kv + pos_k)
        x_v = self.norm_v(x_kv)

        x = self.cross_attn(x_q, k=x_k, v=x_v)

        return x


class RegressorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.norm2_cross = norm_layer(dim)
        self.cross_attn =  CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        self.drop_path = DropPath(drop_path)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp_cross = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1_cross = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2_cross = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1_cross = nn.Parameter(torch.ones((dim)),requires_grad=False)
            self.gamma_2_cross = nn.Parameter(torch.ones((dim)),requires_grad=False)

    def forward(self, x_q, x_kv, pos_q, pos_k):
        x = x_q + self.drop_path(self.gamma_1_cross * self.cross_attn(self.norm1_q(x_q + pos_q), k=self.norm1_k(x_kv + pos_k), v=self.norm1_v(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.gamma_2_cross * self.mlp_cross(x))

        return x


class LatentRegressor(nn.Module):
    def __init__(self, embed_dim=768, regressor_depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0, norm_layer=None, init_values=None, init_std=0.02, model_type='cae'):
        super(LatentRegressor, self).__init__()
        self.model_type = model_type

        self.num_features = self.embed_dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, regressor_depth)]

        self.regressor_blocks = nn.ModuleList([
            RegressorBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values)
            for i in range(regressor_depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.init_std = init_std

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.regressor_blocks):
            rescale(layer.cross_attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp_cross.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked):
        for blk in self.regressor_blocks:
            x_masked = blk(x_masked, torch.cat([x_unmasked, x_masked], dim=1), pos_embed_masked, torch.cat([pos_embed_unmasked, pos_embed_masked], dim=1))

        if self.model_type != 'caev2':
            x_masked = self.norm(x_masked)

        return x_masked


class VitEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=6, num_heads=6, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None, init_std=0.02, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(H=img_size*2, W=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.transform = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_embed.patch_size[0], p2=self.patch_embed.patch_size[1])
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoder = BrainSpherePositionEncoder(embed_dim=self.embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, attn_head_dim=attn_head_dim
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std

        # init the model
        trunc_normal_(self.cls_token, std=self.init_std)
        self.apply(self._init_weights)

        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_num_layer(self):
        return len(self.blocks)

    def forward(self, x, bool_masked_pos):
        B, C, H, W = x.shape
        x_raw = x.detach()
        x_raw = self.transform(x_raw)
        x_pixel_target = x_raw[bool_masked_pos].reshape(B, -1, self.patch_embed.patch_size[0] ** 2)

        x = self.patch_embed(x)
        B, P, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)

        # position embedding
        patch_shape = self.patch_embed.patch_shape
        X, Y = np.meshgrid(np.linspace(0, patch_shape[1]-1, patch_shape[1]), np.linspace(0, patch_shape[1]-1, patch_shape[1]))
        X = np.expand_dims(X, axis=-1)
        X = repeat(X, 'H W C -> B H W C', B=B)
        X = (X - (patch_shape[0]-1) / 2) / ((patch_shape[0]-1) / 2) * np.pi
        Y = np.expand_dims(Y, axis=-1)
        Y = repeat(Y, 'H W C -> B H W C', B=B)
        Y = np.arcsin((Y - (patch_shape[1]-1) / 2) / ((patch_shape[1]-1) / 2))

        lh = np.zeros_like(X)
        lh_coords = np.concatenate([lh, X, Y], axis=-1)
        rh = np.ones_like(X)
        rh_coords = np.concatenate([rh, X, Y], axis=-1)
        coords = np.concatenate([lh_coords, rh_coords], axis=-3)

        pe = self.pos_encoder(coords).view(B, P, D).contiguous()

        x = x + pe
        x_unmasked = x[~bool_masked_pos].reshape(B, -1, D)
        x_unmasked = torch.cat((cls_tokens, x_unmasked), dim=-2)

        x_unmasked = self.pos_drop(x_unmasked)

        for blk in self.blocks:
            x_unmasked = blk(x_unmasked)

        x_unmasked = self.norm(x_unmasked)
        return x_unmasked, pe, x_pixel_target


class Decoder(nn.Module):
    def __init__(self, pixels_per_patch, embed_dim=768, decoder_depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=None, init_std=0.02):
        super(Decoder, self).__init__()
        self.num_features = self.embed_dim = embed_dim

        if decoder_depth > 0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
            self.decoder_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values
                )
                for i in range(decoder_depth)])
        else:
            self.decoder_blocks = None

        self.norm = norm_layer(embed_dim)
        self.to_pixel = nn.Linear(embed_dim, pixels_per_patch)

        self.init_std = init_std

        trunc_normal_(self.to_pixel.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.decoder_blocks is not None:
            for layer_id, layer in enumerate(self.decoder_blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, x_masked, pos_embed_masked):
        x_masked = x_masked + pos_embed_masked
        for blk in self.decoder_blocks:
            x_masked = blk(x_masked)
        x_masked = self.norm(x_masked)

        pixels = self.to_pixel(x_masked)
        return pixels


class BrainSRLViT(nn.Module):
    def __init__(self, vertices, sph_coords, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, attn_head_dim=None, init_std=0.02, decoder_embed_dim=768, regressor_depth=2,
                 decoder_num_heads=12, decoder_layer_scale_init_value=0.1, decoder_depth=4, model_type='cae', use_regressor=1, use_head=0, head_depth=1, fix_init_weight=False, **kwargs):
        super(BrainSRLViT, self).__init__()
        # if kwargs['args'].depth != depth: depth = kwargs['args'].depth
        # if kwargs['args'].regressor_depth != regressor_depth: regressor_depth = kwargs['args'].regressor_depth
        # if kwargs['args'].decoder_embed_dim != decoder_embed_dim: decoder_embed_dim = kwargs['args'].decoder_embed_dim
        # if kwargs['args'].decoder_depth != decoder_depth: decoder_depth = kwargs['args'].decoder_depth

        print('Encoder_depth: ', depth)
        print("Regressor_depth: ", regressor_depth)
        print("Decoder_embed_dim: ", decoder_embed_dim)
        print("Decoder_depth: ", decoder_depth)
        self.model_type = model_type

        self.vertices = vertices
        self.sph_coords = sph_coords
        self.index = self.calculate_2d_index(224)

        self.inverse_transform = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=2*img_size//patch_size, w=img_size//patch_size,
                                           p1=patch_size, p2=patch_size)

        self.encoder = VitEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth,
                                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim, init_std=init_std)

        self.init_std = init_std
        self.num_patches = self.encoder.num_patches

        self.use_regressor = use_regressor
        if self.use_regressor == 1:
            # alignment branch
            self.alignment_encoder = VitEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth,
                                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim, init_std=init_std)

            # context regressor
            self.regressor = LatentRegressor(embed_dim=decoder_embed_dim, regressor_depth=regressor_depth, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                             norm_layer=norm_layer, init_values=decoder_layer_scale_init_value, init_std=init_std, model_type=self.model_type)

            self._init_alignment_encoder()

        # regress is cross attention, mask tokens are queries
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=self.init_std)

        # decoder for reconstruction
        self.decoder = Decoder(pixels_per_patch=patch_size*patch_size, embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, num_heads=decoder_num_heads,
                               mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                               drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=decoder_layer_scale_init_value, init_std=init_std)

        # Attentive Probing Head
        self.use_head = use_head
        if self.use_head == 1:
            self._freeze_encoder()
            self.caption_query = nn.Parameter(torch.ones(1, 1, embed_dim))
            self.image_query = nn.Parameter(torch.ones(1, 1, embed_dim))
            self.attentive_head = nn.ModuleList([
                AttentiveBlock(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer)
                for i in range(head_depth)
            ])

            self.cap_linear = nn.Linear(embed_dim, 1024)
            self.img_linear = nn.Linear(embed_dim, 1024)

    def _init_alignment_encoder(self):
        # init the weights of alignment_encoder with those of backbone
        for param_encoder, param_alignment_encoder in zip(self.encoder.parameters(), self.alignment_encoder.parameters()):
            param_alignment_encoder.detach()
            param_alignment_encoder.data.copy_(param_encoder.data)
            param_alignment_encoder.requires_grad = False

    def alignment_parameter_update(self):
        # parameter update of the alignment_encoder network.
        for param_encoder, param_alignment_encoder in zip(self.encoder.parameters(), self.alignment_encoder.parameters()):
            param_alignment_encoder.data = param_encoder.data # completely copy

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _freeze_encoder(self):
        # freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        # freeze regressor
        if self.use_regressor == 1:
            for p in self.regressor.parameters():
                p.requires_grad = False
        # freeze decoder
        for p in self.decoder.parameters():
            p.requires_grad = False

    def calculate_2d_index(self, img_size):
        index = np.zeros((img_size*2, img_size)).astype(int)
        index[:img_size, :] = fMRI2grid(np.arange(self.vertices[0]), self.sph_coords[:self.vertices[0], 0],
                                   self.sph_coords[:self.vertices[0], 1], img_size)
        index[img_size:, :] = fMRI2grid(np.arange(self.vertices[1]) + self.vertices[0],
                                   self.sph_coords[self.vertices[0]:, 0],
                                   self.sph_coords[self.vertices[0]:, 1], img_size)

        return torch.from_numpy(index).to(dtype=torch.long)

    def forward_ae(self, input):
        input = input.permute(1, 0)
        x = input[self.index]
        x = x.permute(2, 0, 1)
        x = torch.unsqueeze(x, dim=1)

        B, C, H, W = x.shape
        bool_masked_pos = torch.zeros((B, self.encoder.num_patches), dtype=torch.bool).to(x.device)

        x_unmasked, pos_embed, _ = self.encoder(x, bool_masked_pos=bool_masked_pos)

        x_pixel_predict = self.decoder(x_unmasked[:, 1:, :], pos_embed)
        x_pixel_predict = self.inverse_transform(x_pixel_predict)

        return x_pixel_predict

    def forward(self, input, bool_masked_pos):
        input = input.permute(1, 0)
        x = input[self.index]
        x = x.permute(2, 0, 1)
        x = torch.unsqueeze(x, dim=1)
        
        B, C, H, W = x.shape
        x_unmasked, pos_embed, x_pixel_target = self.encoder(x, bool_masked_pos=bool_masked_pos)

        if self.use_regressor == 1:
            # Alignment branch
            with torch.no_grad():
                latent_target, _, _ = self.alignment_encoder(x, bool_masked_pos=(~bool_masked_pos))
                latent_target = latent_target[:, 1:, :] # remove class token

                self.alignment_parameter_update()
        else:
            latent_target = None

        '''
        Latent contextual regressor
        1. prepare masked, unmasked pos embed and masked embedding
        '''
        B, unmasked_P, D = x_unmasked.shape

        x_cls_token = x_unmasked[:, :1, :]
        x_unmasked = x_unmasked[:, 1:, :]

        pos_embed_masked = pos_embed[bool_masked_pos].reshape(B, -1, D)
        pos_embed_unmasked = pos_embed[~bool_masked_pos].reshape(B, -1, D)

        num_masked_patches = self.num_patches - unmasked_P + 1
        x_masked = self.mask_token.expand(B, num_masked_patches, -1)

        '''
        2. regress masked latent via regressor
        '''
        if self.use_regressor == 1:
            x_masked_predict = self.regressor(x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked)
            latent_predict = x_masked_predict
        else:
            x_masked_predict = x_masked
            latent_predict = None


        # decoder for reconstruction
        x_total = torch.cat((x_masked_predict, x_unmasked), dim=1)
        pos_embed_total = torch.cat((pos_embed_masked, pos_embed_unmasked), dim=1)
        pixel_predict = self.decoder(x_total, pos_embed_total)

        x_pixel_predict = pixel_predict[:, :num_masked_patches, :]

        return latent_predict, latent_target, x_pixel_predict, x_pixel_target

    def forward_feature(self, input, query_type):
        input = input.permute(1, 0)
        x = input[self.index]
        x = x.permute(2, 0, 1)
        x = torch.unsqueeze(x, dim=1)
        
        B, C, H, W = x.shape
        bool_masked_pos = torch.zeros((B, self.num_patches), dtype=torch.bool).to(x.device)
        x_unmasked, pos_embed, x_pixel_target = self.encoder(x, bool_masked_pos=bool_masked_pos)
        cls_token = x_unmasked[:, 0, :]
        x_unmasked = x_unmasked[:, 1:, :]

        if query_type == 'caption':
            query = repeat(self.caption_query, '1 1 D -> B 1 D', B=B)
            for blk in self.attentive_head:
                query = blk(query, x_unmasked, 0, pos_embed)
            feature = self.cap_linear(query)
        else:
            query = repeat(self.image_query, '1 1 D -> B 1 D', B=B)
            for blk in self.attentive_head:
                query = blk(query, x_unmasked, 0, pos_embed)
            feature = self.img_linear(query)
        feature = torch.squeeze(feature)
        return feature

    def forward_encoder(self, input):
        input = input.permute(1, 0)
        x = input[self.index]
        x = x.permute(2, 0, 1)
        x = torch.unsqueeze(x, dim=1)

        B, C, H, W = x.shape
        bool_masked_pos = torch.zeros((B, self.num_patches), dtype=torch.bool).to(x.device)
        x_unmasked, pos_embed, x_pixel_target = self.encoder(x, bool_masked_pos=bool_masked_pos)
        cls_token = x_unmasked[:, 0, :]
        x_unmasked = x_unmasked[:, 1:, :]

        return cls_token, x_unmasked


class VAE(nn.Module):
    def __init__(self, input_dim_A=768, input_dim_B=1024, latent_dim=64):
        super(VAE, self).__init__()

        # **Encoder (A → z)**
        self.encoder = nn.Sequential(
            nn.Linear(input_dim_A, 1024),  # 增加更大的隐藏层
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1024),  # 添加BatchNorm
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),  # 添加BatchNorm
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),  # 添加BatchNorm
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),  # 添加BatchNorm
        )
        self.fc_mu = nn.Linear(128, latent_dim)  # 均值层
        self.fc_logvar = nn.Linear(128, latent_dim)  # 方差层

        # **Decoder (z → B)**
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),  # 添加BatchNorm
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),  # 添加BatchNorm
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),  # 添加BatchNorm
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1024),  # 添加BatchNorm
            nn.Linear(1024, input_dim_B)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_A):
        x = self.encoder(x_A)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_B_hat = self.decoder(z)
        return x_B_hat, mu, logvar


class BrainCLIP(nn.Module):
    def __init__(self, vertices, sph_coords, img_size=224, patch_size=16, in_chans=1, embed_dim=1024, depth=6, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, attn_head_dim=None, init_std=0.02, type='vit', post_type='linear', encoder_dim=2*224*224,
                 Index_root=None):
        super(BrainCLIP, self).__init__()

        self.vertices = vertices
        self.sph_coords = sph_coords
        self.type = type
        self.post_type = post_type
        self.index = self.calculate_2d_index(224)

        if Index_root is not None:
            temp = np.loadtxt(Index_root)
            self.index = temp[temp != -1].astype(int)
            self.index = torch.from_numpy(self.index).to(dtype=torch.long)
            encoder_dim = len(self.index)

        # if Index_root is not None:
        #     temp = np.loadtxt(Index_root)
        #     if 'whole_brain' not in Index_root:
        #         self.index = temp[temp != -1].astype(int)
        #         self.index = torch.from_numpy(self.index).to(dtype=torch.long)
        #     else:
        #         temp = temp[temp != -1].astype(int)
        #         temp = torch.from_numpy(temp).to(dtype=torch.long)
        #         A = set(self.index.reshape(-1).numpy())
        #         B = set(temp.numpy())
        #         intersection = torch.tensor(list(A.intersection(B))).to(dtype=torch.long)
        #         self.index = intersection
        #         print(len(self.index))
        #
        #     encoder_dim = len(self.index)
        #
        # if encoder_dim > 35000 and 'whole_brain' in Index_root:
        #     indices = torch.randperm(encoder_dim)[:35000]
        #     self.index = self.index[indices]
        #     encoder_dim = len(self.index)

        if self.type == 'vit':
            self.encoder = VitEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth,
                                     num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                     norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim, init_std=init_std)
            self.num_patches = self.encoder.num_patches
            self.post = nn.Linear(embed_dim, 1024)
        else:
            self.encoder = nn.Linear(encoder_dim, embed_dim)
            if post_type == 'linear':
                self.post = nn.Linear(embed_dim, 1024)
            elif post_type == 'linear_res':
                self.post = nn.Linear(embed_dim, 1024)
                self.sc = nn.Linear(encoder_dim, 1024)
            elif post_type == 'mlp':
                self.post = Mlp(embed_dim, 2048, 1024)
            elif post_type == 'vae':
                self.post = VAE(embed_dim, 1024)

    def calculate_2d_index(self, img_size):
        index = np.zeros((img_size*2, img_size)).astype(int)
        index[:img_size, :] = fMRI2grid(np.arange(self.vertices[0]), self.sph_coords[:self.vertices[0], 0],
                                   self.sph_coords[:self.vertices[0], 1], img_size)
        index[img_size:, :] = fMRI2grid(np.arange(self.vertices[1]) + self.vertices[0],
                                   self.sph_coords[self.vertices[0]:, 0],
                                   self.sph_coords[self.vertices[0]:, 1], img_size)

        return torch.from_numpy(index).to(dtype=torch.long)

    def forward(self, input, i=None, j=None):
        if self.type == 'vit':
            input = input.permute(1, 0)
            x = input[self.index]
            x = x.permute(2, 0, 1)
            x = torch.unsqueeze(x, dim=1)

            B, C, H, W = x.shape
            bool_masked_pos = torch.zeros((B, self.num_patches), dtype=torch.bool).to(x.device)
            x_unmasked, pos_embed, x_pixel_target = self.encoder(x, bool_masked_pos=bool_masked_pos)

            feature = x_unmasked[:, 0, :]
            clip_feature = self.post(feature)
        else:
            input = input.permute(1, 0)
            x = input[self.index]
            if len(x.shape) == 3:
                x = x.permute(2, 0, 1)
                B, H, W = x.shape
            elif len(x.shape) == 2:
                x = x.permute(1, 0)
                B, D = x.shape

            if i is not None and j is not None:
                x = x[:, 16*i:16*i+16, 16*j:16*j+16]

            x = x.reshape(B, -1)
            feature = self.encoder(x)
            if self.post_type == 'vae':
                clip_feature, mu, logvar = self.post(feature)
                return feature, clip_feature, mu, logvar
            elif self.post_type == 'linear_res':
                clip_feature = self.sc(x)
            else:
                clip_feature = self.post(feature)
        return feature, clip_feature

    def forward_test(self, input, i, j):
        if self.type == 'vit':
            input = input.permute(1, 0)
            x = input[self.index]
            x = x.permute(2, 0, 1)
            x = torch.unsqueeze(x, dim=1)

            B, C, H, W = x.shape
            bool_masked_pos = torch.zeros((B, self.num_patches), dtype=torch.bool).to(x.device)
            x_unmasked, pos_embed, x_pixel_target = self.encoder(x, bool_masked_pos=bool_masked_pos)

            feature = torch.squeeze(x_unmasked[:, 0, :])
        else:
            input = input.permute(1, 0)
            x = input[self.index]
            x = x.permute(2, 0, 1)

            x[:, 16*i:16*i+16, 16*j:16*j+16] = 0

            B, H, W = x.shape
            x = x.reshape(B, -1)
            feature = self.encoder(x)
            clip_feature = self.post(feature)
        return feature, clip_feature


class CLIP_mapper(nn.Module):
    def __init__(self, in_dim, hemi_vertices, sph_coords, args, seq_len=73, out_dim=1024, depth=6, decoder_depth=6, drop_path_rate=0.):
        super(CLIP_mapper, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len

        self.Brain_CLIP = BrainCLIP(
            vertices=hemi_vertices,
            sph_coords=sph_coords,
            img_size=args.image_size,
            depth=args.depth,
            type=args.type,
            embed_dim=args.embed_dim,
            post_type=args.post_type
        )

        ckpt = torch.load(args.BrainCLIP_root, map_location=self.device)
        self.Brain_CLIP.load_state_dict(ckpt)
        if args.freeze_CLIP:
            self.Brain_CLIP.requires_grad_(False)

        self.linear = nn.Linear(in_dim, seq_len * out_dim)
        self.pe = torch.from_numpy(positional_encoding(seq_len, out_dim)).to(self.device, dtype=torch.float)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.encoder_block = nn.ModuleList([
            Block(
                dim=out_dim, num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=nn.LayerNorm,
                init_values=None, attn_head_dim=None
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(out_dim)

        self.decoder = Decoder(1024, embed_dim=1024, decoder_depth=decoder_depth, norm_layer=nn.LayerNorm)

        self.cls_token = np.load('/public/home/lishr2022/Project/Cross-modal/reconstruction/stable-diffusion-2-1-base/cls_token.npy')
        self.cls_token = torch.from_numpy(self.cls_token).to(self.device, dtype=torch.float)

    def forward(self, x):
        B, D = x.shape

        brain_feature, _ = self.Brain_CLIP(x)
        brain_feature = brain_feature / brain_feature.norm(dim=-1, keepdim=True)

        feature = self.linear(brain_feature)
        feature = feature.reshape(B, self.seq_len, -1)

        pe = repeat(self.pe, 'L D -> B L D', B = B)
        feature = feature + pe

        for blk in self.encoder_block:
            feature = blk(feature)
        feature = self.norm(feature)

        feature = self.decoder(feature, pe)

        feature = self.norm(feature)
        feature[:, :, :] = feature[:, :, :] * np.sqrt(1.0793) - 0.1682
        # adapt the distribution
        cls_token = repeat(self.cls_token, 'L D -> B L D', B=B)
        feature = torch.cat([cls_token, feature], dim=1)

        return feature


class prev_Structual_reconstruction(nn.Module):
    def __init__(self, num_vertices, vertices, coords, sph_coords, CLIP_model_root, embed_dim, decoder_depth, model_root,
                 behavior_in=8, behavior_hidden=16, final_visual_emb_dim=64, final_bert_emb_dim=64):
        super(Structual_reconstruction, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_vertices = num_vertices
        self.vertices = vertices
        self.sph_coords = sph_coords
        self.index = self.calculate_2d_index()

        self.encoder = EncodingModel(
            num_voxels=num_vertices,
            coords=coords,
            behavior_in=behavior_in,
            behavior_hidden=behavior_hidden,
            final_visual_emb_dim=final_visual_emb_dim,
            final_bert_emb_dim=final_bert_emb_dim,
            CLIP_model_root=CLIP_model_root
        )

        self.patch_embed = PatchEmbed(H=448, W=224, patch_size=16, in_chans=1, embed_dim=embed_dim)
        self.pos_encoder = BrainSpherePositionEncoder(embed_dim=embed_dim)

        self.decoder = Decoder(
            pixels_per_patch=384,
            embed_dim=embed_dim,
            decoder_depth=decoder_depth,
            norm_layer=nn.LayerNorm
        )

        self.post_transform = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=28, w=14, p1=8, p2=16)

        ckpt = torch.load(model_root, map_location=self.device)
        self.encoder.load_state_dict(ckpt)
        
        self._freeze_encoder()

    def calculate_2d_index(self):
        index = np.zeros((448, 224)).astype(int)
        index[:224, :] = fMRI2grid(np.arange(self.vertices[0]), self.sph_coords[:self.vertices[0], 0],
                                   self.sph_coords[:self.vertices[0], 1], 224)
        index[224:, :] = fMRI2grid(np.arange(self.vertices[1]) + self.vertices[0], self.sph_coords[self.vertices[0]:, 0],
                                   self.sph_coords[self.vertices[0]:, 1], 224)

        return torch.from_numpy(index).to(dtype=torch.long)

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, img, type='train'):
        B, C, H, W = img.shape
        img_condition = torch.zeros(B, 8).to(img.device, dtype=torch.float)

        if type == 'train':
            predict_response, _ = self.encoder.VisualEncode([None, img, None, None, None, img_condition])
        else:
            predict_response = self.encoder.VisualEncode([None, img, None, None, None, img_condition])

        predict_response = predict_response.permute(1, 0)
        predict_response_2d = predict_response[self.index]
        predict_response_2d = predict_response_2d.permute(2, 0, 1)
        predict_response_2d = torch.unsqueeze(predict_response_2d, dim=1)

        x = self.patch_embed(predict_response_2d)
        B, P, D = x.shape

        patch_shape = self.patch_embed.patch_shape
        X, Y = np.meshgrid(np.linspace(0, patch_shape[1] - 1, patch_shape[1]),
                           np.linspace(0, patch_shape[1] - 1, patch_shape[1]))
        X = np.expand_dims(X, axis=-1)
        X = repeat(X, 'H W C -> B H W C', B=B)
        X = (X - (patch_shape[0] - 1) / 2) / ((patch_shape[0] - 1) / 2) * np.pi
        Y = np.expand_dims(Y, axis=-1)
        Y = repeat(Y, 'H W C -> B H W C', B=B)
        Y = np.arcsin((Y - (patch_shape[1] - 1) / 2) / ((patch_shape[1] - 1) / 2))

        lh = np.zeros_like(X)
        lh_coords = np.concatenate([lh, X, Y], axis=-1)
        rh = np.ones_like(X)
        rh_coords = np.concatenate([rh, X, Y], axis=-1)
        coords = np.concatenate([lh_coords, rh_coords], axis=-3)

        pe = self.pos_encoder(coords).view(B, P, D).contiguous()

        out_img = self.decoder(x, pe)
        out_img = self.post_transform(out_img)

        return out_img


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class PR_Decoder(nn.Module):
    def __init__(self, in_dim, out_img_res, start_CHW=(64,7,7), n_conv_layers_ramp=3, n_chan=64, n_chan_output=3):
        super(PR_Decoder, self).__init__()

        self.start_CHW = start_CHW
        upsample_scale_factor = (out_img_res / start_CHW[-1]) ** (1/n_conv_layers_ramp)
        self.input_fc = nn.Linear(in_dim, np.prod(start_CHW))

        kernel_size = 5

        pad_size = int(kernel_size // 2)
        # 64 32 16 8 4
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=upsample_scale_factor, mode='bicubic'),
                    nn.ReflectionPad2d(pad_size),
                    nn.Conv2d(start_CHW[0], n_chan, kernel_size),
                    nn.GroupNorm(16, n_chan),
                    MemoryEfficientSwish()
                )
            ] + \
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=upsample_scale_factor, mode='bicubic'),
                    nn.ReflectionPad2d(pad_size),
                    nn.Conv2d(n_chan, n_chan, kernel_size),
                    nn.GroupNorm(16, n_chan),
                    MemoryEfficientSwish()
                ) for block_index in range(n_conv_layers_ramp-1)
            ] + \
            [
                nn.Sequential(
                    nn.Conv2d(n_chan, n_chan, kernel_size, padding=pad_size),
                    MemoryEfficientSwish(),
                    nn.BatchNorm2d(n_chan)
                ) for _ in range(0)
            ]
        )

        self.top = nn.Sequential(
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(n_chan, n_chan_output, kernel_size),
            nn.Sigmoid()
        )

        self.trainable = [self.input_fc, self.blocks, self.top]

    def forward(self, x):
        x = self.input_fc(x)
        x = x.view(-1, *self.start_CHW)

        for block_index, block in enumerate(self.blocks):
            x = block(x)

        x = self.top(x)

        return x


class pr_decoder_block(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, group=16):
        super(pr_decoder_block, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.pad_size = int(kernel_size // 2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.pad = nn.ReflectionPad2d(self.pad_size)
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size)
        self.norm = nn.GroupNorm(group, out_chan)
        self.ac = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.upsample(x)
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.ac(x)

        return x


class Structual_reconstruction(nn.Module):
    def __init__(self, num_vertices, vertices, coords, sph_coords, CLIP_model_root, embed_dim, decoder_depth,
                 model_root, selected_voxel_root, n_chan,
                 behavior_in=8, behavior_hidden=16, final_visual_emb_dim=64, final_bert_emb_dim=64):
        super(Structual_reconstruction, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_chan = n_chan
        self.num_vertices = num_vertices
        self.vertices = vertices
        self.sph_coords = sph_coords
        # self.high_index = self.calculate_2d_index(os.path.join(selected_voxel_root, 'selected_high_visual_cortex.txt'))
        # self.low_index = self.calculate_2d_index(os.path.join(selected_voxel_root, 'selected_primary_visual_cortex.txt'))
        self.index = self.calculate_2d_index(os.path.join(selected_voxel_root, 'selected_visual.txt'))

        self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        self.encoder = EncodingModel(
            num_voxels=300245,
            coords=coords,
            behavior_in=behavior_in,
            behavior_hidden=behavior_hidden,
            final_visual_emb_dim=final_visual_emb_dim,
            final_bert_emb_dim=final_bert_emb_dim,
            CLIP_model_root=CLIP_model_root
        )

        self.img_chan = [64, 64, 64, 64, 32, 16]
        self.latent_chan = [64, 64, 32, 8, 4]
        self.group = [16, 16, 16, 8, 4]
        # self.img_chan = [64, 32]
        # self.latent_chan = [64, 32]
        # self.group = [16, 16]
        kernel_size = 3
        pad_size = int(kernel_size // 2)

        self.fc1 = nn.Linear(len(self.index), self.latent_chan[0] * 7 * 7)
        self.fc2 = nn.Linear(len(self.index), self.latent_chan[1] * 14 * 14)
        self.conv2 = nn.Conv2d(self.latent_chan[1], self.img_chan[1], 1)
        self.fc3 = nn.Linear(len(self.index), self.latent_chan[2] * 28 * 28)
        self.conv3 = nn.Conv2d(self.latent_chan[2], self.img_chan[2], 1)
        self.fc4 = nn.Linear(len(self.index), self.latent_chan[3] * 56 * 56)
        self.conv4 = nn.Conv2d(self.latent_chan[3], self.img_chan[3], 1)
        self.fc5 = nn.Linear(len(self.index), self.latent_chan[4] * 112 * 112)
        self.conv5 = nn.Conv2d(self.latent_chan[4], self.img_chan[4], 1)
        self.ac = nn.LeakyReLU(0.3)

        # 256 7 7 -> 512 14 14
        self.dec_block1 = pr_decoder_block(self.img_chan[0], self.img_chan[1], kernel_size, group=self.group[0])
        # 512 14 14 -> 256 28 28
        self.dec_block2 = pr_decoder_block(self.img_chan[1], self.img_chan[2], kernel_size, group=self.group[1])
        # 256 28 28 -> 128 56 56
        self.dec_block3 = pr_decoder_block(self.img_chan[2], self.img_chan[3], kernel_size, group=self.group[2])
        # 128 56 56 -> 64 112 112
        self.dec_block4 = pr_decoder_block(self.img_chan[3], self.img_chan[4], kernel_size, group=self.group[3])
        # 64 112 112 -> 32 224 224
        self.dec_block5 = pr_decoder_block(self.img_chan[4], self.img_chan[5], kernel_size, group=self.group[4])
        # 32 224 224 -> 3 224 224
        self.top = nn.Sequential(
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(self.img_chan[-1], 3, kernel_size),
            nn.Sigmoid()
        )

        # 初始化Encoding model
        ckpt = torch.load(model_root, map_location=self.device)
        self.encoder.load_state_dict(ckpt)
        self.encoder.requires_grad_(False)

    def calculate_2d_index(self, root):
        # index = np.zeros((img_size*2, img_size)).astype(int)
        # index[:img_size, :] = fMRI2grid(np.arange(self.vertices[0]), self.sph_coords[:self.vertices[0], 0],
        #                            self.sph_coords[:self.vertices[0], 1], img_size)
        # index[img_size:, :] = fMRI2grid(np.arange(self.vertices[1]) + self.vertices[0],
        #                            self.sph_coords[self.vertices[0]:, 0],
        #                            self.sph_coords[self.vertices[0]:, 1], img_size)
        #
        # return torch.from_numpy(index).to(dtype=torch.long)
        temp = np.loadtxt(root)
        index = temp[temp != -1].astype(int)
        return torch.from_numpy(index).to(dtype=torch.long)

    def forward_decoder(self, x):
        predict_img = self.ac(self.fc1(x).view(-1, self.latent_chan[0], 7, 7))
        predict_img = self.dec_block1(predict_img) + self.ac(self.conv2(self.fc2(x).view(-1, self.latent_chan[1], 14, 14)))
        predict_img = self.dec_block2(predict_img) + self.ac(self.conv3(self.fc3(x).view(-1, self.latent_chan[2], 28, 28)))
        predict_img = self.dec_block3(predict_img) + self.ac(self.conv4(self.fc4(x).view(-1, self.latent_chan[3], 56, 56)))
        predict_img = self.dec_block4(predict_img) + self.ac(self.conv5(self.fc5(x).view(-1, self.latent_chan[4], 112, 112)))
        predict_img = self.dec_block5(predict_img)
        predict_img = self.top(predict_img)
        # predict_img = self.fc1(x).view(-1, self.latent_chan[0], 7, 7)
        # predict_img = self.dec_block1(predict_img)+ self.ac(self.conv2(self.fc2(x).view(-1, self.latent_chan[1], 14, 14)))
        # predict_img = self.dec_block2(predict_img)
        # predict_img = self.dec_block3(predict_img)
        # predict_img = self.dec_block4(predict_img)
        # predict_img = self.dec_block5(predict_img)
        # predict_img = self.top(predict_img)

        return predict_img

    def forward_fake(self, img, type='train'):
        B, C, H, W = img.shape
        img_condition = torch.zeros(B, 8).to(img.device, dtype=torch.float)

        img_encode = self.norm(img)

        if type == 'train':
            predict_response, _ = self.encoder.VisualEncode([None, img_encode, None, None, None, img_condition])
        else:
            predict_response = self.encoder.VisualEncode([None, img_encode, None, None, None, img_condition])

        predict_response = predict_response.permute(1, 0)

        # high_level_feature = predict_response[self.high_index]
        # high_level_feature = high_level_feature.permute(1, 0)
        #
        # low_level_feature = predict_response[self.low_index]
        # low_level_feature = low_level_feature.permute(1, 0)

        feature = predict_response[self.index]
        feature = feature.permute(1, 0)

        # forward decoder
        predict_img = self.forward_decoder(feature)

        return predict_img

    def forward_real(self, x, type='train'):
        B, num_vertices = x.shape
        
        x = x.permute(1, 0)

        # high_level_feature = x[self.high_index]
        # high_level_feature = high_level_feature.permute(1, 0)
        #
        # low_level_feature = x[self.low_index]
        # low_level_feature = low_level_feature.permute(1, 0)

        feature = x[self.index]
        feature = feature.permute(1, 0)

        # forward decoder
        predict_img = self.forward_decoder(feature)

        return predict_img
