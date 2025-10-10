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
from timm.layers.norm import LayerNorm2d, LayerNorm
from timm.models.convnext import ConvNeXtBlock
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import ViTFeatureExtractor, ViTModel, BertTokenizer, BertModel
from cn_clip.clip import load_from_name, image_transform, load_from_name, _tokenizer
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, R2Score, MetricCollection
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)

    @property
    def weight(self):
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return 0


class MonkeyLoRALinear(nn.Module):
    def __init__(self, fc: nn.Linear, rank=4, lora_scale=1):
        super().__init__()
        if rank > min(fc.in_features, fc.out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(fc.in_features, fc.out_features)}"
            )
        if not isinstance(fc, nn.Linear):
            raise ValueError(
                f"MonkeyLoRALinear only support nn.Linear, but got {type(fc)}"
            )

        self.fc = fc
        self.rank = rank
        self.lora_scale = lora_scale

        in_features = fc.in_features
        out_features = fc.out_features
        self.fc_lora = LoRALinearLayer(in_features, out_features, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc(hidden_states) + self.lora_scale * self.fc_lora(
            hidden_states
        )
        return hidden_states

    @property
    def weight(self):
        return self.fc.weight + self.lora_scale * self.fc_lora.weight

    @property
    def bias(self):
        return self.fc.bias


class AdaLNZeroPatch(nn.Module):
    def __init__(self, embed_dim, d_c=64, adaln_scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_c = d_c
        self.adaln_scale = adaln_scale

        # for condition (behavior data)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(self.d_c, 6 * self.embed_dim, bias=False),
            nn.Tanh(),
        )

        nn.init.zeros_(self.adaLN_modulation[0].weight)

    def forward(self, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (
            self.adaLN_modulation(c) * self.adaln_scale
        ).chunk(6, dim=1)

        scale_msa = scale_msa + 1
        gate_msa = gate_msa + 1
        scale_mlp = scale_mlp + 1
        gate_mlp = gate_mlp + 1

        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLNLoRACLIPResidualAttentionBlock(nn.Module):
    def __init__(self, block, lora_scale=1.0, rank=4, d_c=16, adaln_scale=1.0):
        super().__init__()
        self.block = block
        self.lora_scale = lora_scale
        self.rank = rank
        self.d_c = d_c

        self.embed_dim = self.block.attn.embed_dim

        ### these are nn.Parameter, can not be monkey-patched
        # patch qkv
        self.w_clone = self.block.attn.in_proj_weight.clone()
        self.w_clone.requires_grad_(False)
        self.attn_in_proj_lora = LoRALinearLayer(
            self.embed_dim, 3 * self.embed_dim, rank=rank
        )

        ### these are nn.Linear, can be monkey-patched
        self.block.attn.out_proj = MonkeyLoRALinear(
            self.block.attn.out_proj, rank=rank, lora_scale=lora_scale
        )
        self.block.mlp[0] = MonkeyLoRALinear(
            self.block.mlp[0], rank=rank, lora_scale=lora_scale
        )
        self.block.mlp[2] = MonkeyLoRALinear(
            self.block.mlp[2], rank=rank, lora_scale=lora_scale
        )

        # for condition (behavior data)
        self.adaLN = AdaLNZeroPatch(self.embed_dim, d_c=d_c, adaln_scale=adaln_scale)

    def forward(
        self,
        q_x: torch.Tensor,
        c: Optional[torch.Tensor] = None
    ):
        # lora patch qkv
        self.block.attn.in_proj_weight.data = (
            self.w_clone.to(q_x.device)
            + self.lora_scale * self.attn_in_proj_lora.weight
        )

        # conditioning can be None
        bsz = q_x.shape[1]
        if c is None:
            c = torch.zeros(bsz, self.d_c, device=q_x.device, dtype=q_x.dtype)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(c)

        # attention with adaLN, LoRA is applied to weight
        x = q_x + gate_msa.unsqueeze(0) * self.block.attention(
                self.modulate(self.block.ln_1(q_x), shift_msa, scale_msa)
            )

        x = x + gate_mlp.unsqueeze(0) * self.block.mlp(self.modulate(self.block.ln_2(x), shift_mlp, scale_mlp))

        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * scale.unsqueeze(0) + shift.unsqueeze(0)


class AdaLNLoRAResidualAttentionBlock(nn.Module):
    def __init__(self, block, lora_scale=1.0, rank=4, d_c=16, adaln_scale=1.0):
        super().__init__()
        self.block = block
        self.lora_scale = lora_scale
        self.rank = rank
        self.d_c = d_c

        self.embed_dim = self.block.attention.attention.query.in_features

        # patch qkv
        self.block.attention.attention.query = MonkeyLoRALinear(
            self.block.attention.attention.query, rank=rank, lora_scale=lora_scale
        )
        self.block.attention.attention.key = MonkeyLoRALinear(
            self.block.attention.attention.key, rank=rank, lora_scale=lora_scale
        )
        self.block.attention.attention.value = MonkeyLoRALinear(
            self.block.attention.attention.value, rank=rank, lora_scale=lora_scale
        )

        # patch proj
        self.block.attention.output.dense = MonkeyLoRALinear(
            self.block.attention.output.dense, rank=rank, lora_scale=lora_scale
        )

        # patch intermediate
        self.block.intermediate.dense = MonkeyLoRALinear(
            self.block.intermediate.dense, rank=rank, lora_scale=lora_scale
        )

        # patch output
        self.block.output.dense = MonkeyLoRALinear(
            self.block.output.dense, rank=rank, lora_scale=lora_scale
        )

        # for condition (behavior data)
        self.adaLN = AdaLNZeroPatch(self.embed_dim, d_c=d_c, adaln_scale=adaln_scale)

    def forward(
        self,
        q_x: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        c: Optional[torch.Tensor] = None
    ):
        bsz = q_x.shape[1]
        if c is None:
            c = torch.zeros(bsz, self.d_c, device=q_x.device, dtype=q_x.dtype)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(c)

        self_attention_outputs = self.block.attention(self.modulate(self.block.layernorm_before(q_x), shift_msa, scale_msa))
        attention_output = self_attention_outputs[0] * gate_msa.unsqueeze(1)
        outputs = self_attention_outputs[1:]

        hidden_states = attention_output + q_x

        layer_output = gate_mlp.unsqueeze(1) * self.block.output(self.block.intermediate(self.modulate(self.block.layernorm_after(hidden_states), shift_mlp, scale_mlp)), hidden_states)
        outputs = (layer_output,) + outputs
        return outputs

    @staticmethod
    def modulate(x, shift, scale):
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)


def maxavg_globalpool2d(x):
    out = torch.cat([F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)], dim=1)
    out = out.squeeze(-1).squeeze(-1)
    return out


class AdaLNLoRACLIPBertLayer(nn.Module):
    def __init__(self, layer, lora_scale=1.0, rank=4, d_c=16, adaln_scale=1.0):
        super().__init__()
        self.layer = layer
        self.lora_scale = lora_scale
        self.rank = rank
        self.d_c = d_c

        self.hidden_dim, self.all_head_size = self.layer.attention.self.query.weight.T.shape

        # patch BertAttention - BertSelfAttention
        self.layer.attention.self.query = MonkeyLoRALinear(
            self.layer.attention.self.query, rank=rank, lora_scale=lora_scale
        )
        self.layer.attention.self.key = MonkeyLoRALinear(
            self.layer.attention.self.key, rank=rank, lora_scale=lora_scale
        )
        self.layer.attention.self.value = MonkeyLoRALinear(
            self.layer.attention.self.value, rank=rank, lora_scale=lora_scale
        )

        # patch BertAttention - BertSelfOutput
        self.layer.attention.output.dense = MonkeyLoRALinear(
            self.layer.attention.output.dense, rank=rank, lora_scale=lora_scale
        )

        # patch BertIntermediate
        self.layer.intermediate.dense = MonkeyLoRALinear(
            self.layer.intermediate.dense, rank=rank, lora_scale=lora_scale
        )

        # patch BertOutput
        self.layer.output.dense = MonkeyLoRALinear(
            self.layer.output.dense, rank=rank, lora_scale=lora_scale
        )

        self.adaLN = AdaLNZeroPatch(self.hidden_dim, d_c=d_c, adaln_scale=adaln_scale)

    def forward(
        self,
        q_x: torch.Tensor,
        attention_mask = None,
        head_mask = None,
        c: Optional[torch.Tensor] = None
    ):
        bsz = q_x.shape[0]
        if c is None:
            c = torch.zeros(bsz, self.d_c, device=q_x.device, dtype=q_x.dtype)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(c)

        # calculate BertAttention
        x = self.layer.attention.output.LayerNorm(q_x + gate_msa.unsqueeze(1) * \
            self.layer.attention.output.dropout(
                self.layer.attention.output.dense(
                    self.layer.attention.self(
                        self.modulate(q_x, shift_msa, scale_msa), attention_mask, head_mask)[0])))

        # calculate Bert Intermediate & BertOutput
        x = self.layer.output.LayerNorm(x + gate_mlp.unsqueeze(1) *
            self.layer.output.dropout(
                self.layer.output.dense(
                    self.layer.intermediate(
                        self.modulate(x, shift_mlp, scale_mlp)))))

        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)


class AdaLNLoRABertLayer(nn.Module):
    def __init__(self, layer, lora_scale=1.0, rank=4, d_c=16, adaln_scale=1.0):
        super(AdaLNLoRABertLayer, self).__init__()
        self.layer = layer
        self.lora_scale = lora_scale
        self.rank = rank
        self.d_c = d_c

        self.hidden_dim = self.layer.attention.self.query.in_features

        # patch BertAttention - BertSelfAttention
        self.layer.attention.self.query = MonkeyLoRALinear(
            self.layer.attention.self.query, rank=rank, lora_scale=lora_scale
        )
        self.layer.attention.self.key = MonkeyLoRALinear(
            self.layer.attention.self.key, rank=rank, lora_scale=lora_scale
        )
        self.layer.attention.self.value = MonkeyLoRALinear(
            self.layer.attention.self.value, rank=rank, lora_scale=lora_scale
        )

        # patch BertAttention - BertSelfOutput
        self.layer.attention.output.dense = MonkeyLoRALinear(
            self.layer.attention.output.dense, rank=rank, lora_scale=lora_scale
        )

        # patch BertIntermediate
        self.layer.intermediate.dense = MonkeyLoRALinear(
            self.layer.intermediate.dense, rank=rank, lora_scale=lora_scale
        )

        # patch BertOutput
        self.layer.output.dense = MonkeyLoRALinear(
            self.layer.output.dense, rank=rank, lora_scale=lora_scale
        )

        self.adaLN = AdaLNZeroPatch(self.hidden_dim, d_c=d_c, adaln_scale=adaln_scale)

    def forward(
        self,
        q_x: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        c: Optional[torch.Tensor] = None
    ):
        bsz = q_x.shape[0]
        if c is None:
            c = torch.zeros(bsz, self.d_c, device=q_x.device, dtype=q_x.dtype)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(c)

        # calculate BertAttention
        self_outputs = self.layer.attention.self(self.modulate(q_x, shift_msa, scale_msa), attention_mask, head_mask)
        attention_output = self.layer.attention.output.LayerNorm(q_x + gate_msa.unsqueeze(1) * \
                                self.layer.attention.output.dropout(
                                    self.layer.attention.output.dense(self_outputs[0])))

        # calculate Bert Intermediate & BertOutput
        layer_output = self.layer.output.LayerNorm(attention_output + gate_mlp.unsqueeze(1) *
            self.layer.output.dropout(
                self.layer.output.dense(
                    self.layer.intermediate(
                        self.modulate(attention_output, shift_mlp, scale_mlp)))))
        outputs = (layer_output,) + self_outputs[1:]
        return outputs

    @staticmethod
    def modulate(x, shift, scale):
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)


class AdaLNLoRAViT(nn.Module):
    def __init__(self, ViT_model_root, lora_scale=1.0, rank=4, d_c=16, adaln_scale=1.0):
        super(AdaLNLoRAViT, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = ViTModel.from_pretrained(ViT_model_root).to(self.device)
        self.visual = model
        self.visual.requires_grad_(False)
        self.visual = self.inject_lora_and_adaln_vit(
            self.visual,
            lora_scale=lora_scale,
            rank=rank,
            d_c=d_c,
            adaln_scale=adaln_scale
        )
        self.close_dropout(self.visual)

    @staticmethod
    def inject_lora_and_adaln_vit(model, lora_scale=1.0, rank=4, d_c=16, adaln_scale=1.0):
        encoder = model.encoder
        for i in range(len(encoder.layer)):
            block = encoder.layer[i]
            lora_block = AdaLNLoRAResidualAttentionBlock(
                block,
                rank=rank,
                lora_scale=lora_scale,
                d_c=d_c,
                adaln_scale=adaln_scale
            )
            model.encoder.layer[i] = lora_block
        return model

    @staticmethod
    def close_dropout(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    def get_visual_intermediate_layers(
        self,
        x,
        n: List[int] = [1, 2, 4, 8, 14, 20, 26, 32],
        c: Optional[torch.Tensor] = None,
        reshape=True,
        mask_ratio: float = 0.0
    ):
        x = self.visual.embeddings(x)

        output_dict = {}
        cls_dict = {}
        for i, block in enumerate(self.visual.encoder.layer):
            x = block(x, c=c)[0]
            if i+1 not in n:
                continue
            x_save = x.clone()
            x_save = x_save.permute(1, 0, 2)
            cls_dict[str(i+1)] = x_save[0, :, :]
            if reshape == True:
                x_save = x_save[1:, :, :]
                p = int(np.sqrt(x_save.shape[0]))
                x_save = rearrange(x_save, '(p1 p2) b d -> b d p1 p2', p1=p, p2=p)
            output_dict[str(i+1)] = x_save

        x = self.visual.layernorm(x)

        return output_dict, cls_dict, x


class AdaLNLoraBert(nn.Module):
    def __init__(self, Bert_model_root, lora_scale=1.0, rank=4, d_c=16, adaln_scale=1.0):
        super(AdaLNLoraBert, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = BertTokenizer.from_pretrained(Bert_model_root)
        self.bert = BertModel.from_pretrained(Bert_model_root).to(self.device)
        self.bert.requires_grad_(False)
        self.bert = self.inject_lora_and_adaln_bert(
            self.bert,
            lora_scale=lora_scale,
            rank=rank,
            d_c=d_c,
            adaln_scale=adaln_scale
        )
        self.close_dropout(self.bert)

    @staticmethod
    def inject_lora_and_adaln_bert(model, lora_scale=1.0, rank=4, d_c=16, adaln_scale=1.0):
        encoder = model.encoder
        for i in range(len(encoder.layer)):
            layer = encoder.layer[i]
            lora_layer = AdaLNLoRABertLayer(
                layer,
                rank=rank,
                lora_scale=lora_scale,
                d_c=d_c,
                adaln_scale=adaln_scale
            )
            model.encoder.layer[i] = lora_layer
        return model

    @staticmethod
    def close_dropout(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    def get_bert_intermediate_layers(
        self,
        x,
        n: List[int] = [1, 2, 4, 8, 12, 16, 20, 24],
        c: Optional[torch.Tensor] = None
    ):
        dtype = next(self.bert.parameters()).dtype

        inputs = self.tokenizer(x, return_tensors='pt', padding='max_length', max_length=52, truncation=False).to(self.device)
        input_shape = inputs['input_ids'].size()

        batch_size, seq_length = input_shape
        past_key_values_length = 0

        attention_mask = inputs['attention_mask']
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min

        embedding_output = self.bert.embeddings(
            input_ids=inputs['input_ids'],
            position_ids=None,
            token_type_ids=inputs['token_type_ids'],
            inputs_embeds=None,
            past_key_values_length=0
        )

        output_dict = {}
        cls_dict = {}
        for i, layer in enumerate(self.bert.encoder.layer):
            embedding_output = layer(embedding_output, attention_mask=extended_attention_mask, c=c)
            embedding_output = embedding_output[0]
            if i+1 not in n:
                continue
            x_save = embedding_output[:, 1:, :]
            output_dict[str(i+1)] = x_save
            cls_dict[str(i+1)] = embedding_output[:, 0, :]

        return output_dict, cls_dict, embedding_output


class AdaLNLoRACLIP(nn.Module):
    def __init__(self, CLIP_model_root, lora_scale=1.0, rank=4, d_c=16, adaln_scale=1.0, encode_type='visual'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CN-CLIP模型
        model, _ = load_from_name("ViT-H-14", device=self.device, download_root=CLIP_model_root)

        # visual
        if encode_type == 'visual' or encode_type == 'all':
            self.visual = model.visual
            self.visual.requires_grad_(False)
            self.visual = self.inject_lora_and_adaln_clip_vit(
                self.visual,
                lora_scale=lora_scale,
                rank=rank,
                d_c=d_c,
                adaln_scale=adaln_scale
            )
            self.close_dropout(self.visual)

        # caption
        if encode_type == 'caption' or encode_type == 'all':
            self.tokenizer = model.tokenizer
            self.bert = model.bert
            self.bert.requires_grad_(False)
            self.bert = self.inject_lora_and_adaln_clip_bert(
                self.bert,
                lora_scale=lora_scale,
                rank=rank,
                d_c=d_c,
                adaln_scale=adaln_scale
            )
            self.text_projection = model.text_projection
            self.close_dropout(self.bert)

        self.logit_scale = model.logit_scale

    @property
    def dtype(self):
        return torch.float

    @staticmethod
    def inject_lora_and_adaln_clip_vit(model, lora_scale=1.0, rank=4, d_c=16, adaln_scale=1.0):
        transformer = model.transformer
        for i in range(len(transformer.resblocks)):
            block = transformer.resblocks[i]
            lora_block = AdaLNLoRACLIPResidualAttentionBlock(
                block,
                rank=rank,
                lora_scale=lora_scale,
                d_c=d_c,
                adaln_scale=adaln_scale
            )
            model.transformer.resblocks[i] = lora_block
        return model

    @staticmethod
    def inject_lora_and_adaln_clip_bert(model, lora_scale=1.0, rank=4, d_c=16, adaln_scale=1.0):
        encoder = model.encoder
        for i in range(len(encoder.layer)):
            layer = encoder.layer[i]
            lora_layer = AdaLNLoRACLIPBertLayer(
                layer,
                rank=rank,
                lora_scale=lora_scale,
                d_c=d_c,
                adaln_scale=adaln_scale
            )
            model.encoder.layer[i] = lora_layer
        return model

    @staticmethod
    def close_dropout(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    def encode_image(self, image, c, mask_ratio=0):
        x = image
        x = self.visual.conv1(x) # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        if mask_ratio != 0:
            x = self.visual.random_masking(x, mask_ratio)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2) # [*, Length, D] -> [Length, *, D]

        for i, r in enumerate(self.visual.transformer.resblocks):
            x = r(x, c=c)
        x = x.permute(1, 0, 2)

        x = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x

    def encode_text(self, text):
        pad_index = self.tokenizer.vocab['[PAD]']
        attn_mask = text.ne(pad_index).type(self.dtype)
        x = self.bert(text, attention_mask=attn_mask)[0].type(self.dtype) # [batch_size, seq_length, hidden_size]
        return x[:, 0, :] @ self.text_projection

    def get_visual_intermediate_layers(
            self,
            x,
            n: List[int] = [1, 2, 4, 8, 14, 20, 26, 32],
            c: Optional[torch.Tensor] = None,
            reshape=True,
            mask_ratio: float = 0.0
    ):
        x = self.visual.conv1(x) # shape = [*, width, grid, grid]
        if len(x.shape) == 3:
            x = x.reshape(1, x.shape[0], -1)
        else:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        if mask_ratio != 0:
            x = self.visual.random_masking(x, mask_ratio)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2) # [*, Length, D] -> [Length, *, D]

        output_dict = {}
        cls_dict = {}
        for i, r in enumerate(self.visual.transformer.resblocks):
            x = r(x, c) # [1 + grid ** 2, B, D]
            if i+1 not in n:
                continue
            x_save = x.clone()
            cls_dict[str(i+1)] = x_save[0, :, :]
            if reshape == True:
                x_save = x_save[1:, :, :] # [grid ** 2, B, D]
                p = int(np.sqrt(x_save.shape[0]))
                x_save = rearrange(x_save, '(p1 p2) b d -> b d p1 p2', p1=p, p2=p)
            output_dict[str(i+1)] = x_save

        x = x.permute(1, 0, 2)

        x = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return output_dict, cls_dict, x

    def get_bert_intermediate_layers(
            self,
            x,
            n: List[int] = [1, 2, 4, 8, 12, 16, 20, 24],
            c: Optional[torch.Tensor] = None
    ):
        pad_index = self.tokenizer.vocab['[PAD]']
        attention_mask = x.ne(pad_index).type(x.dtype)
        token_type_ids = None
        position_ids = None
        head_mask = None

        if attention_mask is None:
            attention_mask = torch.ones_like(x)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(x)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.bert.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.bert.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.bert.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.bert.config.num_hidden_layers

        embedding_output = self.bert.embeddings(x, position_ids=position_ids, token_type_ids=token_type_ids)

        # calculate BertEncoder
        output_dict = {}
        cls_dict = {}
        for i, layer in enumerate(self.bert.encoder.layer):
            embedding_output = layer(embedding_output, extended_attention_mask, head_mask[i], c)
            if i+1 not in n:
                continue
            x_save = embedding_output[:, 1:, :]
            output_dict[str(i+1)] = x_save
            cls_dict[str(i+1)] = embedding_output[:, 0, :]

        final_outputs = embedding_output[:, 0, :] @ self.text_projection

        return output_dict, cls_dict, final_outputs


class PositionEncoding(nn.Module):
    def __init__(self, max_step=1000, features=32, periods=10000):
        super().__init__()
        self.max_step = max_step
        self.features = features
        self.periods = periods

    @torch.no_grad()
    def forward(self, points):
        low = points.min(0).values
        high = points.max(0).values
        steps = high - low
        steps *= self.max_step / steps.max()
        pe = self.point_pe(points, low, high, steps, self.features, self.periods)
        return pe

    def point_pe(self, points, low=0, high=1, steps=100, features=32, periods=10000):
        positions = (points - low).mul_(steps / (high - low))
        return self.sinusoidal(positions, features, periods).flatten(-3)

    def sinusoidal(self, positions, features=32, periods=10000):
        dtype = positions.dtype if positions.is_floating_point() else None
        kwargs = dict(device=positions.device, dtype=dtype)
        omega = torch.logspace(0, 1 / features - 1, features, periods, **kwargs)
        fraction = omega * positions.unsqueeze(-1)
        return torch.stack((fraction.sin(), fraction.cos()), dim=-1)


def build_coords_mlp(in_dim, out_dim, hidden_dim=256, depth=3, act_fn=nn.Identity()):
    assert depth >= 2
    modules = []
    modules.append(PositionEncoding(max_step=1000, features=32, periods=1000))

    in_dim = in_dim * 32 * 2
    for i in range(depth - 1):
        modules.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
        modules.append(act_fn)
    modules.append(nn.Linear(hidden_dim, out_dim))
    modules.append(act_fn)
    return nn.Sequential(*modules)


class CachedCoordsMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, depth=3, act_fn=nn.Identity()):
        super(CachedCoordsMLP, self).__init__()
        self.mlp = build_coords_mlp(in_dim, out_dim, hidden_dim=hidden_dim, depth=depth, act_fn=act_fn)
        self.cache = None

    def forward(self, coords, voxel_indices):
        if self.training and self.is_req_grad:
            self.cache = None
            return self.mlp(coords[voxel_indices])
        else:
            with torch.no_grad():
                if self.cache is None:
                    self.cache = self.mlp(coords)
                return self.cache[voxel_indices]
    @property
    def is_req_grad(self):
        return next(self.parameters()).requires_grad


class BehaviorEmbed(nn.Module):
    def __init__(self, in_dim, dim, dropout=0.2):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU()
        )

        self.mlp = Mlp(dim, out_features=dim)

        self.dropout = nn.Sequential(
            nn.Unflatten(1, (dim, 1)),
            nn.Dropout1d(dropout),
            nn.Flatten(1, -1)
        )

    def forward(self, c):
        if c is not None:
            c = self.embed(c)
            c = self.mlp(c)
            c = self.dropout(c)
        return c


class SimpleConvBlocks(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        depth=3,
        kernel_size=5,
        max_dim=1024,
        stride=1,
        padding="same",
        norm_layer=LayerNorm2d,
        act=nn.SiLU,
        groups=1,
        bias=False,
        conv1x1=False,
        reduce_dim=False,
        skip_connection=True,
    ):
        super(SimpleConvBlocks, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * depth

        if reduce_dim:
            ch1 = min(in_chs, out_chs)
        else:
            ch1 = max(in_chs, out_chs)

        # ch1 = min(ch1, max_dim)

        layers = []
        self.reduce_dim = False
        if in_chs > max_dim:
            self.reduce_block = nn.Conv2d(in_chs, max_dim, 1, bias=False)
            in_chs = max_dim
            ch1 = max_dim
            self.reduce_dim = True

        # norm_shape = None
        # if norm_layer == nn.BatchNorm2d:
        #     norm_shape = ch1
        # if norm_layer == nn.LayerNorm:
        #     norm_shape = [ch1, 16, 16]  # not elegant

        for i in range(depth - 1):
            block = nn.Sequential(
                nn.Conv2d(
                    in_chs if i == 0 else ch1,
                    ch1,
                    kernel_size[i],
                    stride,
                    padding,
                    groups,
                    bias=bias,
                ),
                norm_layer(ch1),
                act(inplace=True),
            )
            layers.append(block)
        if not conv1x1:
            block = nn.Sequential(
                nn.Conv2d(
                    ch1, out_chs, kernel_size[-1], stride, padding, groups, bias=bias
                ),
                act(inplace=True),
            )
            layers.append(block)
        if conv1x1:
            block = nn.Sequential(
                nn.Conv2d(
                    ch1, ch1, kernel_size[-1], stride, padding, groups, bias=bias
                ),
                norm_layer(ch1),
                act(inplace=True),
                nn.Conv2d(ch1, out_chs, 1, bias=bias),
                act(inplace=True),
            )
            layers.append(block)

        self.block = nn.Sequential(*layers)

        self.skip_connection = skip_connection
        self.depth = depth

    def forward(self, x):
        if self.reduce_dim:
            x = self.reduce_block(x)

        for i, b in enumerate(self.block):
            x_prev = x
            x_next = b(x)
            if i < self.depth - 1:
                x = x_next + x_prev if self.skip_connection else x_next
            else:
                x = x_next
        return x


class Simple1DConvBlocks(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        depth=3,
        kernel_size=5,
        max_dim=1024,
        stride=1,
        padding="same",
        norm_layer=nn.LayerNorm,
        act=nn.SiLU,
        groups=1,
        bias=False,
        conv1x1=False,
        reduce_dim=False,
        skip_connection=True,
    ):
        super(Simple1DConvBlocks, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * depth

        if reduce_dim:
            ch1 = min(in_chs, out_chs)
        else:
            ch1 = max(in_chs, out_chs)

        # ch1 = min(ch1, max_dim)

        layers = []
        self.reduce_dim = False
        if in_chs > max_dim:
            self.reduce_block = nn.Conv1d(in_chs, max_dim, 1, bias=False)
            in_chs = max_dim
            ch1 = max_dim
            self.reduce_dim = True

        # norm_shape = None
        # if norm_layer == nn.BatchNorm2d:
        #     norm_shape = ch1
        # if norm_layer == nn.LayerNorm:
        #     norm_shape = [ch1, 16, 16]  # not elegant

        for i in range(depth - 1):
            block = nn.Sequential(
                nn.Linear(in_features=in_chs if i == 0 else ch1, out_features=ch1),
                norm_layer(ch1),
                act(inplace=True),
            )
            layers.append(block)
        if not conv1x1:
            block = nn.Sequential(
                nn.Linear(ch1, out_chs),
                act(inplace=True),
            )
            layers.append(block)

        self.block = nn.Sequential(*layers)

        self.skip_connection = skip_connection
        self.depth = depth

    def forward(self, x):
        if self.reduce_dim:
            x = self.reduce_block(x)

        for i, b in enumerate(self.block):
            x_prev = x
            x_next = b(x)
            if i < self.depth - 1:
                x = x_next + x_prev if self.skip_connection else x_next
            else:
                x = x_next
        return x


class ConvBlocks(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        max_dim=1024,
        depth=3,
        kernel_size=5,
    ):
        super().__init__()

        dim = min(in_chs, max_dim)

        self.blocks = []
        for i in range(depth):
            _in_chs = in_chs if i == 0 else dim
            norm_layer = None  # defaults to LayerNorm
            # if i == depth - 1 and skip_last_norm:
            # norm_layer = nn.Identity
            self.blocks.append(
                ConvNeXtBlock(_in_chs, dim, kernel_size,
                              norm_layer=norm_layer),
            )
        self.blocks.append(nn.Conv2d(dim, out_chs, 3, padding="same"))
        self.blocks.append(nn.GELU())
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)


class DictConvBlocks(nn.Module):
    def __init__(
        self,
        layers=[1, 2, 4, 8, 14, 20, 26, 32],
        in_dims=[1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280],
        out_dim=256,
        max_dim=1280,
        kernel_sizes=[1, 1, 1, 1, 1, 1, 1, 1],
        depths=[2, 2, 2, 2, 2, 2, 2, 2],
        block=SimpleConvBlocks,
    ):
        super().__init__()

        self.blocks_dict = nn.ModuleDict()
        for i, layer in enumerate(layers):
            self.blocks_dict[str(layer)] = block(
                in_dims[i],
                out_dim,
                max_dim=max_dim,
                depth=depths[i],
                kernel_size=kernel_sizes[i],
            )

    def forward(self, x):
        for layer, block in self.blocks_dict.items():
            x[layer] = block(x[layer])
        return x


class ClassTokenMLPs(nn.Module):
    def __init__(
        self,
        layers = [1, 2, 4, 8, 14, 20, 26, 32],
        in_dims = [1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280],
        out_dim = 256
    ):
        super().__init__()

        self.mlp_dict = nn.ModuleDict()
        for i, layer in enumerate(layers):
            self.mlp_dict[str(layer)] = Mlp(in_features=in_dims[i], out_features=out_dim)

    def forward(self, x):
        for layer, mlp in self.mlp_dict.items():
            x[layer] = mlp(x[layer])
        return x


class VoxelNonShareLinearWeight(nn.Module):
    def __init__(self, d_model, n_voxels, **kwargs):
        super().__init__()
        dummy = nn.Linear(d_model, n_voxels)
        self.weight = nn.Parameter(dummy.weight)  # (n_voxels, d_model)
        self.bias = nn.Parameter(dummy.bias)  # (n_voxels,)

    def forward(self, coords, voxel_indices=..., *args, **kwargs):
        w = self.weight[voxel_indices]  # (n_voxels, d_model)
        b = self.bias[voxel_indices]  # (n_voxels,)
        return w, b


def _stack(d):
    return torch.stack(list(d.values()), dim=-1)


class EncodingModel(nn.Module):
    def __init__(self, num_voxels, coords, behavior_in, behavior_hidden, final_visual_emb_dim, final_bert_emb_dim, CLIP_model_root,
                 encode_type='visual', vit_model_root=None, bert_model_root=None):
        super(EncodingModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mu_sigma = 0.01
        self.num_voxels = num_voxels
        self.coords = coords

        self.vit_model_root = vit_model_root
        self.bert_model_root = bert_model_root
        # visual
        if encode_type == 'visual' or encode_type == 'all':
            self.retinomapper = CachedCoordsMLP(3, 2, act_fn=nn.Tanh())
            self.visuallayerselector = CachedCoordsMLP(3, 8, act_fn=nn.Softmax(dim=-1))
            self.vis_conv_blocks = DictConvBlocks(out_dim=final_visual_emb_dim)
            self.vis_cls_blocks = ClassTokenMLPs(out_dim=final_visual_emb_dim)
            self.visual_linear = VoxelNonShareLinearWeight(final_visual_emb_dim, num_voxels)
        # caption
        if encode_type == 'caption' or encode_type == 'all':
            self.wordmapper = CachedCoordsMLP(3, 51, act_fn=nn.Softmax(dim=-1))
            self.bertlayerselector = CachedCoordsMLP(3, 8, act_fn=nn.Softmax(dim=-1))
            self.cap_conv_blocks = DictConvBlocks(
                layers=[1, 2, 4, 8, 12, 16, 20, 24],
                in_dims=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
                out_dim=final_bert_emb_dim,
                max_dim=1024,
                block=Simple1DConvBlocks
            )
            self.cap_cls_blocks = ClassTokenMLPs(out_dim=final_bert_emb_dim, layers=[1, 2, 4, 8, 12, 16, 20, 24], in_dims=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024])
            self.bert_linear = VoxelNonShareLinearWeight(final_bert_emb_dim, num_voxels)
        # Behavior embedding
        self.behavior_embedding = BehaviorEmbed(in_dim=behavior_in, dim=behavior_hidden)
        # Pre-trained model
        if vit_model_root is None or bert_model_root is None:
            self.CLIP_model = AdaLNLoRACLIP(CLIP_model_root, d_c=behavior_hidden, encode_type=encode_type)
        if vit_model_root is not None:
            self.vit_model = AdaLNLoRAViT(vit_model_root, d_c=behavior_hidden)
        if bert_model_root is not None:
            self.bert_model = AdaLNLoraBert(bert_model_root, d_c=behavior_hidden)
        # Init norm
        if vit_model_root is None:
            self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            self.norm = self.norm.to(self.device)
        else:
            self.norm = ViTFeatureExtractor.from_pretrained(vit_model_root)

    def VisualEncode(self, sample, index):
        caption = sample[0]
        image = sample[1]
        caption_beta = sample[2]
        image_beta = sample[3]
        caption_condition = sample[4]
        image_condition = sample[5]

        # 图片中间向量提取
        if index == 2:
            condition = caption_condition
        else:
            condition = image_condition
        c = self.behavior_embedding(condition) # [B, d_c]
        if self.vit_model_root is None:
            image = self.norm(image)
            x_retina_grid, x_cls_dict, _ = self.CLIP_model.get_visual_intermediate_layers(image, c=c)
        else:
            image = self.norm(images=image, return_tensors='pt')['pixel_values'].to(self.device)
            x_retina_grid, x_cls_dict, _ = self.vit_model.get_visual_intermediate_layers(image, c=c)
        x_retina_grid = self.vis_conv_blocks(x_retina_grid)
        x_cls_dict = self.vis_cls_blocks(x_cls_dict)
        x_cls = _stack(x_cls_dict) # [B, D, 8]

        # divide voxels into chunks
        voxel_indices = torch.arange(self.num_voxels, device=self.coords.device)
        chunks = torch.split(voxel_indices, 10000)

        out_ys, reg_layers = [], []
        for voxel_indices_chunk in chunks:
            out_y, reg_layer = self._forward_visual_encoding(
                x_retina_grid,
                x_cls,
                voxel_indices_chunk
            )
            out_ys.append(out_y)
            reg_layers.append(reg_layer)

        out_y = torch.cat(out_ys, dim=1)
        reg_layer = torch.cat(reg_layers, dim=0).mean()

        if self.training:
            return out_y, reg_layer
        else:
            return out_y

    def _forward_visual_encoding(self, x_retina_grid, x_cls, voxel_indices):
        # RetinoMapper & LayerSelector
        mu = self.retinomapper(self.coords, voxel_indices)

        w_layer = self.visuallayerselector(self.coords, voxel_indices)

        # regularization
        def entropy(x):
            return (x * x.log()).sum(dim=1)

        if self.training and next(self.visuallayerselector.parameters()).requires_grad:
            reg_layer = entropy(w_layer)  # [N]
        else:
            reg_layer = torch.zeros_like(w_layer[:, 0])  # [N]

        x_cls = repeat(x_cls, "b d l -> b n d l", n=1)
        _w_layer = repeat(w_layer, 'n l -> b n d l', b=1, d=1)
        x_cls = (x_cls * _w_layer).sum(dim=-1)  # [B, N, D]

        mu = mu * (1 - self.mu_sigma)
        if self.training:
            norm = torch.normal(0, torch.ones_like(mu) * self.mu_sigma)
            mu = mu + norm
        bsz = x_cls.shape[0]
        mu = repeat(mu, "n d -> b n d", b=bsz)
        mu = rearrange(mu, "b n (d c) -> b n d c", d=1, c=2)

        _w_layer = repeat(w_layer, "n l -> b n l", b=1)
        x_retina = None
        for i, layer in enumerate([1, 2, 4, 8, 14, 20, 26, 32]):
            x = x_retina_grid[str(layer)]
            _x_retina = F.grid_sample(
                x,
                mu,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B, D, N, 1] (C=D_model, D=1, N=N_voxels)
            _x_retina = rearrange(_x_retina, "b c n d -> b n (c d)")
            _x_retina = _x_retina * _w_layer[:, :, i : i + 1]
            if x_retina is None:
                x_retina = _x_retina
            else:
                x_retina += _x_retina
        x_y = x_retina + x_cls

        w, b = self.visual_linear(self.coords, voxel_indices)
        out_y = (x_y * w.unsqueeze(0)).mean(-1) + b.unsqueeze(0)

        return out_y, reg_layer

    def BertEncode(self, sample, index):
        caption = sample[0]
        image = sample[1]
        caption_beta = sample[2]
        image_beta = sample[3]
        caption_condition = sample[4]
        image_condition = sample[5]

        # 文本预处理
        inputs = clip.tokenize(caption).to(self.coords.device)

        # 文本中间向量提取
        if index == 2:
            condition = caption_condition
        else:
            condition = image_condition
        c = self.behavior_embedding(condition) # [B, d_c]
        if self.bert_model_root is None:
            x_word_emb, x_cls_dict, _ = self.CLIP_model.get_bert_intermediate_layers(inputs, c=c)
        else:
            x_word_emb, x_cls_dict, _ = self.bert_model.get_bert_intermediate_layers(caption, c=c)
        x_word_emb = self.cap_conv_blocks(x_word_emb)
        x_cls_dict = self.cap_cls_blocks(x_cls_dict)
        x_cls = _stack(x_cls_dict) # [B, D, 8]

        # divide voxels into chunks
        voxel_indices = torch.arange(self.num_voxels, device=self.coords.device)
        chunks = torch.split(voxel_indices, 10000)

        out_ys, reg_layers = [], []
        for voxel_indices_chunk in chunks:
            out_y, reg_layer = self._forward_bert_encoding(
                x_word_emb,
                x_cls,
                voxel_indices_chunk
            )
            out_ys.append(out_y)
            reg_layers.append(reg_layer)

        out_y = torch.cat(out_ys, dim=1)
        reg_layer = torch.cat(reg_layers, dim=0).mean()

        if self.training:
            return out_y, reg_layer
        else:
            return out_y

    def _forward_bert_encoding(self, x_word_emb, x_cls, voxel_indices):
        # WordMapper & BertLayerSelector
        word_weight = self.wordmapper(self.coords, voxel_indices)  # [N, L]

        w_layer = self.bertlayerselector(self.coords, voxel_indices)

        # regularization
        def entropy(x):
            return (x * x.log()).sum(dim=1)

        if self.training and next(self.bertlayerselector.parameters()).requires_grad:
            # reg_layer = entropy(w_layer) + entropy(word_weight)
            reg_layer = entropy(w_layer)
        else:
            reg_layer = torch.zeros_like(w_layer[:, 0])

        x_cls = repeat(x_cls, 'b d l -> b n d l', n=1)
        _w_layer = repeat(w_layer, 'n l -> b n d l', b=1, d=1)
        x_cls = (x_cls * _w_layer).sum(dim=-1)

        _w_layer = repeat(w_layer, 'n l -> b n l', b=1)
        word_weight = repeat(word_weight, 'n l -> b n l d', b=1, d=1)
        x_word = None
        for i, layer in enumerate([1, 2, 4, 8, 12, 16, 20, 24]):
            x = x_word_emb[str(layer)] # [B, L, D]
            x = repeat(x, 'b l d -> b n l d', n=1)
            temp = (x * word_weight).sum(dim=-2) # [B N D]
            temp = temp * _w_layer[:, :, i : i + 1]
            if x_word is None:
                x_word = temp
            else:
                x_word += temp
        x_y = x_word + x_cls

        w, b = self.bert_linear(self.coords, voxel_indices)
        out_y = (x_y * w.unsqueeze(0)).mean(-1) + b.unsqueeze(0)

        return out_y, reg_layer

    def predictive_coding(self, sample):
        if self.training:
            image_signal, image_reg = self.VisualEncode(sample)
            caption_signal, caption_reg = self.BertEncode(sample)
            out_y = image_signal + caption_signal
            reg_layer = image_reg + caption_reg
            return out_y, reg_layer
        else:
            image_signal = self.VisualEncode(sample)
            caption_signal = self.BertEncode(sample)
            out_y = image_signal + caption_signal
            return out_y


class LayerPreferrenceEncodingModel(nn.Module):
    def __init__(self, behavior_in, behavior_hidden, CLIP_model_root,
                 encode_type='visual', vit_model_root=None, bert_model_root=None):
        super(LayerPreferrenceEncodingModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vit_model_root = vit_model_root
        self.bert_model_root = bert_model_root

        # Behavior embedding
        self.behavior_embedding = BehaviorEmbed(in_dim=behavior_in, dim=behavior_hidden)
        # Pre-trained model
        if vit_model_root is None or bert_model_root is None:
            self.CLIP_model = AdaLNLoRACLIP(CLIP_model_root, d_c=behavior_hidden, encode_type=encode_type)
        if vit_model_root is not None:
            self.vit_model = AdaLNLoRAViT(vit_model_root, d_c=behavior_hidden)
        if bert_model_root is not None:
            self.bert_model = AdaLNLoraBert(bert_model_root, d_c=behavior_hidden)
        # Init norm
        if vit_model_root is None:
            self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            self.norm = self.norm.to(self.device)
        else:
            self.norm = ViTFeatureExtractor.from_pretrained(vit_model_root)

    def VisualEncode(self, sample, index):
        caption = sample[0]
        image = sample[1]
        caption_beta = sample[2]
        image_beta = sample[3]
        caption_condition = sample[4]
        image_condition = sample[5]

        # 图片中间向量提取
        if index == 2:
            condition = caption_condition
        else:
            condition = image_condition
        c = self.behavior_embedding(condition)  # [B, d_c]
        if self.vit_model_root is None:
            image = self.norm(image)
            x_retina_grid, x_cls_dict, _ = self.CLIP_model.get_visual_intermediate_layers(image, c=c)
        else:
            image = self.norm(images=image, return_tensors='pt')['pixel_values'].to(self.device)
            x_retina_grid, x_cls_dict, _ = self.vit_model.get_visual_intermediate_layers(image, c=c)

        return x_cls_dict

    def BertEncode(self, sample, index):
        caption = sample[0]
        image = sample[1]
        caption_beta = sample[2]
        image_beta = sample[3]
        caption_condition = sample[4]
        image_condition = sample[5]

        # 文本预处理
        inputs = clip.tokenize(caption).to(self.device)

        # 文本中间向量提取
        if index == 2:
            condition = caption_condition
        else:
            condition = image_condition
        c = self.behavior_embedding(condition) # [B, d_c]
        if self.bert_model_root is None:
            x_word_emb, x_cls_dict, _ = self.CLIP_model.get_bert_intermediate_layers(inputs, c=c)
        else:
            x_word_emb, x_cls_dict, _ = self.bert_model.get_bert_intermediate_layers(caption, c=c)

        return x_cls_dict


class PredictiveEncodingModel(nn.Module):
    def __init__(self, num_voxels, coords, behavior_in, behavior_hidden, final_visual_emb_dim, final_bert_emb_dim, CLIP_model_root,
                 encode_type='visual', vit_model_root=None, bert_model_root=None):
        super(PredictiveEncodingModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mu_sigma = 0.01
        self.num_voxels = num_voxels
        self.coords = coords

        self.vit_model_root = vit_model_root
        self.bert_model_root = bert_model_root
        # visual
        if encode_type == 'visual' or encode_type == 'all':
            self.retinomapper = CachedCoordsMLP(3, 2, act_fn=nn.Tanh())
            self.visuallayerselector = CachedCoordsMLP(3, 8, act_fn=nn.Softmax(dim=-1))
            self.vis_conv_blocks = DictConvBlocks(
                layers=[1, 2, 4, 8, 14, 20, 26, 32],
                in_dims=[2560, 2560, 2560, 2560, 2560, 2560, 2560, 2560],
                out_dim=final_visual_emb_dim,
                max_dim=2560
            )
            self.vis_cls_blocks = ClassTokenMLPs(
                layers=[1, 2, 4, 8, 14, 20, 26, 32],
                in_dims=[2560, 2560, 2560, 2560, 2560, 2560, 2560, 2560],
                out_dim=final_visual_emb_dim
            )
            self.visual_linear = VoxelNonShareLinearWeight(final_visual_emb_dim, num_voxels)
        # caption
        if encode_type == 'caption' or encode_type == 'all':
            self.wordmapper = CachedCoordsMLP(3, 51, act_fn=nn.Softmax(dim=-1))
            self.bertlayerselector = CachedCoordsMLP(3, 8, act_fn=nn.Softmax(dim=-1))
            self.cap_conv_blocks = DictConvBlocks(
                layers=[1, 2, 4, 8, 12, 16, 20, 24],
                in_dims=[2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
                out_dim=final_bert_emb_dim,
                max_dim=2048,
                block=Simple1DConvBlocks
            )
        # Behavior embedding
        self.behavior_embedding = BehaviorEmbed(in_dim=behavior_in, dim=behavior_hidden)
        # Pre-trained model
        if vit_model_root is None or bert_model_root is None:
            self.CLIP_model = AdaLNLoRACLIP(CLIP_model_root, d_c=behavior_hidden, encode_type=encode_type)
        if vit_model_root is not None:
            self.vit_model = AdaLNLoRAViT(vit_model_root, d_c=behavior_hidden)
        if bert_model_root is not None:
            self.bert_model = AdaLNLoraBert(bert_model_root, d_c=behavior_hidden)
        # Init norm
        if vit_model_root is None:
            self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            self.norm = self.norm.to(self.device)
        else:
            self.norm = ViTFeatureExtractor.from_pretrained(vit_model_root)

    def VisualEncode(self, sample, index):
        caption_image = sample[0]
        image = sample[1]
        caption_beta = sample[2]
        image_beta = sample[3]
        caption_condition = sample[4]
        image_condition = sample[5]

        # Extract caption generated image feature
        caption_c = self.behavior_embedding(image_condition)

        if self.vit_model_root is None:
            caption_image = self.norm(caption_image)
            x_retina_grid_caption, x_cls_dict_caption, _ = self.CLIP_model.get_visual_intermediate_layers(caption_image, c=caption_c)
        else:
            caption_image = self.norm(images=caption_image, return_tensors='pt')['pixel_values'].to(self.device)
            x_retina_grid_caption, x_cls_dict_caption, _ = self.vit_model.get_visual_intermediate_layers(caption_image, c=caption_c)

        # Extract image feature
        image_c = self.behavior_embedding(image_condition)
        if self.vit_model_root is None:
            image = self.norm(image)
            x_retina_grid_image, x_cls_dict_image, _ = self.CLIP_model.get_visual_intermediate_layers(image, c=image_c)
        else:
            image = self.norm(images=image, return_tensors='pt')['pixel_values'].to(self.device)
            x_retina_grid_image, x_cls_dict_image, _ = self.vit_model.get_visual_intermediate_layers(image, c=image_c)

        # Integrate Caption generated image & Image
        x_retina_grid = {}
        for k in x_retina_grid_image.keys():
            x_retina_grid[k] = torch.concat((x_retina_grid_caption[k], x_retina_grid_image[k]), dim=1)

        x_cls_dict = {}
        for k in x_cls_dict_image.keys():
            x_cls_dict[k] = torch.concat((x_cls_dict_caption[k], x_cls_dict_image[k]), dim=-1)
            print(x_cls_dict_caption[k].shape)
            print(x_cls_dict[k].shape)

        x_retina_grid = self.vis_conv_blocks(x_retina_grid)
        x_cls_dict = self.vis_cls_blocks(x_cls_dict)
        x_cls = _stack(x_cls_dict) # [B, D, 8]

        # divide voxels into chunks
        voxel_indices = torch.arange(self.num_voxels, device=self.coords.device)
        chunks = torch.split(voxel_indices, 10000)

        out_ys, reg_layers = [], []
        for voxel_indices_chunk in chunks:
            out_y, reg_layer = self._forward_visual_encoding(
                x_retina_grid,
                x_cls,
                voxel_indices_chunk
            )
            out_ys.append(out_y)
            reg_layers.append(reg_layer)

        out_y = torch.cat(out_ys, dim=1)
        reg_layer = torch.cat(reg_layers, dim=0).mean()

        if self.training:
            return out_y, reg_layer
        else:
            return out_y

    def _forward_visual_encoding(self, x_retina_grid, x_cls, voxel_indices):
        # RetinoMapper & LayerSelector
        mu = self.retinomapper(self.coords, voxel_indices)

        w_layer = self.visuallayerselector(self.coords, voxel_indices)

        # regularization
        def entropy(x):
            return (x * x.log()).sum(dim=1)

        if self.training and next(self.visuallayerselector.parameters()).requires_grad:
            reg_layer = entropy(w_layer)  # [N]
        else:
            reg_layer = torch.zeros_like(w_layer[:, 0])  # [N]

        x_cls = repeat(x_cls, "b d l -> b n d l", n=1)
        _w_layer = repeat(w_layer, 'n l -> b n d l', b=1, d=1)
        x_cls = (x_cls * _w_layer).sum(dim=-1)  # [B, N, D]

        mu = mu * (1 - self.mu_sigma)
        if self.training:
            norm = torch.normal(0, torch.ones_like(mu) * self.mu_sigma)
            mu = mu + norm
        bsz = x_cls.shape[0]
        mu = repeat(mu, "n d -> b n d", b=bsz)
        mu = rearrange(mu, "b n (d c) -> b n d c", d=1, c=2)

        _w_layer = repeat(w_layer, "n l -> b n l", b=1)
        x_retina = None
        for i, layer in enumerate([1, 2, 4, 8, 14, 20, 26, 32]):
            x = x_retina_grid[str(layer)]
            _x_retina = F.grid_sample(
                x,
                mu,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B, D, N, 1] (C=D_model, D=1, N=N_voxels)
            _x_retina = rearrange(_x_retina, "b c n d -> b n (c d)")
            _x_retina = _x_retina * _w_layer[:, :, i : i + 1]
            if x_retina is None:
                x_retina = _x_retina
            else:
                x_retina += _x_retina
        x_y = x_retina + x_cls

        w, b = self.visual_linear(self.coords, voxel_indices)
        out_y = (x_y * w.unsqueeze(0)).mean(-1) + b.unsqueeze(0)

        return out_y, reg_layer