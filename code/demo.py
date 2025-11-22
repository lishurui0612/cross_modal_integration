import os
import torch
import numpy as np

from model import EncodingModel
from model_cae import BrainCLIP

if __name__ == '__main__':

    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is ', device)

    '''
    demo use for voxel-wise encoding model
    '''
    print('----------Demo code for voxel-wise encoding model----------')
    CLIP_model_root = '../model_cache'  # change it for your model root

    num_voxels = 10000
    coords = np.random.randn(num_voxels, 3)  # in real case, coords is the
    coords = torch.from_numpy(coords).to(device, dtype=torch.float)

    model = EncodingModel(
        num_voxels=num_voxels,
        coords=coords,
        behavior_in=8,  # input dimension of participants behavior embedding
        behavior_hidden=16,  # output dimension of participants behavior embedding
        final_visual_emb_dim=256,  # not used in caption-evoked encoding model, used in image-evoked encoding model which is not reported in paper
        final_bert_emb_dim=256,  # final dimension used in voxel-wise linear regression
        CLIP_model_root=CLIP_model_root,
        encode_type='caption',  # 'caption' or 'visual'
        vit_model_root=None,  # if not None, use pure vit as backbone rather than vision encoder in CLIP model
        bert_model_root=None  # if not None, use Bert as backbone rather than text encoder in CLIP model
    ).to(device, dtype=torch.float)

    # input & output
    caption_evoked_response = torch.rand((1, num_voxels)).to(device, dtype=torch.float)
    behavior = torch.randn((1, 8)).to(device, dtype=torch.float)

    sample = [[] for i in range(6)]
    sample[0] = ['Example caption used in demo code!']  # in real case, the captions are all Chinese.
    sample[1] = None
    sample[2] = caption_evoked_response
    sample[3] = None
    sample[4] = behavior
    sample[5] = None

    prediction, layer_reg = model.BertEncode(sample, 2)  # 2 means using caption-evoked encoding model
    print('Shape of encoding model prediction is', prediction.shape)
    print('Regulation of 8 layers is', layer_reg.item())

    '''
    demo use of BrainCLIP model
    '''
    print('----------Demo code for BrainCLIP model----------')
    hemi_voxels = [5000, 5000]  # voxels of each hemisphere
    num_voxels = hemi_voxels[0] + hemi_voxels[1]
    sph_coords = np.random.randn(num_voxels, 3)  # sphere coords of each voxel

    model = BrainCLIP(
        vertices=hemi_voxels,
        sph_coords=sph_coords,  # if Index_root is not None, model will use sphere coords to build index
        img_size=224,  # response can be reordered as an image based on voxels' sphere coords
        depth=3,  # if type is 'vit', depth means the number of blocks
        type='linear',  # In reported results, we use linear transformation as encoder
        embed_dim=768,
        post_type='linear',
        Index_root=None
    ).to(device, dtype=torch.float)

    caption_evoked_response = torch.rand((1, num_voxels)).to(device, dtype=torch.float)
    image_evoked_response = torch.rand((1, num_voxels)).to(device, dtype=torch.float)

    sample = [[] for i in range(6)]
    sample[0] = ['Example caption used in demo code!']  # in real case, the captions are all Chinese.
    sample[1] = None
    sample[2] = caption_evoked_response
    sample[3] = image_evoked_response
    sample[4] = None
    sample[5] = None

    caption_feature, _ = model(sample[2])
    image_feature, _ = model(sample[3])

    caption_feature_norm = caption_feature / caption_feature.norm(dim=-1, keepdim=True)
    image_feature_norm = image_feature / image_feature.norm(dim=-1, keepdim=True)
    similarity = image_feature_norm @ caption_feature_norm.T

    print('In BrainCLIP demo, the similarity between caption response and image response is', similarity.item())