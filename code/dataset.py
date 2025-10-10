# coding=gbk
import os
import torch
import random
import nilearn
import numpy as np
from scipy import io
from PIL import Image
from tqdm import tqdm
import torch.utils.data as data
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from cn_clip.clip import load_from_name, image_transform
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode


def _convert_to_rgb(image):
    return image.convert('RGB')


def fMRI2grid(data, azimuth, elevation, img_size):
    X, Y = np.meshgrid(np.linspace(-1, 1, img_size), np.linspace(-1, 1, img_size))

    elevation = np.sin(elevation)
    elevation = 2 * elevation / (np.max(elevation) - np.min(elevation))
    azimuth = 2 * azimuth / (np.max(azimuth) - np.min(azimuth))

    transformed_gridmap = griddata((azimuth, elevation), data, (X, Y), method='nearest')

    return transformed_gridmap


def prepare_pair_data(subject, caption_image_pairs_root, stimulus_index_root, processed_root, data_type='beta_zscore', session_type='Stimulus', trial_type='match'):
    # 读取图片文字对
    Stimulus_pairs = dict()
    with open(caption_image_pairs_root, 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            Stimulus_pairs[temp[0]] = temp[1]
            Stimulus_pairs[temp[1]] = temp[0]
    f.close()
    # 读取刺激编号
    Stimulus_index = dict()
    with open(stimulus_index_root, 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            Stimulus_index[int(temp[0])] = temp[1]
            Stimulus_index[temp[1]] = int(temp[0])
    f.close()

    subject_beta_root = os.path.join(processed_root, subject, session_type, data_type)
    # 读取图片训练集和验证集
    train_image = []
    test_image = []
    with open(os.path.join(processed_root, subject, 'train_img.txt'), 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            train_image.append(temp[0])
    f.close()

    with open(os.path.join(processed_root, subject, 'test_img.txt'), 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            test_image.append(temp[0])
    f.close()

    # 归类每个刺激的响应
    FileList = sorted(os.listdir(subject_beta_root))
    betaList = []
    for file in FileList:
        betaList.append(os.path.join(subject_beta_root, file))

    stim_dict = dict()
    for beta in betaList:
        if beta[-5] != 'a':
            stim = int(beta[-24:-19])
        else:
            stim = int(beta[-22:-17])
        if stim not in stim_dict:
            stim_dict[stim] = []
        stim_dict[stim].append(beta)

    if trial_type == 'match':
        train_list = []
        val_list = []
        for beta in betaList:
            if beta[-5] != 'a':
                stim = int(beta[-24:-19])
                pair_stim = Stimulus_index[Stimulus_pairs[Stimulus_index[stim]]]
            else:
                stim = int(beta[-22:-17])
                pair_stim = Stimulus_index[Stimulus_pairs[Stimulus_index[stim]]]
            if Stimulus_index[stim] in train_image and pair_stim in stim_dict:
                train_list.append(beta)
            elif Stimulus_index[stim] in test_image and pair_stim in stim_dict:
                val_list.append(beta)
            # elif Stimulus_index[pair_stim] in train_image:
            #     train_list.append(beta)
            # elif Stimulus_index[pair_stim] in test_image:
            #     val_list.append(beta)
        print(f'For {subject}, Caption-Image pair train beta: {len(train_list)}, Caption-Image pair validation beta: {len(val_list)}')

        return Stimulus_index, Stimulus_pairs, stim_dict, train_list, val_list
    elif trial_type == 'unmatch':
        train_list = []
        val_list = []
        for beta in betaList:
            if beta[-5] != 'a':
                stim = int(beta[-24:-19])
            else:
                stim = int(beta[-22:-17])
            if Stimulus_index[stim] in train_image:
                train_list.append(beta)
            elif Stimulus_index[stim] in test_image:
                val_list.append(beta)
        print(f'For {subject}, Caption-Image pair train beta: {len(train_list)}, Caption-Image pair validation beta: {len(val_list)}')

        return Stimulus_index, Stimulus_pairs, stim_dict, train_list, val_list


class SingleTrial_dataset(data.Dataset):
    def __init__(self, beta_list, Stimulus_index, Stimulus_pairs, stim_dict, image_root, norm=True, ne=False):
        self.beta_list = beta_list
        self.Stimulus_index = Stimulus_index
        self.Stimulus_pairs = Stimulus_pairs
        self.stim_dict = stim_dict
        self.image_root = image_root
        self.ne = ne
        if norm:
            self.transform = image_transform()
        else:
            self.transform = Compose([
                                Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                                _convert_to_rgb,
                                ToTensor(),
                            ])

    def __len__(self):
        return len(self.beta_list)

    def __getitem__(self, item):
        beta = self.beta_list[item]
        if beta[-5] != 'a':
            stim = int(beta[-24:-19])
        else:
            stim = int(beta[-22:-17])

        if self.Stimulus_index[stim][-3:] == 'jpg':
            image_stim = stim

            img_beta = beta

            image_dir = os.path.join(self.image_root, self.Stimulus_index[image_stim])
            stimulus = self.transform(Image.open(image_dir))

            image_mat = io.loadmat(img_beta)
            beta = np.squeeze(image_mat['beta'])
            # condition = np.array([
            #     image_mat['run'][0][0],
            #     image_mat['timestamp'][0][0], image_mat['behavior'][0][0], image_mat['behavior_rt'][0][0],
            #     image_mat['caption'][0][0], image_mat['image'][0][0], image_mat['unmatch'][0][0],
            #     image_mat['repeat'][0][0]
            # ])
        else:
            caption_stim = stim

            cap_beta = beta
            stimulus = self.Stimulus_index[caption_stim]

            caption_mat = io.loadmat(cap_beta)
            beta = np.squeeze(caption_mat['beta'])
            # condition = np.array([
            #     caption_mat['run'][0][0],
            #     caption_mat['timestamp'][0][0], caption_mat['behavior'][0][0], caption_mat['behavior_rt'][0][0],
            #     caption_mat['caption'][0][0], caption_mat['image'][0][0], caption_mat['unmatch'][0][0],
            #     caption_mat['repeat'][0][0]
            # ])

        stim = np.array([stim]).astype(np.int32)
        beta = torch.from_numpy(beta).to(torch.float)
        # condition = torch.from_numpy(condition).to(torch.float)

        # return stim, stimulus, beta, condition
        return stim, stimulus, beta


class Unmatch_dataset(data.Dataset):
    def __init__(self, beta_list, Stimulus_index, Stimulus_pairs, stim_dict, image_root, encode_type='caption', type='caption', norm=True,
                 image_size=224):
        self.beta_list = beta_list
        self.Stimulus_index = Stimulus_index
        self.Stimulus_pairs = Stimulus_pairs
        self.stim_dict = stim_dict
        self.image_root = image_root
        self.encode_type = encode_type
        self.type = type
        if norm:
            self.transform = image_transform()
        else:
            self.transform = Compose([
                                Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                                _convert_to_rgb,
                                ToTensor(),
                            ])

    def __len__(self):
        return len(self.beta_list)

    def __getitem__(self, item):
        beta = self.beta_list[item]
        if beta[-5] != 'a':
            stim = int(beta[-24:-19])
        else:
            stim = int(beta[-22:-17])

        if self.type == 'caption':
            caption_stim = stim
            cap_beta = beta
            caption_mat = io.loadmat(cap_beta)
            beta = np.squeeze(caption_mat['beta'])
            condition = np.array([
                caption_mat['run'][0][0],
                caption_mat['timestamp'][0][0], caption_mat['behavior'][0][0], caption_mat['behavior_rt'][0][0],
                caption_mat['caption'][0][0], caption_mat['image'][0][0], caption_mat['unmatch'][0][0],
                caption_mat['repeat'][0][0]
            ])

            pair_stim = caption_mat['pair_stim'][0][0]

            if self.encode_type == 'caption':
                stimulus = self.Stimulus_index[caption_stim]
                pair_stimulus = self.Stimulus_pairs[self.Stimulus_index[pair_stim]]
                pair_stim = self.Stimulus_index[pair_stimulus]
            elif self.encode_type == 'visual':
                image_dir = os.path.join(self.image_root, self.Stimulus_pairs[self.Stimulus_index[caption_stim]])
                stimulus = self.transform(Image.open(image_dir))

                image_dir = os.path.join(self.image_root, self.Stimulus_index[pair_stim])
                pair_stimulus = self.transform(Image.open(image_dir))
            else:
                stimulus = self.Stimulus_index[caption_stim]

                image_dir = os.path.join(self.image_root, self.Stimulus_index[pair_stim])
                pair_stimulus = self.transform(Image.open(image_dir))

            stim = np.array([stim]).astype(np.int32)
            pair_stim = np.array([pair_stim]).astype(np.int32)
            beta = torch.from_numpy(beta).to(torch.float)
            condition = torch.from_numpy(condition).to(torch.float)

            return stim, pair_stim, stimulus, pair_stimulus, beta, condition
        else:
            image_stim = stim
            img_beta = beta
            image_mat = io.loadmat(img_beta)
            beta = np.squeeze(image_mat['beta'])
            condition = np.array([
                image_mat['run'][0][0],
                image_mat['timestamp'][0][0], image_mat['behavior'][0][0], image_mat['behavior_rt'][0][0],
                image_mat['caption'][0][0], image_mat['image'][0][0], image_mat['unmatch'][0][0], image_mat['repeat'][0][0]
            ])

            pair_stim = image_mat['pair_stim'][0][0]

            if self.encode_type == 'caption':
                stimulus = self.Stimulus_pairs[self.Stimulus_index[image_stim]]
                pair_stimulus = self.Stimulus_index[pair_stim]
            elif self.encode_type == 'visual':
                image_dir = os.path.join(self.image_root, self.Stimulus_index[image_stim])
                stimulus = self.transform(Image.open(image_dir))

                image_dir = os.path.join(self.image_root, self.Stimulus_pairs[self.Stimulus_index[pair_stim]])
                pair_stimulus = self.transform(Image.open(image_dir))
            else:
                image_dir = os.path.join(self.image_root, self.Stimulus_index[image_stim])
                stimulus = self.transform(Image.open(image_dir))

                pair_stimulus = self.Stimulus_index[pair_stim]

            init_image = os.path.join(self.image_root, self.Stimulus_index[image_stim])
            pair_init_image = os.path.join(self.image_root, self.Stimulus_pairs[self.Stimulus_index[pair_stim]])

            stim = np.array([stim]).astype(np.int32)
            pair_stim = np.array([pair_stim]).astype(np.int32)
            beta = torch.from_numpy(beta).to(torch.float)
            condition = torch.from_numpy(condition).to(torch.float)

            return stim, pair_stim, stimulus, pair_stimulus, beta, condition, init_image, pair_init_image


class unmatch_session_dataset(data.Dataset):
    def __init__(self, beta_list, Stimulus_index, Stimulus_pairs, stim_dict, image_root, encode_type='caption',
                 type='caption', norm=True,
                 image_size=224):
        self.beta_list = beta_list
        self.Stimulus_index = Stimulus_index
        self.Stimulus_pairs = Stimulus_pairs
        self.stim_dict = stim_dict
        self.image_root = image_root
        self.encode_type = encode_type
        self.type = type
        if norm:
            self.transform = image_transform()
        else:
            self.transform = Compose([
                Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
            ])

    def __len__(self):
        return len(self.beta_list)

    def __getitem__(self, item):
        beta = self.beta_list[item]
        if beta[-5] != 'a':
            stim = int(beta[-24:-19])
        else:
            stim = int(beta[-22:-17])

        if self.type == 'caption':
            caption_stim = stim
            cap_beta = beta
            caption_mat = io.loadmat(cap_beta)
            beta = np.squeeze(caption_mat['beta'])
            condition = np.array([0, 0, 0, 0, 0, 0, 0, 0])

            pair_stim = caption_mat['pair_stim'][0][0]

            if self.encode_type == 'caption':
                stimulus = self.Stimulus_index[caption_stim]
                pair_stimulus = self.Stimulus_pairs[self.Stimulus_index[pair_stim]]
                pair_stim = self.Stimulus_index[pair_stimulus]
            elif self.encode_type == 'visual':
                image_dir = os.path.join(self.image_root, self.Stimulus_pairs[self.Stimulus_index[caption_stim]])
                stimulus = self.transform(Image.open(image_dir))

                image_dir = os.path.join(self.image_root, self.Stimulus_index[pair_stim])
                pair_stimulus = self.transform(Image.open(image_dir))
            else:
                stimulus = self.Stimulus_index[caption_stim]

                image_dir = os.path.join(self.image_root, self.Stimulus_index[pair_stim])
                pair_stimulus = self.transform(Image.open(image_dir))

            stim = np.array([stim]).astype(np.int32)
            pair_stim = np.array([pair_stim]).astype(np.int32)
            beta = torch.from_numpy(beta).to(torch.float)
            condition = torch.from_numpy(condition).to(torch.float)

            return stim, pair_stim, stimulus, pair_stimulus, beta, condition
        else:
            image_stim = stim
            img_beta = beta
            image_mat = io.loadmat(img_beta)
            beta = np.squeeze(image_mat['beta'])
            condition = np.array([0, 0, 0, 0, 0, 0, 0, 0])

            pair_stim = image_mat['pair_stim'][0][0]

            if self.encode_type == 'caption':
                stimulus = self.Stimulus_pairs[self.Stimulus_index[image_stim]]
                pair_stimulus = self.Stimulus_index[pair_stim]
            elif self.encode_type == 'visual':
                image_dir = os.path.join(self.image_root, self.Stimulus_index[image_stim])
                stimulus = self.transform(Image.open(image_dir))

                image_dir = os.path.join(self.image_root, self.Stimulus_pairs[self.Stimulus_index[pair_stim]])
                pair_stimulus = self.transform(Image.open(image_dir))
            else:
                image_dir = os.path.join(self.image_root, self.Stimulus_index[image_stim])
                stimulus = self.transform(Image.open(image_dir))

                pair_stimulus = self.Stimulus_index[pair_stim]

            init_image = os.path.join(self.image_root, self.Stimulus_index[image_stim])
            pair_init_image = os.path.join(self.image_root, self.Stimulus_pairs[self.Stimulus_index[pair_stim]])

            stim = np.array([stim]).astype(np.int32)
            pair_stim = np.array([pair_stim]).astype(np.int32)
            beta = torch.from_numpy(beta).to(torch.float)
            condition = torch.from_numpy(condition).to(torch.float)

            return stim, pair_stim, stimulus, pair_stimulus, beta, condition, init_image, pair_init_image


class Caption_Image_dataset(data.Dataset):
    def __init__(self, beta_list, Stimulus_index, Stimulus_pairs, stim_dict, image_root, norm=True, ne=False, te=False, image_size=224):
        self.beta_list = beta_list
        self.Stimulus_index = Stimulus_index
        self.Stimulus_pairs = Stimulus_pairs
        self.stim_dict = stim_dict
        self.image_root = image_root
        self.ne = ne
        self.te = te
        if norm:
            self.transform = image_transform()
        else:
            self.transform = Compose([
                                Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                                _convert_to_rgb,
                                ToTensor(),
                            ])

    def __len__(self):
        return len(self.beta_list)

    def __getitem__(self, item):
        beta = self.beta_list[item]
        if beta[-5] != 'a':
            stim = int(beta[-24:-19])
        else:
            stim = int(beta[-22:-17])

        if self.Stimulus_index[stim][-3:] == 'jpg':
            image_stim = stim
            caption_stim = self.Stimulus_index[self.Stimulus_pairs[self.Stimulus_index[stim]]]

            img_beta = beta
            cap_beta = random.choice(self.stim_dict[caption_stim])
        else:
            image_stim = self.Stimulus_index[self.Stimulus_pairs[self.Stimulus_index[stim]]]
            caption_stim = stim

            img_beta = random.choice(self.stim_dict[image_stim])
            cap_beta = beta

        caption = self.Stimulus_index[caption_stim]
        image_dir = os.path.join(self.image_root, self.Stimulus_index[image_stim])
        image = self.transform(Image.open(image_dir))

        caption_mat = io.loadmat(cap_beta)
        caption_beta = np.squeeze(caption_mat['beta'])
        caption_condition = np.array([
            caption_mat['run'][0][0],
            caption_mat['timestamp'][0][0], caption_mat['behavior'][0][0], caption_mat['behavior_rt'][0][0],
            caption_mat['caption'][0][0], caption_mat['image'][0][0], caption_mat['unmatch'][0][0],
            caption_mat['repeat'][0][0]
        ])
        if self.ne:
            caption_ne = np.squeeze(caption_mat['neural_encoding'])
        if self.te:
            text_embeddings = np.squeeze(caption_mat['text_embeddings'])

        image_mat = io.loadmat(img_beta)
        image_beta = np.squeeze(image_mat['beta'])
        image_condition = np.array([
            image_mat['run'][0][0],
            image_mat['timestamp'][0][0], image_mat['behavior'][0][0], image_mat['behavior_rt'][0][0],
            image_mat['caption'][0][0], image_mat['image'][0][0], image_mat['unmatch'][0][0], image_mat['repeat'][0][0]
        ])
        if self.ne:
            image_ne = np.squeeze(image_mat['neural_encoding'])

        # if 'prev_caption' in image_mat:
        #     caption = self.Stimulus_index[image_mat['prev_caption'][0][0]]

        if self.ne:
            if self.te:
                return caption, image, \
                       torch.from_numpy(caption_beta).to(torch.float), torch.from_numpy(image_beta).to(torch.float), \
                       torch.from_numpy(caption_condition).to(torch.float), torch.from_numpy(image_condition).to(torch.float), \
                       torch.from_numpy(caption_ne).to(torch.float), torch.from_numpy(image_ne).to(torch.float), \
                       torch.from_numpy(text_embeddings).to(torch.float)
            else:
                return caption, image, \
                       torch.from_numpy(caption_beta).to(torch.float), torch.from_numpy(image_beta).to(torch.float), \
                       torch.from_numpy(caption_condition).to(torch.float), torch.from_numpy(image_condition).to(torch.float), \
                       torch.from_numpy(caption_ne).to(torch.float), torch.from_numpy(image_ne).to(torch.float)
        else:
            if self.te:
                return caption, image, \
                       torch.from_numpy(caption_beta).to(torch.float), torch.from_numpy(image_beta).to(torch.float), \
                       torch.from_numpy(caption_condition).to(torch.float), torch.from_numpy(image_condition).to(torch.float), \
                       torch.from_numpy(text_embeddings).to(torch.float)
            else:
                return caption, image, \
                       torch.from_numpy(caption_beta).to(torch.float), torch.from_numpy(image_beta).to(torch.float), \
                       torch.from_numpy(caption_condition).to(torch.float), torch.from_numpy(image_condition).to(torch.float)


class Caption_Image_2d_dataset(data.Dataset):
    def __init__(self, beta_list, Stimulus_index, Stimulus_pairs, stim_dict, sph_coords, num_vertices, image_size, image_root):
        self.beta_list = beta_list
        self.Stimulus_index = Stimulus_index
        self.Stimulus_pairs = Stimulus_pairs
        self.stim_dict = stim_dict
        self.sph_coords = sph_coords
        self.num_vertices = num_vertices
        self.image_size = image_size
        self.image_root = image_root
        self.transform = image_transform()

    def __len__(self):
        return len(self.beta_list)

    def __getitem__(self, item):
        beta = self.beta_list[item]
        if beta[-5] != 'a':
            stim = int(beta[-24:-19])
        else:
            stim = int(beta[-22:-17])

        if self.Stimulus_index[stim][-3:] == 'jpg':
            image_stim = stim
            caption_stim = self.Stimulus_index[self.Stimulus_pairs[self.Stimulus_index[stim]]]

            img_beta = beta
            cap_beta = random.choice(self.stim_dict[caption_stim])
        else:
            image_stim = self.Stimulus_index[self.Stimulus_pairs[self.Stimulus_index[stim]]]
            caption_stim = stim

            img_beta = random.choice(self.stim_dict[image_stim])
            cap_beta = beta

        caption = self.Stimulus_index[caption_stim]
        image_dir = os.path.join(self.image_root, self.Stimulus_index[image_stim])
        image = self.transform(Image.open(image_dir))

        caption_mat = io.loadmat(cap_beta)
        # caption_beta = np.squeeze(caption_mat['beta'])
        # caption_beta_2d = np.zeros((1, 2*self.image_size, self.image_size))
        # caption_beta_2d[0, :self.image_size, :] = fMRI2grid(caption_beta[:self.num_vertices[0]], self.sph_coords[:self.num_vertices[0], 0], self.sph_coords[:self.num_vertices[0], 1], self.image_size)
        # caption_beta_2d[0, self.image_size:, :] = fMRI2grid(caption_beta[self.num_vertices[0]:], self.sph_coords[self.num_vertices[0]:, 0], self.sph_coords[self.num_vertices[0]:, 1], self.image_size)
        caption_beta_2d = caption_mat['beta_2d']
        # caption_condition = np.array([
        #     caption_mat['run'][0][0],
        #     caption_mat['timestamp'][0][0], caption_mat['behavior'][0][0], caption_mat['behavior_rt'][0][0],
        #     caption_mat['caption'][0][0], caption_mat['image'][0][0], caption_mat['unmatch'][0][0],
        #     caption_mat['repeat'][0][0]
        # ])

        image_mat = io.loadmat(img_beta)
        # image_beta = np.squeeze(image_mat['beta'])
        # image_beta_2d = np.zeros((1, 2*self.image_size, self.image_size))
        # image_beta_2d[0, :self.image_size, :] = fMRI2grid(image_beta[:self.num_vertices[0]], self.sph_coords[:self.num_vertices[0], 0], self.sph_coords[:self.num_vertices[0], 1], self.image_size)
        # image_beta_2d[0, self.image_size:, :] = fMRI2grid(image_beta[self.num_vertices[0]:], self.sph_coords[self.num_vertices[0]:, 0], self.sph_coords[self.num_vertices[0]:, 1], self.image_size)
        image_beta_2d = image_mat['beta_2d']
        # image_condition = np.array([
        #     image_mat['run'][0][0],
        #     image_mat['timestamp'][0][0], image_mat['behavior'][0][0], image_mat['behavior_rt'][0][0],
        #     image_mat['caption'][0][0], image_mat['image'][0][0], image_mat['unmatch'][0][0], image_mat['repeat'][0][0]
        # ])

        return caption, image, torch.from_numpy(caption_beta_2d).to(torch.float), torch.from_numpy(image_beta_2d).to(
            torch.float)


class Predictive_coding_dataset(data.Dataset):
    def __init__(self, beta_list, Stimulus_index, Stimulus_pairs, stim_dict, image_root, norm=True, ne=False, te=False,
                 image_size=224,
                 generated_image_root='/public_bme/data/lishr/COCO_CN/Generated_images_480'):
        self.beta_list = beta_list
        self.Stimulus_index = Stimulus_index
        self.Stimulus_pairs = Stimulus_pairs
        self.stim_dict = stim_dict
        self.image_root = image_root
        self.generated_image_root = generated_image_root
        self.ne = ne
        self.te = te
        if norm:
            self.transform = image_transform()
        else:
            self.transform = Compose([
                Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
            ])

    def __len__(self):
        return len(self.beta_list)

    def __getitem__(self, item):
        beta = self.beta_list[item]
        if beta[-5] != 'a':
            stim = int(beta[-24:-19])
        else:
            stim = int(beta[-22:-17])

        image_mat = io.loadmat(beta)
        beta = np.squeeze(image_mat['beta'])
        condition = np.array([
            image_mat['run'][0][0],
            image_mat['timestamp'][0][0], image_mat['behavior'][0][0], image_mat['behavior_rt'][0][0],
            image_mat['caption'][0][0], image_mat['image'][0][0], image_mat['unmatch'][0][0], image_mat['repeat'][0][0]
        ])

        pair_stim = image_mat['pair_stim'][0][0]
        pair_image_stim = self.Stimulus_index[self.Stimulus_pairs[self.Stimulus_index[pair_stim]]]

        generated_image_dir = os.path.join(self.generated_image_root, self.Stimulus_index[pair_image_stim])
        generated_image = self.transform(Image.open(generated_image_dir))

        image_dir = os.path.join(self.image_root, self.Stimulus_index[stim])
        image = self.transform(Image.open(image_dir))

        return generated_image, image, \
               0, torch.from_numpy(beta).to(torch.float), \
               0, torch.from_numpy(condition).to(torch.float)


class natural_scene_dataset(data.Dataset):
    def __init__(self, root, beta_list, sph_coords, num_vertices, image_size):
        self.root = root
        self.beta_list = beta_list
        self.sph_coords = sph_coords
        self.num_vertices = num_vertices
        self.image_size = image_size

    def __len__(self):
        return len(self.beta_list)

    def __getitem__(self, item):
        dir = self.beta_list[item]
        subject = dir[-16:-10]
        # beta = np.squeeze(io.loadmat(os.path.join(self.root, dir))['beta'])
        # beta_2d = np.zeros((1, 2*self.image_size, self.image_size))
        # beta_2d[0, :self.image_size, :] = fMRI2grid(beta[:self.num_vertices[subject]], self.sph_coords[subject][:self.num_vertices[subject], 0],
        #                                             self.sph_coords[subject][:self.num_vertices[subject], 1], self.image_size)
        # beta_2d[0, self.image_size:, :] = fMRI2grid(beta[self.num_vertices[subject]:], self.sph_coords[subject][self.num_vertices[subject]:, 0],
        #                                             self.sph_coords[subject][self.num_vertices[subject]:, 1], self.image_size)
        # beta_2d = beta_2d / 300

        beta_2d = io.loadmat(os.path.join(self.root, dir))['beta_2d']

        return torch.from_numpy(beta_2d).to(torch.float)


class ImageNetDataset(data.Dataset):
    def __init__(self, root, norm=True):
        self.root = root
        self.list = os.listdir(self.root)
        if norm:
            self.transform = image_transform()
        else:
            self.transform = Compose([
                                Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                                _convert_to_rgb,
                                ToTensor(),
                            ])

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        image_dir = os.path.join(self.root, self.list[item])
        return self.transform(Image.open(image_dir))
