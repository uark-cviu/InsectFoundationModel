# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import torch

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np
import json
import random


class IP102(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split='train'):
        assert split in ['train', 'val', 'test']
        self.image_list = []
        self.label_list = []

        with open(os.path.join(root, f'{split}.txt')) as f:
            lines = f.readlines()

        for line in lines:
            filename, class_id = line.strip().split()
            class_id = int(class_id)

            self.image_list.append(os.path.join(root, 'images', filename))
            self.label_list.append(class_id)

        self.transform = transform

        print(f"Dataset has {len(self.image_list)} images")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = self.label_list[index]
        image = PIL.Image.open(image_path)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


class Insect1MDataset(torch.utils.data.Dataset):
    def __init__(self, root, metadata_filepath, transform=None):
        self.image_list = []
        self.caption_list = []

        with open(metadata_filepath) as f:
            metadata = json.load(f)

        insect_records = metadata['insect_records']
        desc_records = metadata['description_records']
        desc_dict = {}
        self.descriptions = {}

        for item in desc_records:
            desc_dict[item['id']] = item

        for item in insect_records:
            self.image_list.append(item['image_url'])
            attri = []
            for key, value in item.items():
                if key in ['No Taxon', 'image_url']:
                    continue
                attri.append(value)

            caption = 'An image of ' + ', '.join(attri)
            captions = [caption.lower()]
            for desc_id in item['description_ids']:
                assert desc_id in desc_dict
                caption = desc_dict[desc_id]['desciption']

                sentences = caption.split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) <= 5:
                        continue

                    captions.append(sentence.lower())

            self.caption_list.append(captions)

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        caption = np.random.choice(self.caption_list[index])
        for i in range(10):
            try:
                image = PIL.Image.open(image_path)
                image = image.convert("RGB")
                break
            except:
                print(f'Could not open {image_path}')
                index = np.random.randint(0, len(self.image_list))
                image_path = self.image_list[index]
                caption = np.random.choice(self.caption_list[index])

        if self.transform:
            image = self.transform(image)

        return image, caption


class ImageDatasetList(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.image_list = []
        with open(root, "r") as reader:
            for line in reader:
                image_path = line.rstrip("\r\n")
                self.image_list.append(image_path)
        self.transform = transform
        print(f"Dataset has {len(self.image_list)} images")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        for _ in range(10):
            try:
                image = PIL.Image.open(image_path)
                image = image.convert("RGB")
                break
            except:
                print(f'Could not open {image_path}')
                index = random.randint(0, len(self.image_list))
                image_path = self.image_list[index]
        if self.transform:
            image = self.transform(image)
        return image, 0


class ImageDatasetListV2(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.image_list = []
        with open(root, "r") as reader:
            for line in reader:
                image_path = line.rstrip("\r\n")
                self.image_list.append(image_path)
        self.transform = transform
        # self.ref_transform = transforms.AutoAugment()
        self.ref_transform = transform
        print(f"Dataset has {len(self.image_list)} images")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        for _ in range(10):
            try:
                image = PIL.Image.open(image_path)
                image = image.convert("RGB")
                break
            except:
                print(f'Could not open {image_path}')
                index = random.randint(0, len(self.image_list))
                image_path = self.image_list[index]
        ref_image = self.ref_transform(image)
        if self.transform:
            image = self.transform(image)
        return image, ref_image, 0


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    dataset = ImageDatasetList(args.data_path, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
