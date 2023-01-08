#将json文件中的图片去重，分别提取

import sys
import os
import clip
import torch
import argparse
import numpy as np
import json
import time
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader
from data.utils import pre_caption
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import time


class coco_karpathy(Dataset):
    # def __init__(self, transform, image_root, ann_root, split, max_words=30):
    def __init__(self, preprocess, image_root, ann_root, split, max_words=30):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): train or val or test
        '''
        urls = {'train': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json',
                'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'train':'coco_karpathy_train.json', 'val': 'coco_karpathy_val.json', 'test': 'coco_karpathy_test.json'}

        # download_url(urls[split], ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        # self.transform = transforms.Compose([
        #     transforms.Resize(32, interpolation=InterpolationMode.BICUBIC),
        #     transforms.CenterCrop(32),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])
        self.transform = preprocess
        # self.transform = transforms.ToTensor()
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path)
        image = self.transform(image)

        return image, index, self.annotation[index]['image']


class flickr30k(Dataset):
    # def __init__(self, transform, image_root, ann_root, split, max_words=30):
    def __init__(self, image_root, ann_root, split, max_words=30):
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): train or val or test
        '''
        urls = {'train': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_train.json',
                'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'train': 'flickr30k_train.json','val': 'flickr30k_val.json', 'test': 'flickr30k_test.json'}

        # download_url(urls[split], ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transforms.Compose([
        transforms.Resize((360,640),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        ])
        # self.transform = transforms.ToTensor()
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path)
        image = self.transform(image)


        return image, self.annotation[index]['image']

ann_root =  '/mnt/data/yangzhenbang/BLIP/annotation'
#vg
# with open('/mnt/data/yangzhenbang/BLIP/annotation/vg_caption.json','r',encoding='utf8')as fp:
#     vg = json.load(fp)
# vg_pictures=[]
# for i in tqdm(vg):
#     vg_pictures.append(i['image'])
# vg_pictures = list(set(vg_pictures))
# with open("/mnt/data/yangzhenbang/BLIP/annotation/vg_pictures.json", 'w', encoding='utf-8') as fw:
#     json.dump(vg_pictures, fw, indent=4)

# #coco
# with open(os.path.join(ann_root, 'coco_karpathy_train.json'),'r',encoding='utf8')as fp:
#     coco_train = json.load(fp)
# coco_train_pictures=[]
# for i in tqdm(coco_train):
#     coco_train_pictures.append(i['image'])
#
# coco_train_pictures = list(set(coco_train_pictures))
# with open("/mnt/data/yangzhenbang/BLIP/annotation/coco_train_pictures.json", 'w', encoding='utf-8') as fw:
#     json.dump(coco_train_pictures, fw, indent=4)
#
# with open(os.path.join(ann_root, 'coco_karpathy_test.json'),'r',encoding='utf8')as fp:
#     coco_test = json.load(fp)
# coco_test_pictures = []
# for i in tqdm(coco_test):
#     coco_test_pictures.append(i['image'])
#
# coco_test_pictures = list(set(coco_test_pictures))
# with open("/mnt/data/yangzhenbang/BLIP/annotation/coco_test_pictures.json", 'w', encoding='utf-8') as fw:
#     json.dump(coco_test_pictures, fw, indent=4)
#
# with open(os.path.join(ann_root, 'coco_karpathy_val.json'),'r',encoding='utf8')as fp:
#     coco_val = json.load(fp)
# coco_val_pictures = []
# for i in tqdm(coco_val):
#     coco_val_pictures.append(i['image'])
#
# coco_val_pictures = list(set(coco_val_pictures))
# with open("/mnt/data/yangzhenbang/BLIP/annotation/coco_val_pictures.json", 'w', encoding='utf-8') as fw:
#     json.dump(coco_val_pictures, fw, indent=4)
#
# #flickr
# with open(os.path.join(ann_root, 'flickr30k_train.json'),'r',encoding='utf8')as fp:
#     flickr_train = json.load(fp)
#
# flickr_train_pictures=[]
# for i in tqdm(flickr_train):
#     flickr_train_pictures.append(i['image'])
#
# flickr_train_pictures = list(set(flickr_train_pictures))
# with open("/mnt/data/yangzhenbang/BLIP/annotation/flickr_train_pictures.json", 'w', encoding='utf-8') as fw:
#     json.dump(flickr_train_pictures, fw, indent=4)
#
# with open(os.path.join(ann_root, 'flickr30k_test.json'),'r',encoding='utf8')as fp:
#     flickr_test = json.load(fp)
#
# flickr_test_pictures=[]
# for i in tqdm(flickr_test):
#     flickr_test_pictures.append(i['image'])
#
# flickr_test_pictures = list(set(flickr_test_pictures))
# with open("/mnt/data/yangzhenbang/BLIP/annotation/flickr_test_pictures.json", 'w', encoding='utf-8') as fw:
#     json.dump(flickr_test_pictures, fw, indent=4)
#
#
# with open(os.path.join(ann_root, 'flickr30k_val.json'),'r',encoding='utf8')as fp:
#     flickr_val = json.load(fp)
#
# flickr_val_pictures=[]
# for i in tqdm(flickr_val):
#     flickr_val_pictures.append(i['image'])
#
# flickr_val_pictures = list(set(flickr_val_pictures))
# with open("/mnt/data/yangzhenbang/BLIP/annotation/flickr_val_pictures.json", 'w', encoding='utf-8') as fw:
#     json.dump(flickr_val_pictures, fw, indent=4)