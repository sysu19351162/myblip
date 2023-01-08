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
import time


class coco_karpathy(Dataset):
    # def __init__(self, transform, image_root, ann_root, split, max_words=30):
    def __init__(self, image_root, ann_root, split, max_words=30):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): train or val or test
        '''
        urls = {'train': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json',
                'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'train':'coco_karpathy_train.json', 'val': 'coco_karpathy_val.json', 'test': 'coco_karpathy_test.json'}

        download_url(urls[split], ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        # self.transform = transform
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
        image = Image.open(image_path).convert('RGB')
        # image = self.transform(image)

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

        download_url(urls[split], ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        # self.transform = transform
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
        image = Image.open(image_path).convert('RGB')
        # image = self.transform(image)

        return image, self.annotation[index]['image']

    ##########读取单词表

object_vocab_file = "/mnt/data/linbingqian_new_1/project/LearnBeyondCap/blip_caption/coco_flickr_vocab_count.json"

with open(object_vocab_file) as f:
    object_vocab = json.load(f)

object_vocab_new = []

for item in list(object_vocab.keys()):
    object_vocab_new.append(item)

vocab = object_vocab_new[:2000]
#基本参数设定
bs = 8
n_worker = 2
###########构建COCO的dataloader
ann_root =  '/mnt/data/yangzhenbang/BLIP/annotation'
coco_image_root = "/mnt/data/datasets/coco/coco2014"
train_coco_dataset = coco_karpathy(coco_image_root, ann_root,'train')
val_coco_dataset = coco_karpathy(coco_image_root, ann_root,'val')
test_coco_dataset = coco_karpathy(coco_image_root, ann_root,'test')
train_coco_loader = DataLoader(train_coco_dataset,batch_size=bs,num_workers=n_worker)
val_coco_loader = DataLoader(val_coco_dataset,batch_size=bs,num_workers=n_worker)
test_coco_loader = DataLoader(test_coco_dataset,batch_size=bs,num_workers=n_worker)

###########构建Flickr30K的dataloader
ann_root =  '/mnt/data/yangzhenbang/BLIP/annotation'
flickr_image_root = "/mnt/data/datasets/flickr30k/flickr30k-images"
train_flickr_dataset = flickr30k(flickr_image_root, ann_root,'train')
val_flickr_dataset = flickr30k(flickr_image_root, ann_root,'val')
test_flickr_dataset = flickr30k(flickr_image_root, ann_root,'test')

train_flickr_loader = DataLoader(train_flickr_dataset,batch_size=bs,num_workers=n_worker)
val_flickr_loader = DataLoader(val_flickr_dataset,batch_size=bs,num_workers=n_worker)
test_flickr_loader = DataLoader(test_flickr_dataset,batch_size=bs,num_workers=n_worker)
#print(vocab)
# print("ok")

#image_path = "/mnt/data/datasets/coco/coco2014/val2014/COCO_val2014_000000535253.jpg"


# with open(caption_file) as f:
#     caption = json.load(f)

visual_concept = []

# for i in tqdm(range(len(caption["images"]))):
for image, index, path in tqdm(test_coco_loader):
    visual_concept_item = {}
    #image_path = "/mnt/data/datasets/coco/coco2014/train2014/" + caption["images"][i]["file_name"]
    # image_path = "/mnt/data/datasets/coco/coco2014/val2014/" + caption["images"][i]["file_name"]
    #
    # image = Image.open(image_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(device)
    model, preprocess = clip.load('ViT-B/32', device)

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in vocab]).to(device)
    # image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(20)

    print("\nTop predictions:\n")

    visual_concept_item["image_path"] = path
    visual_concept_item["concepts"] = {}
    for value, index in zip(values, indices):
        #print(value)
        #print(index)
        print(f"{vocab[index]:>16s}: {100 * value.item():.2f}%")
        visual_concept_item["concepts"][vocab[index]] = value.item()

    visual_concept.append(visual_concept_item)

#preserve_file = "/mnt/data/linbingqian_new_1/project/LearnBeyondCap/" + "visual_concept" + "_train2014.json"
preserve_file = "/mnt/data/linbingqian_new_1/project/LearnBeyondCap/" + "visual_concept" + "_val2014.json"


json.dump(visual_concept, open(preserve_file, 'w'), indent=4)
