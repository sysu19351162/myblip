import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption
import os,glob

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, cc3m_file, cc3m_path, laion_path, transform):
        self.cc3m_path = cc3m_path
        self.ann_pretrain = []
        for f in ann_file:
            print('loading '+f)
            ann = json.load(open(f,'r'))
            self.ann_pretrain += ann

        for f in cc3m_file:
            print('loading ' + f)
            ann = json.load(open(f, 'r'))
            self.ann_pretrain += ann.values()

        
        self.laion_path = laion_path
        if self.laion_path:
            self.laion_files = glob.glob(os.path.join(laion_path,'*.json'))

            print('loading '+self.laion_files[0])
            with open(self.laion_files[0],'r') as f:
                self.ann_laion = json.load(f)  

            self.annotation = self.ann_pretrain + self.ann_laion
        else:
            self.annotation = self.ann_pretrain
            
        self.transform = transform


    def reload_laion(self, epoch):
        n = epoch%len(self.laion_files)
        print('loading '+self.laion_files[n])
        with open(self.laion_files[n],'r') as f:
            self.ann_laion = json.load(f)      
        
        self.annotation = self.ann_pretrain + self.ann_laion    
        
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        # print(ann)
        # print(ann['img_name'])
        # print(type(ann['img_name']))
        # print(ann['img_name'][0])
        # print(type(ann['img_name'][0]))
        if ann['img_name'][0] == '/':
            image = Image.open(ann['img_name']).convert('RGB')
        else:
            image = Image.open(os.path.join(self.cc3m_path, ann['img_name'])).convert('RGB')
        image = self.transform(image)
        caption = pre_caption(ann['caption'],30)
        caption
        
        return image, caption


class pretrain_knowledge_dataset(Dataset):
    def __init__(self, ann_file, image_knowledge_file, laion_path, image_root, transform,knowledge_num,max_words):

        self.ann_pretrain = []
        for f in ann_file:
            print('loading ' + f)
            ann = json.load(open(f, 'r'))
            self.ann_pretrain += ann
        self.img_knowledge_pretrain ={}
        for f in image_knowledge_file:
            print('loading ' + f)
            ann = json.load(open(f, 'r'))
            # print(type(ann))
            self.img_knowledge_pretrain.update(ann)
        print(type(self.img_knowledge_pretrain))

        self.laion_path = laion_path
        if self.laion_path:
            self.laion_files = glob.glob(os.path.join(laion_path, '*.json'))

            print('loading ' + self.laion_files[0])
            with open(self.laion_files[0], 'r') as f:
                self.ann_laion = json.load(f)

            self.annotation = self.ann_pretrain + self.ann_laion
        else:
            self.annotation = self.ann_pretrain

        self.transform = transform
        self.image_root = image_root
        self.knowledge_num = knowledge_num
        self.max_words = max_words

    def reload_laion(self, epoch):
        n = epoch % len(self.laion_files)
        print('loading ' + self.laion_files[n])
        with open(self.laion_files[n], 'r') as f:
            self.ann_laion = json.load(f)

        self.annotation = self.ann_pretrain + self.ann_laion

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        # print(type(self.img_knowledge_pretrain[ann['image_id']]))
        # print(self.img_knowledge_pretrain[ann['image_id']])
        # print(type(self.img_knowledge_pretrain[ann['image_id']]['knowledge_conceptnet']))
        image_path = os.path.join(self.image_root, ann['image_id'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = pre_caption(ann['caption'], 30)
        # cc_knowledge = random.sample(ann['knowledge_conceptnet'],self.knowledge_num)
        # vg_knowledge = random.sample(ann['knowledge_vg'],self.knowledge_num)
        knowledge_conceptnet = ''
        knowledge_vg = ''
        knowledge_conceptnet_image = ''
        knowledge_vg_image = ''
        #todo：加开关
        if len(self.img_knowledge_pretrain[ann['image_id']]['knowledge_vg_overlap'])< self.knowledge_num:

            # print("knowledge_num")
            # print(self.knowledge_num)
            print("annotation_knowledge_num")
            print(len(self.img_knowledge_pretrain[ann['image_id']]['knowledge_vg_overlap']))
            knowledge_num = len(self.img_knowledge_pretrain[ann['image_id']]['knowledge_vg_overlap'])
            # print("img_id")
            # print(ann['image_id'])
        else:
            knowledge_num = self.knowledge_num

        for i in  random.sample(self.img_knowledge_pretrain[ann['image_id']]['knowledge_vg_overlap'], knowledge_num):
            knowledge_vg_image += i
            knowledge_vg_image += ','
        knowledge_vg_image = pre_caption(knowledge_vg_image,self.max_words)

        # for i in ann['knowledge_conceptnet']:
        #     knowledge_vg += i
        #     knowledge_vg += ','
        # knowledge_vg = pre_caption(knowledge_vg,self.max_words)



        return image, caption, knowledge_conceptnet, knowledge_vg, knowledge_conceptnet_image, knowledge_vg_image