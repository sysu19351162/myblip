import os
import json
import random

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption


class nlvr_dataset_knowledge(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30,sample_rate=1):
        '''
        image_root (string): Root directory of images
        ann_root (string): directory to store the annotation file
        split (string): train, val or test
        '''
        urls = {'train': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_train.json',
                'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_dev.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_test.json'}
        filenames = {'train': 'nlvr_train_has_knowledge.json', 'val': 'nlvr_dev_has_knowledge.json', 'test': 'nlvr_test_has_knowledge.json'}

        # download_url(urls[split], ann_root)
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.sample_rate = sample_rate
        self.max_words = max_words
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image0_path = os.path.join(self.image_root, ann['image_id'][0])
        image0 = Image.open(image0_path).convert('RGB')
        image0 = self.transform(image0)

        image1_path = os.path.join(self.image_root, ann['image_id'][1])
        image1 = Image.open(image1_path).convert('RGB')
        image1 = self.transform(image1)

        sentence = pre_caption(ann['sentence'], 40)

        knowledge_conceptnet = ''
        knowledge_vg = ''
        # print(len(ann['knowledge_conceptnet']))
        # print( round(len(ann['knowledge_conceptnet'])* self.sample_rate))
        for item in random.sample(ann['knowledge_conceptnet'], round(len(ann['knowledge_conceptnet'])* self.sample_rate)):
            knowledge_conceptnet += item
            knowledge_conceptnet += ','
        knowledge_conceptnet = pre_caption(knowledge_conceptnet, self.max_words)

        if ann['label'] == 'True':
            label = 1
        else:
            label = 0

        words = sentence.split(' ')

        if 'left' not in words and 'right' not in words:
            if random.random() < 0.5:
                return image0, image1, sentence, label, knowledge_conceptnet, knowledge_vg
            else:
                return image1, image0, sentence, label, knowledge_conceptnet, knowledge_vg
        else:
            if random.random() < 0.5:
                return image0, image1, sentence, label, knowledge_conceptnet, knowledge_vg
            else:
                new_words = []
                for word in words:
                    if word == 'left':
                        new_words.append('right')
                    elif word == 'right':
                        new_words.append('left')
                    else:
                        new_words.append(word)

                sentence = ' '.join(new_words)
                return image1, image0, sentence, label, knowledge_conceptnet, knowledge_vg




class nlvr_dataset(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images 
        ann_root (string): directory to store the annotation file
        split (string): train, val or test
        '''
        urls = {'train':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_train.json',
                'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_dev.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_test.json'}
        filenames = {'train':'nlvr_train.json','val':'nlvr_dev.json','test':'nlvr_test.json'}
        
        download_url(urls[split],ann_root)
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        
        self.transform = transform
        self.image_root = image_root

        
    def __len__(self):
        return len(self.annotation)
    

    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image0_path = os.path.join(self.image_root,ann['images'][0])        
        image0 = Image.open(image0_path).convert('RGB')   
        image0 = self.transform(image0)   
        
        image1_path = os.path.join(self.image_root,ann['images'][1])              
        image1 = Image.open(image1_path).convert('RGB')     
        image1 = self.transform(image1)          

        sentence = pre_caption(ann['sentence'], 40)
        
        if ann['label']=='True':
            label = 1
        else:
            label = 0
            
        words = sentence.split(' ')
        
        if 'left' not in words and 'right' not in words:
            if random.random()<0.5:
                return image0, image1, sentence, label
            else:
                return image1, image0, sentence, label
        else:
            if random.random()<0.5:
                return image0, image1, sentence, label
            else:
                new_words = []
                for word in words:
                    if word=='left':
                        new_words.append('right')
                    elif word=='right':
                        new_words.append('left')        
                    else:
                        new_words.append(word)                    
                        
                sentence = ' '.join(new_words)
                return image1, image0, sentence, label
            
            
        