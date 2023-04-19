import os
import json
import random

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption


class coco_karpathy_train_knowledge(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='',sample_rate=0.5):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_train_text_knowledge.json'
        # image_knowledge_name = "coco_train_visual_knowledge.json"
        # filename = 'example_has_knowledge.json'
        # download_url(url, ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        # self.image_knowledge = json.load(open(os.path.join(ann_root, image_knowledge_name), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.sample_rate = sample_rate

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image_id'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.prompt + pre_caption(ann['caption'], self.max_words)

        knowledge_conceptnet = ''
        knowledge_vg = ''

        for i in random.sample(ann['knowledge_vg'], round(len(ann['knowledge_vg']) * self.sample_rate)):
            knowledge_vg += i
            knowledge_vg += ','
        knowledge_vg = pre_caption(knowledge_vg, self.max_words)
        knowledge_vg = ' Related information: '+knowledge_vg
        # knowledge_conceptnet_image = ''
        # knowledge_vg_image = ''
        #
        # for i in random.sample(self.image_knowledge[ann['image_id']]['knowledge_conceptnet'], round(len(
        #         self.image_knowledge[ann['image_id']]['knowledge_conceptnet']) * self.sample_rate)):
        #     knowledge_conceptnet += i
        #     knowledge_conceptnet += ','
        # knowledge_conceptnet = pre_caption(knowledge_conceptnet, self.max_words)

        caption = caption + knowledge_vg
        return image, caption, self.img_ids[ann['image_id']], knowledge_conceptnet, knowledge_vg



class coco_karpathy_retrieval_eval_knowledge(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val': 'coco_val_text_knowledge.json', 'test': 'coco_test_text_knowledge.json'}
        # filenames = {'val': 'example_has_knowledge.json', 'test': 'example_has_knowledge.json'}
        # download_url(urls[split], ann_root)
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        # print(len(self.annotation))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        # self.knowledge_cc = []
        self.all_knowledge_vg = []
        self.txt2img = {}
        self.img2txt = {}

        img_id = 0
        pre_img_id = ''
        txt_id = 0
        flag = False
        for idx, ann in enumerate(self.annotation):
            # # if idx == 1000:
            # #     break
            # # print(ann['image_id'] != pre_img_id)
            if ann['image_id'] != pre_img_id:
                # print(idx)
                # print(img_id == 0)
                if not flag:
                    flag = True
                else:
                    img_id += 1
                self.image.append(ann['image_id'])

                pre_img_id = ann['image_id']
                # print(self.image[img_id])
                # print(img_id)

            if not self.img2txt.__contains__(img_id):
            # img_id = idx
                self.img2txt[img_id] = []

            # knowledge_conceptnet = ''
            knowledge_vg = ''
            # print(self.sample_rate)
            for i in ann['knowledge_vg']:
                knowledge_vg += i
                knowledge_vg += ','
            knowledge_vg = pre_caption(knowledge_vg, self.max_words)
            knowledge_vg = '. Related information: ' + knowledge_vg

            text =pre_caption(ann['caption'], max_words) + knowledge_vg
            self.text.append(text)
            # print(text)


            # self.all_knowledge_vg.append(knowledge_vg)
            # print(img_id)
            # print(txt_id)
            self.img2txt[img_id].append(txt_id)
            # print(img_id)
            self.txt2img[txt_id] = img_id
            # self.k2img[idx] = img_id
            txt_id += 1

            # for i, caption in enumerate(ann['caption']):
            #     self.text.append(pre_caption(caption, max_words))
            #     self.img2txt[img_id].append(txt_id)
            #     self.txt2img[txt_id] = img_id
            #     txt_id += 1
            # print(len(self.img2txt))
            # knowledge_conceptnet = ''
            #
            # for k in ann['knowledge_conceptnet']:
            #     knowledge_conceptnet += k
            #     knowledge_conceptnet += ','
            # knowledge_conceptnet = pre_caption(knowledge_conceptnet, self.max_words)
            # self.knowledge_cc.append(knowledge_conceptnet)

        # print(len(self.text))
        # print(len(self.knowledge_cc))
        # print(knowledge_conceptnet.shape)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index

        # return image, index, knowledge_conceptnet, knowledge_vg

class coco_caption_train_knowledge(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='',sample_rate=0.5):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_caption_train_has_knowledge.json'
        # filename = 'example_has_knowledge.json'
        # download_url(url, ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.sample_rate = sample_rate

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image_id'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.prompt + pre_caption(ann['caption'], self.max_words)

        knowledge_conceptnet = ''
        knowledge_vg = ''

        for i in random.sample(ann['knowledge_conceptnet'],round(len(ann['knowledge_conceptnet'])*self.sample_rate)):
            knowledge_conceptnet += i
            knowledge_conceptnet += ','
        knowledge_conceptnet = pre_caption(knowledge_conceptnet,self.max_words)


        return image, caption, self.img_ids[ann['image_id']], knowledge_conceptnet, knowledge_vg

class coco_karpathy_caption_eval_knowledge(Dataset):
    def __init__(self, transform, image_root, ann_root, split):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val': 'coco_caption_train_has_knowledge.json', 'test': 'coco_karpathy_test.json'}

        # download_url(urls[split], ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]

        return image, int(img_id)

class coco_karpathy_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'
        # filename = 'coco_karpathy_val.json'
        download_url(url,ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
    
class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))

        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)   
    
    
class coco_karpathy_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        print('len(self.annotation)')
        print(len(self.annotation))
        self.transform = transform
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
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index