'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader
from data.utils import pre_caption
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader
from models.blip_retrieval import blip_retrieval
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

class visual_genome(Dataset):
    # def __init__(self, transform, image_root, ann_root, split, max_words=30):
    def __init__(self, vg_root, max_words=30):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): train or val or test
        '''

        # download_url(urls[split], ann_root)
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.annotation = json.load(open(vg_root, 'r'))
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        # self.transform = transforms.ToTensor()
        self.text = []
        self.image = []

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = self.annotation[index]
        image = Image.open(image_path).convert('RGB')
        # image = Image.open(image_path)
        image = self.transform(image)

        return image, index , self.annotation[index]

class coco_karpathy(Dataset):
    # def __init__(self, transform, image_root, ann_root, split, max_words=30):
    def __init__(self, image_root, ann_root, split, max_words=30):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): train or val or test
        '''

        filenames = {'example':'coco_example_pictures.json','train':'coco_train_pictures.json', 'val': 'coco_val_pictures.json', 'test': 'coco_test_pictures.json'}

        # download_url(urls[split], ann_root)
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        # self.transform = transforms.ToTensor()
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        # txt_id = 0
        # for img_id, ann in enumerate(self.annotation):
        #     self.image.append(ann['image'])
        #     self.img2txt[img_id] = []
        #     for i, caption in enumerate(ann['caption']):
        #         self.text.append(pre_caption(caption, max_words))
        #         self.img2txt[img_id].append(txt_id)
        #         self.txt2img[txt_id] = img_id
        #         txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index])
        image = Image.open(image_path).convert('RGB')
        # image = Image.open(image_path)
        image = self.transform(image)

        return image, index , self.annotation[index]


class flickr30k(Dataset):
    # def __init__(self, transform, image_root, ann_root, split, max_words=30):
    def __init__(self, image_root, ann_root, split, max_words=30):
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): train or val or test
        '''

        filenames = {'example':'flickr_example_pictures.json','train': 'flickr_train_pictures.json','val': 'flickr_val_pictures.json', 'test': 'flickr_test_pictures.json'}

        # download_url(urls[split], ann_root)
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transforms.Compose([
        transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
        # self.transform = transforms.ToTensor()
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        # txt_id = 0
        # for img_id, ann in enumerate(self.annotation):
        #     self.image.append(ann['image'])
        #     self.img2txt[img_id] = []
        #     for i, caption in enumerate(ann['caption']):
        #         self.text.append(pre_caption(caption, max_words))
        #         self.img2txt[img_id].append(txt_id)
        #         self.txt2img[txt_id] = img_id
        #         txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index])
        image = Image.open(image_path).convert('RGB')
        # image = Image.open(image_path)
        image = self.transform(image)


        return image, index, self.annotation[index]

class NLVR(Dataset):
    # def __init__(self, transform, image_root, ann_root, split, max_words=30):
    def __init__(self, image_root, ann_root, split, max_words=30):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): train or val or test
        '''

        filenames = {'train':'nlvr_train_pictures.json', 'val': 'nlvr_dev_pictures.json', 'test': 'nlvr_test_pictures.json'}

        # download_url(urls[split], ann_root)
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        # self.transform = transforms.ToTensor()
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        # txt_id = 0
        # for img_id, ann in enumerate(self.annotation):
        #     self.image.append(ann['image'])
        #     self.img2txt[img_id] = []
        #     for i, caption in enumerate(ann['caption']):
        #         self.text.append(pre_caption(caption, max_words))
        #         self.img2txt[img_id].append(txt_id)
        #         self.txt2img[txt_id] = img_id
        #         txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index])
        image = Image.open(image_path).convert('RGB')
        # image = Image.open(image_path)
        image = self.transform(image)

        return image, index , self.annotation[index]



def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)                  
        loss = loss_ita + loss_itm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()
    print('process text feature')
    #todo:这里可以直接读取vocab,下面的for循环可以全部改掉，照着clip的逻辑写
    object_vocab_file = '/mnt/data/yangzhenbang/datasets/blip_caption/coco_flickr_vg_vocab_count.json'
    with open(object_vocab_file) as f:
        texts = json.load(f)
    num_text = len(texts)
    # text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    # print(len(texts))
    texts = list(texts)
    for i in tqdm(texts[0:2000]):
        #todo:下面三行的处理还是要的
        text_input = model.tokenizer(i, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        #下面三行可以直接去掉了
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    #下面三行也可以不用管了
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id
    print('process image feature')
    #todo:图像这里就不用管了
    image_feats = []
    image_embeds = []

    result = {}
    for image, img_id, name in tqdm(data_loader):
        image_names = []
        # flag= flag + 1
        # if flag == 2:
        #     break

        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)
        image_feats.append(image_feat.cpu())
        # image_embeds.append(image_embed)
        for i in name:
            image_names.append(i)
        # todo:相似性算完之后直接按照clip取top20
        sims_matrix = image_embed @ text_embeds.t()
        score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)
        # print(score_matrix_i2t.shape)
        num_tasks = utils.get_world_size()
        rank = utils.get_rank()
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)
        sim_list = []
        # print(sims_matrix.size(0))
        # print(sims_matrix.size(1))
        for i, sims in tqdm(enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header))):

            sim_list = []
            topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
            topk_sim = topk_sim.cpu().numpy()
            for idx, j in enumerate(topk_idx.cpu().numpy()):
                sim_list.append({texts[j]: np.float(topk_sim[idx])})
            # print(len(image_names))
            # print(i)
            result[image_names[i]]= sim_list
     
    # image_feats = torch.cat(image_feats,dim=0)
    # image_embeds = torch.cat(image_embeds,dim=0)


        # encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(device)
        # encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        # output = model.text_encoder(text_ids[topk_idx],
        #                             attention_mask = text_atts[topk_idx],
        #                             encoder_hidden_states = encoder_output,
        #                             encoder_attention_mask = encoder_att,
        #                             return_dict = True,
        #                            )
        # score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        # score = score.cpu().numpy()
        # for idx,j in enumerate(topk_idx.cpu().numpy()):
        #     sim_list.append({texts[j]:score[idx]})
        # result.append({image_names[i]:sim_list})


    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        # torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    # print(score_matrix_i2t.shape)
    score_matrix_i2t=score_matrix_i2t.cpu().numpy()
    # print(score_matrix_i2t.shape)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 
    # print(score_matrix_i2t.cpu())
    # return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
    # return score_matrix_i2t.cpu().numpy()sim_list
    # print(result)
    return result


            
@torch.no_grad()
def itm_eval(scores_i2t, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    # #Text->Images
    # ranks = np.zeros(scores_t2i.shape[0])
    #
    # for index,score in enumerate(scores_t2i):
    #     inds = np.argsort(score)[::-1]
    #     ranks[index] = np.where(inds == txt2img[index])[0][0]
    #
    # # Compute metrics
    # ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    # ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    # ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    #
    # tr_mean = (tr1 + tr5 + tr10) / 3
    # ir_mean = (ir1 + ir5 + ir10) / 3
    # r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,}
                    # 'txt_r_mean': tr_mean,
                    # 'img_r1': ir1,
                    # 'img_r5': ir5,
                    # 'img_r10': ir10,
                    # 'img_r_mean': ir_mean,
                    # 'r_mean': r_mean}
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    #设置基本参数
    bs = 1


    #### Dataset #### 
    print("Creating retrieval dataset")
    ann_root = '/mnt/data/yangzhenbang/BLIP/annotation/'
    coco_image_root = "/mnt/data/datasets/coco/coco2014"
    flickr_image_root = '/mnt/data/datasets/flickr30k'
    # vg_root = '/mnt/data/yangzhenbang/BLIP/annotation/vg_pictures.json'
    vg_root = '/mnt/data/yangzhenbang/BLIP/annotation/vg_example_pictures.json'
    nocaps_root = '/mnt/data/datasets/nocaps/'
    nlvr_root = '/mnt/data/datasets/NLVR2/'

    # train_nlvr_dataset = NLVR(nlvr_root, ann_root, 'train')
    # val_nlvr_dataset = NLVR(nlvr_root, ann_root, 'val')
    # test_nlvr_dataset = NLVR(nlvr_root, ann_root, 'test')
    # train_nlvr_loader = DataLoader(train_nlvr_dataset, batch_size=config['batch_size_test'], num_workers=1)
    # val_nlvr_loader = DataLoader(val_nlvr_dataset, batch_size=config['batch_size_test'], num_workers=1)
    # test_nlvr_loader = DataLoader(test_nlvr_dataset, batch_size=config['batch_size_test'], num_workers=1)

    # nocaps_val_dataset = nocaps_eval(nocaps_root,ann_root,'val')
    # nocaps_val_loader = DataLoader(nocaps_val_dataset, batch_size=config['batch_size_test'], num_workers=1)
    # nocaps_test_dataset = nocaps_eval(nocaps_root, ann_root, 'test')
    # nocaps_test_loader = DataLoader(nocaps_test_dataset, batch_size=config['batch_size_test'], num_workers=1)

    vg_dataset = visual_genome(vg_root)
    vg_loader = DataLoader(vg_dataset, batch_size=config['batch_size_test'], num_workers=1)
    # train_coco_dataset = coco_karpathy(coco_image_root, ann_root,'train')
    # val_coco_dataset = coco_karpathy(coco_image_root, ann_root,'val')
    # test_coco_dataset = coco_karpathy(coco_image_root, ann_root, 'test')
    # train_coco_loader = DataLoader(train_coco_dataset, batch_size=config['batch_size_test'], num_workers=1)
    # val_coco_loader = DataLoader(val_coco_dataset, batch_size=config['batch_size_test'], num_workers=1)
    # test_coco_loader = DataLoader(test_coco_dataset, batch_size=config['batch_size_test'], num_workers=1)

    # train_flickr_dataset = flickr30k(flickr_image_root, ann_root, 'train')
    # val_flickr_dataset = flickr30k(flickr_image_root, ann_root,'val')
    # test_flickr_dataset = flickr30k(flickr_image_root, ann_root, 'test')
    # example_flickr_dataset = coco_karpathy(coco_image_root, ann_root, 'example')
    # train_flickr_loader = DataLoader(train_flickr_dataset, batch_size=config['batch_size_test'], num_workers=1)
    # val_flickr_loader = DataLoader(val_flickr_dataset, batch_size=config['batch_size_test'], num_workers=1)
    # test_flickr_loader = DataLoader(test_flickr_dataset, batch_size=config['batch_size_test'], num_workers=1)
    # example_flickr_loader = DataLoader(example_flickr_dataset, batch_size=config['batch_size_test'], num_workers=2)
    # if args.distributed:
    #     num_tasks = utils.get_world_size()
    #     global_rank = utils.get_rank()
    #     samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    # else:
    #     samplers = [None, None, None]
    
    # train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
    #                                                       batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
    #                                                       num_workers=[4,4,4],
    #                                                       is_trains=[True, False, False],
    #                                                       collate_fns=[None,None,None])
   

    #### Model #### 
    print("Creating model")
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    

    # for epoch in range(0, config['max_epoch']):
    args.evaluate=True
    for epoch in range(0,2):
        if not args.evaluate:        
            # if args.distributed:
                # train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
            # train_stats = train(model, train_loader, optimizer, epoch, device, config)
         #todo:修改时增加train的
        #score_train_i2t, score_train_t2i, = evaluation(model_without_ddp, train_coco_loader, device, config)
        # score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_coco_loader, device, config)
        print("Start evaluating")
        #################nlvr
        # test_result = evaluation(model_without_ddp, test_nlvr_loader, device, config)
        # with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/nlvr_test_visual_concept.json", 'w',
        #           encoding='utf-8') as fw:
        #     json.dump(test_result, fw, indent=4)
        # val_result = evaluation(model_without_ddp, val_nlvr_loader, device, config)
        # with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/nlvr_dev_visual_concept.json", 'w',
        #           encoding='utf-8') as fw:
        #     json.dump(val_result, fw, indent=4)
        # train_result = evaluation(model_without_ddp, train_nlvr_loader, device, config)
        # with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_flickr_coco_vg/nlvr_train_visual_concept.json",
        #           'w', encoding='utf-8') as fw:
        #     json.dump(train_result, fw, indent=4)
        # test_result = evaluation(model_without_ddp, vg_loader, device, config)
        # with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/vg_visual_concept.json", 'w',
        #           encoding='utf-8') as fw:
        #     json.dump(test_result, fw, indent=4)

        test_result = evaluation(model_without_ddp, vg_loader, device, config)
        with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/vg_example_visual_concept.json", 'w',
                  encoding='utf-8') as fw:
            json.dump(test_result, fw, indent=4)
        # test_result = evaluation(model_without_ddp, test_coco_loader, device, config)
        # with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/coco_test_visual_concept.json", 'w', encoding='utf-8') as fw:
        #     json.dump(test_result, fw, indent=4)
        # val_result = evaluation(model_without_ddp, val_coco_loader, device, config)
        # with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/coco_val_visual_concept.json", 'w', encoding='utf-8') as fw:
        #     json.dump(val_result, fw, indent=4)
        # train_result = evaluation(model_without_ddp, train_coco_loader, device, config)
        # with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/coco_train_visual_concept.json", 'w', encoding='utf-8') as fw:
        #     json.dump(train_result, fw, indent=4)

        # test_result = evaluation(model_without_ddp, test_flickr_loader, device, config)
        # with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/flickr_test_visual_concept.json", 'w', encoding='utf-8') as fw:
        #     json.dump(test_result, fw, indent=4)
        # val_result = evaluation(model_without_ddp, val_flickr_loader, device, config)
        # with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/flickr_val_visual_concept.json", 'w', encoding='utf-8') as fw:
        #     json.dump(val_result, fw, indent=4)
        # train_result = evaluation(model_without_ddp, train_flickr_loader, device, config)
        # with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/flickr_train_visual_concept.json", 'w', encoding='utf-8') as fw:
        #     json.dump(train_result, fw, indent=4)
        # example_result = evaluation(model_without_ddp, example_flickr_loader, device, config)
        # with open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/example_visual_concept.json", 'w',
        #           encoding='utf-8') as fw:
        #     json.dump(example_result, fw, indent=4)
        if args.evaluate: 
            break

        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('start saving')





    print('Training time {}'.format(total_time_str)) 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    parser.add_argument('--config', default='./configs/visual_concept.yaml')
    parser.add_argument('--output_dir', default='output/Visual_concept')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)