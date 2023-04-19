'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.clip_model import clip_pretrain
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from data import create_dataset, create_sampler, create_loader
from train_retrieval import itm_eval
from torch.cuda.amp import autocast as autocast, GradScaler

def train(model, data_loader, optimizer, epoch, device, config, lr_schedule, scaler):
    # train
    model.train()  
    iters_per_epoch = len(data_loader)
    # print(iters_per_epoch)
    # print(len(data_loader.dataset))
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    # metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    data_loader.sampler.set_epoch(epoch)

    for i, (image, caption, knowledge_vg) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print(i)
        # if epoch==0:
        #     warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])
        # print()
        it = iters_per_epoch * epoch + i
        # print('it')
        # print(it)
        # print(lr_schedule[it] )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[it] 

        optimizer.zero_grad()
        
        image = image.to(device,non_blocking=True)
        with autocast():
            loss_ita = model(image, caption, knowledge_vg)
            loss = loss_ita 

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()    

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id
    
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    #将数据分段rank
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1#334
    start = rank*step
    end = min(sims_matrix.size(0),start+step)
    #i2t
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 500, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        score_matrix_i2t[start+i,topk_idx] = topk_sim
    #t2i    
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 500, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        score_matrix_t2i[start+i,topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)      #-100*rank  
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    # train_dataset, test_dataset = create_dataset(config['dataset'], config, min_scale=0.2)
    # train_dataset = create_dataset(config['dataset'], config, min_scale=0.2)'pretrain'
    train_dataset = create_dataset('pretrain', config, min_scale=0.2)
    # train_dataset.annotation = train_dataset.annotation[0:1000]

    datasets = [train_dataset]
    print('number of training samples: %d'%len(datasets[0]))
    # print('number of testing samples: %d'%len(datasets[1]))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        # samplers = create_sampler(train_dataset, [True], num_tasks, global_rank)+ [None]
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)

    else:
        samplers = [None]

    # data_loader, test_loader = create_loader(datasets,samplers,batch_size=[config['batch_size_train']]+[config['batch_size_test']], num_workers=[8,8], is_trains=[True,False], collate_fns=[None,None])
    data_loader = create_loader(datasets, samplers,
                                             batch_size=[config['batch_size_train']],
                                             num_workers=[4], is_trains=[True], collate_fns=[None])[0]
    # print(config['batch_size_train'])
    # print(type(data_loader))
    # print(type(data_loader[0]))
    # print(type(data_loader[0].sampler))
    #### Model #### 
    print("Creating model")
    model = clip_pretrain(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                            vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    lr_schedule = utils.cosine_scheduler(init_lr = config['init_lr'], min_lr = config['min_lr'], warmup_lr = config['warmup_lr'], epochs =config['max_epoch'],
        niter_per_ep = len(data_loader), warmup_epochs=1, warmup_iters = config['warmup_steps'], start_warmup_value=config['init_lr'])

    start_epoch = 0
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict,strict=False)    
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1                
        print('resume checkpoint from %s'%args.checkpoint)    
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module 

    best = 0
    best_epoch = 0   
    scaler = GradScaler()
    print("Start training")
    start_time = time.time()    
    for epoch in range(start_epoch, config['max_epoch']):
        # print("!!!epoch")
        # print(epoch)
        step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])
            
        train_stats = train(model, data_loader, optimizer, epoch, device, config, lr_schedule, scaler)
        # print("!!train end!!!")
        # score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)
         
        if utils.is_main_process():  

            # test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            # print(test_result)
            #
            # if test_result['r_mean']>best:
            #     best = test_result['r_mean']
            #     best_epoch = epoch

            # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #     **{f'test_{k}': v for k, v in test_result.items()},
            #     'epoch': epoch,
            #     'best_epoch': best_epoch,
            # }
            # print("log_stats")
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'best_epoch': best_epoch,
                         }
            # print("save_obj")
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            if epoch == config['max_epoch']-1:
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, "log.json"),"a",encoding="utf-8") as f:
                # f.write(json.dumps(log_stats) + "\n")
                json.dump(log_stats,f,indent=2,ensure_ascii=False)

        dist.barrier()     
        torch.cuda.empty_cache()      
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/clip_knowledge.yaml')
    parser.add_argument('--output_dir', default='output/Pretrain')  
    parser.add_argument('--checkpoint', default='')    
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