import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval, coco_karpathy_train_knowledge, coco_caption_train_knowledge, coco_karpathy_caption_eval_knowledge, coco_karpathy_retrieval_eval_knowledge
from data.nocaps_dataset import nocaps_eval, nocaps_eval_knowledge
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval, flickr30k_train_knowledge, flickr30k_retrieval_eval_knowledge
from data.vqa_dataset import vqa_dataset, vqa_dataset_knowledge
from data.nlvr_dataset import nlvr_dataset, nlvr_dataset_knowledge
from data.pretrain_dataset import pretrain_dataset, pretrain_knowledge_dataset
from transform.randaugment import RandomAugment

def create_dataset(dataset, config, min_scale=0.5):


    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
        
    if dataset=='pretrain':
        dataset = pretrain_dataset(config['train_file'], config['cc3m_file'], config['cc3m_path'], config['laion_path'], transform_train)
        return dataset

    elif dataset=='pretrain_knowledge':
        dataset = pretrain_knowledge_dataset(config['train_file'], config['image_knowledge'], config['laion_path'], config['image_root'], transform_train, config['knowledge_num'],config['max_words'])
        return dataset
    
    elif dataset=='caption_coco':   
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'], prompt=config['prompt'])
        val_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return train_dataset, val_dataset, test_dataset

    elif dataset=='caption_coco_knowledge':
        train_dataset = coco_caption_train_knowledge(transform_train, config['image_root'], config['ann_root'], prompt=config['prompt'])
        val_dataset = coco_karpathy_caption_eval_knowledge(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = coco_karpathy_caption_eval_knowledge(transform_test, config['image_root'], config['ann_root'], 'test')
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='nocaps':   
        val_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return val_dataset, test_dataset

    elif dataset=='nocaps_knowledge':
        val_dataset = nocaps_eval_knowledge(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = nocaps_eval_knowledge(transform_test, config['image_root'], config['ann_root'], 'test')
        return val_dataset, test_dataset
    
    elif dataset=='retrieval_coco':          
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        test_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset, test_dataset
    elif dataset=='retrieval_coco_knowledge':
        train_dataset = coco_karpathy_train_knowledge(transform_train, config['image_root'], config['ann_root'],sample_rate=config['sample_rate'])
        val_dataset = coco_karpathy_retrieval_eval_knowledge(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = coco_karpathy_retrieval_eval_knowledge(transform_test, config['image_root'], config['ann_root'], 'test')
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='retrieval_flickr':          
        train_dataset = flickr30k_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        test_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset, test_dataset

    elif dataset=='retrieval_flickr_knowledge':
        train_dataset = flickr30k_train_knowledge(transform_train, config['image_root'], config['ann_root'],sample_rate=config['sample_rate'])
        val_dataset = flickr30k_retrieval_eval_knowledge(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr30k_retrieval_eval_knowledge(transform_test, config['image_root'], config['ann_root'], 'test')
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='vqa': 
        train_dataset = vqa_dataset(transform_train, config['ann_root'], config['vqa_root'], config['vg_root'], 
                                    train_files = config['train_files'], split='train') 
        test_dataset = vqa_dataset(transform_test, config['ann_root'], config['vqa_root'], config['vg_root'], split='test')
        return train_dataset, test_dataset

    elif dataset=='vqa_knowledge':
        train_dataset = vqa_dataset_knowledge(transform_train, config['ann_root'], config['vqa_root'], config['vg_root'],
                                    train_files = config['train_files'], split='train')
        test_dataset = vqa_dataset_knowledge(transform_test, config['ann_root'], config['vqa_root'], config['vg_root'], split='test')
        return train_dataset, test_dataset
    
    elif dataset=='nlvr': 
        train_dataset = nlvr_dataset(transform_train, config['image_root'], config['ann_root'],'train')
        val_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'val')
        test_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'test')     
        return train_dataset, val_dataset, test_dataset

    elif dataset=='nlvr_knowledge':
        train_dataset = nlvr_dataset_knowledge(transform_train, config['image_root'], config['ann_root'],'train',sample_rate=config['sample_rate'])
        val_dataset = nlvr_dataset_knowledge(transform_test, config['image_root'], config['ann_root'],'val')
        test_dataset = nlvr_dataset_knowledge(transform_test, config['image_root'], config['ann_root'],'test')
        return train_dataset, val_dataset, test_dataset



    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

