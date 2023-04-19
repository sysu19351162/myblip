'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer, Wav2Vec2ForMaskedLM
import transformers
transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F
import os
from torch.nn.modules.loss import _Loss

from models.blip import create_vit, init_tokenizer, load_checkpoint


class CLIP_Pretrain(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/bert_config.json',
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                    
                 embed_dim = 256,     
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        
        if vit=='base':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]     
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
        elif vit=='large':
            from timm.models.helpers import load_custom_pretrained
            from timm.models.vision_transformer import default_cfgs
            load_custom_pretrained(self.visual_encoder,default_cfgs['vit_large_patch16_224_in21k'])        
               
        self.tokenizer = init_tokenizer()   
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased',config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer)) 

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # self.vision_proj = ProjectMLP(in_dim=vision_width,out_dim=embed_dim)#nn.Linear(vision_width, embed_dim)
        # self.text_proj = ProjectMLP(in_dim=text_width,out_dim=embed_dim)#nn.Linear(text_width, embed_dim)
        # self.vision_proj = nn.SyncBatchNorm.convert_sync_batchnorm(self.vision_proj)
        # self.text_proj = nn.SyncBatchNorm.convert_sync_batchnorm(self.text_proj)

        self.temp = nn.Parameter(0.07*torch.ones([]))  

        self.criterion = ClipInfoCELoss()
          
    def forward(self, image, caption):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        logit_scale = torch.ones([])/self.temp
        logit_scale.data = torch.clamp(logit_scale.data, max=100)
        

        image_embeds = self.visual_encoder(image)       
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, 
                              return_tensors="pt").to(image.device)  
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            

        image_feat = self.vision_proj(image_embeds[:,0,:])
        text_feat = self.text_proj(text_output.last_hidden_state[:,0,:])

        image_feat = image_feat / (image_feat.norm(dim=-1, keepdim=True))
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True)+1e-10)

        # cosine similarity as logits


        gathered_image_features = all_gather_with_grad(image_feat)
        gathered_text_features = all_gather_with_grad(text_feat)

        logits_per_image = logit_scale * image_feat @ gathered_text_features.t()
        logits_per_text = logit_scale * text_feat @ gathered_image_features.t()

        loss, _ = self.criterion(logits_per_image, logits_per_text)
        loss /= torch.distributed.get_world_size()
        return loss
          
def clip_pretrain(**kwargs):
    model = CLIP_Pretrain(**kwargs)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


class ClipInfoCELoss(_Loss):
    # def __init__(self, partition_num):
    def __init__(self):
        super(ClipInfoCELoss, self).__init__()

    def forward(self, logits_per_image, logits_per_text):
        bs, l_bs = logits_per_image.shape
        if l_bs == bs:
            labels = torch.arange(len(logits_per_image)).cuda()
        else:
            labels = torch.distributed.get_rank() * bs + torch.arange(0, bs, dtype=torch.long).cuda()

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i+loss_t)/2
        return loss, labels 


