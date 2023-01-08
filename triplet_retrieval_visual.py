import sys
import os
import numpy as np
import json
import time
from tqdm import tqdm
import time
import random

seed = 6666
random.seed(seed)
np.random.seed(seed)

# print("COCO train processing visual")
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/coco_train_visual_concept.json'
# print("COCO val processing visual")
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/coco_val_visual_concept.json'
# print("COCO test processing visual")
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/coco_test_visual_concept.json'

# print("flickr train processing visual")
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/flickr_train_visual_concept.json'
print("flickr val processing visual")
visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/flickr_val_visual_concept.json'
# print("flickr test processing visual")
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/flickr_test_visual_concept.json'


# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/vg_example_visual_concept.json'

# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/example_visual_concept.json'

# print("vg processing visual")
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/vg_visual_concept.json'


with open(visual_concept_path) as f:
    visual_concept = json.load(f)
# visual_concept = list(visual_concept)
image_list = list(visual_concept.keys())
visual_concept = list(visual_concept.values())
# print(visual_concept[0])

knowledge_base_path_conceptnet = "/mnt/data/linbingqian_new_1/project/LearnBeyondCap/kg_data/conceptnet_kg_object_as_key.json"

with open(knowledge_base_path_conceptnet) as f:
    knowledge_base_conceptnet = json.load(f)

knowledge_base_path_vg = "/mnt/data/linbingqian_new_1/project/LearnBeyondCap/kg_data/vg_kg_object_as_key.json"

with open(knowledge_base_path_vg) as f:
    knowledge_base_vg = json.load(f)

entity_threshold = 10#先取text concept，再取visual concept直至取到10个
threshold = 50
result = {}
for idx in tqdm(range(len(visual_concept))):
    entity={}
    Entitie={}
    # print(visual_concept[i])
    image_id= image_list[idx]
    visual_set = visual_concept[idx]
    entity_set=[]
    for item in visual_set:
        entity_set.append(list(item.keys())[0])
    # visual_concept_set = []
    # print('entity_set: ', entity_set)

    # entity_knowledge_conceptnet = {}
    # entity_knowledge_vg = {}
    entity_overlap_knowledge_conceptnet = []
    entity_overlap_knowledge_vg = []
    triplet_threshold = int(threshold/len(entity_set))
    conceptnet_knowledge = []
    vg_knowledge = []
    relation_cc = []
    relation_vg = []
    entity_pair = []  # 记录vg已有的entity_pair
    # print(entity_set)
    # print(type(entity_set))
    # print(type(entity_set[0]))
    for entity_item in entity_set:
        ####get conceptnet knowledge
        count = 0

        for entity_key, knowledge_value in knowledge_base_conceptnet.items():
            if entity_item == entity_key:
                repeat = 0
                while (count <= triplet_threshold) and (len(knowledge_value)>0):
                    knowledge_item = random.sample(knowledge_value, 1)[0]
                    # print(knowledge_item)
                    triplet = knowledge_item.split('#')
                    r = triplet[1]
                    if r not in relation_cc:
                        relation_cc.append(r)
                        conceptnet_knowledge.append(knowledge_item)
                        count = count + 1
                        knowledge_value = list(set(knowledge_value) - set(knowledge_item))
                        # print(count)
                    else:
                        repeat = repeat + 1
                    if repeat == 10:
                        # print('can not find more ')
                        break




                # if len(knowledge_value) > threshold:
                #     entity_knowledge_conceptnet[entity_item] = random.sample(knowledge_value, threshold)
                # else:
                #     entity_knowledge_conceptnet[entity_item] = knowledge_value
                # for other_entity_item in entity_set:
                #     if other_entity_item != entity_item:
                #         for knowledge_item in knowledge_value:
                #             triplet = knowledge_item.split('#')
                #             if triplet[0] == entity_item and triplet[-1] == other_entity_item or triplet[0] == other_entity_item and triplet[-1] == entity_item:
                #                 entity_overlap_knowledge_conceptnet.append(knowledge_item)

                break

        ##get vg knowledge
        count = 0
        count_overlap = 0
        for entity_key, knowledge_value in knowledge_base_vg.items():
            if entity_item == entity_key:

                repeat = 0
                knowledge = knowledge_value
                while (count <= triplet_threshold) and (len(knowledge) > 0):
                    knowledge_item = random.sample(knowledge, 1)[0]
                    # print(knowledge_item)
                    triplet = knowledge_item.split('#')
                    r = triplet[1]
                    if r not in relation_vg:
                        relation_vg.append(r)
                        vg_knowledge.append(knowledge_item)
                        count = count + 1
                        knowledge = list(set(knowledge) - set(knowledge_item))
                        # print(count)
                    else:
                        repeat = repeat + 1
                    if repeat == 10:
                        # print('can not find more ')
                        break

                ####get vg_overlap，将所有符合overlap标准的都选出来
                # if len(knowledge_value) > threshold:
                #     entity_knowledge_vg[entity_item] = random.sample(knowledge_value, threshold)
                # else:
                #     entity_knowledge_vg[entity_item] = knowledge_value
                overlap_knowledge = []  # 存储筛选前的knowledge

                for other_entity_item in entity_set:
                    if other_entity_item != entity_item:
                        for knowledge_item in knowledge_value:
                            triplet = knowledge_item.split('#')
                            if triplet[0] == entity_item and triplet[-1] == other_entity_item or triplet[0] == other_entity_item and triplet[-1] == entity_item:
                                overlap_knowledge.append(knowledge_item)
                #开始筛选
                repeat = 0
                while (count_overlap <= triplet_threshold) and (len(overlap_knowledge) > 0):
                    knowledge_item = random.sample(overlap_knowledge, 1)[0]
                    # print(knowledge_item)
                    triplet = knowledge_item.split('#')
                    pair = triplet[0]+triplet[-1]
                    if pair not in entity_pair:
                        entity_pair.append(pair)
                        entity_overlap_knowledge_vg.append(knowledge_item)
                        count_overlap = count_overlap + 1
                        overlap_knowledge = list(set(overlap_knowledge) - set(knowledge_item))
                        # print(count)
                    else:
                        repeat = repeat + 1
                    if repeat == 10:
                        # print('can not find more ')
                        break

                break


    # print(type(entity[i]))
    entity['idx'] = idx

    entity['entity'] = entity_set
    entity['knowledge_conceptnet'] = conceptnet_knowledge
    entity['knowledge_vg'] = vg_knowledge
    entity['knowledge_vg_overlap'] = entity_overlap_knowledge_vg
    result[image_id] = entity
    # entity[i]['knowledge_conceptnet'] = entity_knowledge_conceptnet
    # entity[i]['knowledge_vg'] = entity_knowledge_vg
    #
    # entity[i]['overlap_knowledge_conceptnet'] = entity_overlap_knowledge_conceptnet

    #print('entity_knowledge: ', entity_knowledge)
    #print('overlap knowledge conceptnet', entity_overlap_knowledge_conceptnet)
    #print('overlap knowledge vg', entity_overlap_knowledge_vg)
#coco
# json.dump(result, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/coco_train_visual_knowledge.json", 'w'))
# json.dump(result, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/coco_val_visual_knowledge.json", 'w'))
# json.dump(result, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/coco_test_visual_knowledge.json", 'w'))

# #vg
# json.dump(result, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/vg_visual_knowledge.json", 'w'))

#flickr
# json.dump(result, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/flickr_train_visual_knowledge.json", 'w'))
json.dump(result, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/flickr_val_visual_knowledge.json", 'w'))
# json.dump(result, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/flickr_test_visual_knowledge.json", 'w'))

#example
# json.dump(result, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/example_visual_knowledge.json", 'w'))
# json.dump(result, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/vg_example_visual_knowledge.json", 'w'))

