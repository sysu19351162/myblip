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

# print("COCO train processing")
# entity_file_path = "/mnt/data/yangzhenbang/datasets/blip_caption/coco_train_entity.json"
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/coco_train_visual_concept.json'
# print("COCO val processing")
# entity_file_path = "/mnt/data/yangzhenbang/datasets/blip_caption/data_knowledge/coco_val_entity.json"
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/coco_val_visual_concept.json'
print("COCO test processing")
entity_file_path = "/mnt/data/yangzhenbang/datasets/blip_caption/data_knowledge/coco_test_entity.json"
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/coco_test_visual_concept.json'

# print("flickr train processing")
# entity_file_path = "/mnt/data/yangzhenbang/datasets/blip_caption/flickr_train_entity.json"
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/flickr_train_visual_concept.json'
# print("flickr val processing")
# entity_file_path = "/mnt/data/yangzhenbang/datasets/blip_caption/flickr_val_entity.json"
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/flickr_val_visual_concept.json'
# print("flickr test processing")
# entity_file_path = "/mnt/data/yangzhenbang/datasets/blip_caption/flickr_test_entity.json"
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/flickr_test_visual_concept.json'


# entity_file_path = "/mnt/data/yangzhenbang/datasets/blip_caption/text_entity_example.json"
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/example_visual_concept.json'

# print('cc3m preprocess')
# entity_file_path = "/mnt/data/yangzhenbang/datasets/blip_caption/cc3m_entity.json"

# print("vg processing")
# entity_file_path = "/data1/yangzhenbang/datasets/blip_caption/vg_entity.json"
# visual_concept_path = '/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/vg_visual_concept.json'

with open(entity_file_path) as f:
    entity = json.load(f)

# with open(visual_concept_path) as f:
#     visual_concept = json.load(f)

knowledge_base_path_conceptnet = "/mnt/data/linbingqian_new_1/project/LearnBeyondCap/kg_data/conceptnet_kg_object_as_key.json"

with open(knowledge_base_path_conceptnet) as f:
    knowledge_base_conceptnet = json.load(f)

knowledge_base_path_vg = "/mnt/data/linbingqian_new_1/project/LearnBeyondCap/kg_data/vg_kg_object_as_key.json"

with open(knowledge_base_path_vg) as f:
    knowledge_base_vg = json.load(f)

entity_threshold = 10#先取text concept，再取visual concept直至取到10个
threshold = 50
no_knowledge_num = 0
for i in tqdm(range(len(entity))):
    # print(i)
    # if i==2:
    #     break
#for i in tqdm(range(2)):
    # print(type(entity))
    # print(type(entity[i]))
    # print(i)
    entity_set = entity[i]['entity']
    image_id = entity[i]["image_id"]
    # image_id, sep, tail = image_id.partition('#')
    # image_id = 'flickr30k-images/'+
    # print(image_id)
    # visual_concept_list = visual_concept[image_id]
    # visual_concept_set = []
    # print('entity_set: ', entity_set)
    # for idx,vc in enumerate(visual_concept_list):
    #     vc_entity=list(vc.keys())[0]
    #     if len(entity_set)>=10 and idx ==0:
    #         entity_set = random.sample(entity_set, 10)
    #         break
    #
    #     if len(entity_set) > 10 and idx != 0:
    #         print("Entity Set Error!")
    #         entity_set = random.sample(entity_set, 10)
    #         break
    #
    #     if len(entity_set)==10:
    #         # print("already have ten")
    #         break
    #
    #     if vc_entity not in entity_set:
    #         entity_set.append(vc_entity)

    if len(entity_set)>10:
        entity_set = entity[0:9]
    # entity_set.extend(visual_concept_set)
    # entity_set = list(set(entity_set))
    # print(len(entity_set))
    # print('entity_set: ', entity_set)

    # entity_knowledge_conceptnet = {}
    entity_overlap_knowledge_conceptnet = []
    # entity_knowledge_vg = {}
    entity_overlap_knowledge_vg = []
    if len(entity_set) != 0 :
        triplet_threshold = int(threshold/len(entity_set))
    else:
        triplet_threshold = 1
    count = 0
    conceptnet_knowledge = []
    vg_knowledge = []
    # print(entity_set)
    # print(type(entity_set))
    # print(type(entity_set[0]))
    for entity_item in entity_set:
        relation = []
        count = 0

        # for entity_key, knowledge_value in knowledge_base_conceptnet.items():
        #     if entity_item == entity_key:
        #         repeat = 0
        #         while (count <= triplet_threshold) and (len(knowledge_value)>0):
        #             knowledge_item = random.sample(knowledge_value, 1)[0]
        #             # print(knowledge_item)
        #             triplet = knowledge_item.split('#')
        #             r = triplet[1]
        #             if r not in relation:
        #                 relation.append(r)
        #                 conceptnet_knowledge.append(knowledge_item)
        #                 count = count + 1
        #                 knowledge_value = list(set(knowledge_value) - set(knowledge_item))
        #                 # print(count)
        #             else:
        #                 repeat = repeat + 1
        #             if repeat == 10:
        #                 # print('can not find more ')
        #                 break
        #
        #
        #
        #
        #         # if len(knowledge_value) > threshold:
        #         #     entity_knowledge_conceptnet[entity_item] = random.sample(knowledge_value, threshold)
        #         # else:
        #         #     entity_knowledge_conceptnet[entity_item] = knowledge_value
        #         # for other_entity_item in entity_set:
        #         #     if other_entity_item != entity_item:
        #         #         for knowledge_item in knowledge_value:
        #         #             triplet = knowledge_item.split('#')
        #         #             if triplet[0] == entity_item and triplet[-1] == other_entity_item or triplet[0] == other_entity_item and triplet[-1] == entity_item:
        #         #                 entity_overlap_knowledge_conceptnet.append(knowledge_item)
        #
        #         break

        for entity_key, knowledge_value in knowledge_base_vg.items():
            if entity_item == entity_key:
                repeat = 0
                while (count <= triplet_threshold) and (len(knowledge_value) > 0):
                    knowledge_item = random.sample(knowledge_value, 1)[0]
                    # print(knowledge_item)
                    triplet = knowledge_item.split('#')
                    r = triplet[1]
                    if r not in relation:
                        relation.append(r)
                        vg_knowledge.append(knowledge_item)
                        count = count + 1
                        knowledge_value = list(set(knowledge_value) - set(knowledge_item))
                        # print(count)
                    else:
                        repeat = repeat + 1
                    if repeat == 10:
                        # print('can not find more ')
                        break





                # if len(knowledge_value) > threshold:
                #     entity_knowledge_vg[entity_item] = random.sample(knowledge_value, threshold)
                # else:
                #     entity_knowledge_vg[entity_item] = knowledge_value
                # for other_entity_item in entity_set:
                #     if other_entity_item != entity_item:
                #         for knowledge_item in knowledge_value:
                #             triplet = knowledge_item.split('#')
                #             if triplet[0] == entity_item and triplet[-1] == other_entity_item or triplet[0] == other_entity_item and triplet[-1] == entity_item:
                #                 entity_overlap_knowledge_vg.append(knowledge_item)
                #
                # break

    # print(type(entity[i]))
    # entity[i]['knowledge_conceptnet'] = conceptnet_knowledge
    if vg_knowledge == []:
        vg_knowledge.append(' ')
        print("no knowledge finded ")
        no_knowledge_num +=1
        print(no_knowledge_num)
    entity[i]['knowledge_vg'] = vg_knowledge
    # entity[i]['knowledge_conceptnet'] = entity_knowledge_conceptnet
    # entity[i]['knowledge_vg'] = entity_knowledge_vg
    #
    # entity[i]['overlap_knowledge_conceptnet'] = entity_overlap_knowledge_conceptnet
    # entity[i]['overlap_knowledge_vg'] = entity_overlap_knowledge_vg
    #print('entity_knowledge: ', entity_knowledge)
    #print('overlap knowledge conceptnet', entity_overlap_knowledge_conceptnet)
    #print('overlap knowledge vg', entity_overlap_knowledge_vg)
#coco
# json.dump(entity, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/coco_train_text_knowledge.json", 'w'))
# json.dump(entity, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_knowledge/coco_val_text_knowledge.json", 'w'))
json.dump(entity, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_knowledge/coco_test_text_knowledge.json", 'w'))

# #vg
# json.dump(entity, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/vg_has_knowledge.json", 'w'))

#flickr
# json.dump(entity, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/flickr_train_has_knowledge.json", 'w'))
# json.dump(entity, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/flickr_val_has_knowledge.json", 'w'))
# json.dump(entity, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/flickr_test_has_knowledge.json", 'w'))

#example
# json.dump(entity, open("/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/cc3m_has_knowledge.json", 'w'))

