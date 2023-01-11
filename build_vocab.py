import spacy
from tqdm import tqdm
import json
import re
import pandas as pd
from collections import Counter+
import csv

###########data format###########
#idx, entity, caption, image_id

plural = True
#####读取数据###########

if plural:
    print('COCO captions_train2014 and captions_val2014 ')
else:
    print('COCO captions_train2014 and captions_val2014 For vocab no plural')

# path='/data1/yangzhenbang_new/blip/BLIP/annotation/coco_karpathy_train.json'
# path1='/data1/yangzhenbang_new/blip/BLIP/annotation/coco_karpathy_val.json'
path = '/data/cc12m/annotations/cc12m_split_1.csv'
path1 = '/data/cc12m/annotations/cc12m_split_0.csv'


# with open(path,'r',encoding='utf8')as fp:
#     json_data = json.load(fp)
#     json_data = json_data['annotations']
#
# with open(path1,'r',encoding='utf8')as fp:
#     json_data1 = json.load(fp)
#     json_data1 = json_data1['annotations']

# with open(path,'r',encoding='utf8')as fp:
#     json_data = json.load(fp)


# with open(path1,'r',encoding='utf8')as fp:
#     json_data1 = json.load(fp)
with open(path, 'r') as f:
    c_reader = csv.reader(f)
    hear_row = next(c_reader)

nlp = spacy.load("en_core_web_sm")

# json_data.extend(json_data1)
entity = []
# print(json_data)
for idx, i in enumerate(tqdm(c_reader)):
    # if idx == 5:
    #     break
    caption = (i[3])
    # image_id = i['image_id']
    image_id = i[1]
    if type(caption) == str:
        caption = caption.lower()
        doc = nlp(caption)
        for chunk in doc.noun_chunks:
            # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
            if ' and ' in str(chunk):
                l = re.split(r' and ', str(chunk))

                for i in l:
                    doc1 = nlp(i)
                    item = ''
                    for w in doc1:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP')
                                and w.text != 'the' and w.text != 'a'):
                            if plural:
                                if item == '':
                                    item = w.text
                                else:
                                    item = item + ' ' + w.text
                            else:
                                if item == '':
                                    item = w.lemma_
                                else:
                                    item = item + ' ' + w.lemma_

                    if item != '' and item != ' ':
                        entity.append(item)
            else:
                item = ''
                for w in chunk:
                    if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP')
                            and w.text != 'the' and w.text != 'a'):
                        if plural:
                            if item == '':
                                item = w.text
                            else:
                                item = item + ' ' + w.text
                        else:
                            if item == '':
                                item = w.lemma_
                            else:
                                item = item + ' ' + w.lemma_
                if item != '' and item != ' ':
                    entity.append(item)
    else:
        for j in caption:
            j = j.lower()
            doc = nlp(j)
            for chunk in doc.noun_chunks:
                # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
                if ' and ' in str(chunk):
                    l = re.split(r' and ', str(chunk))

                    for i in l:
                        doc1 = nlp(i)
                        item = ''
                        for w in doc1:
                            if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP')
                                    and w.text != 'the' and w.text != 'a'):
                                if plural:
                                    if item == '':
                                        item = w.text
                                    else:
                                        item = item + ' ' + w.text
                                else:
                                    if item == '':
                                        item = w.lemma_
                                    else:
                                        item = item + ' ' + w.lemma_

                        if item != '' and item != ' ':
                            entity.append(item)
                else:
                    item = ''
                    for w in chunk:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP')
                                and w.text != 'the' and w.text != 'a'):
                            if plural:
                                if item == '':
                                    item = w.text
                                else:
                                    item = item + ' ' + w.text
                            else:
                                if item == '':
                                    item = w.lemma_
                                else:
                                    item = item + ' ' + w.lemma_
                    if item != '' and item != ' ':
                        entity.append(item)

with open(path1, 'r') as f:
    c_reader1 = csv.reader(f)
    hear_row = next(c_reader1)
for idx, i in enumerate(tqdm(c_reader1)):
    # if idx == 5:
    #     break
    caption = (i[3])
    # image_id = i['image_id']
    image_id = i[1]
    if type(caption) == str:
        caption = caption.lower()
        doc = nlp(caption)
        for chunk in doc.noun_chunks:
            # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
            if ' and ' in str(chunk):
                l = re.split(r' and ', str(chunk))

                for i in l:
                    doc1 = nlp(i)
                    item = ''
                    for w in doc1:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP')
                                and w.text != 'the' and w.text != 'a'):
                            if plural:
                                if item == '':
                                    item = w.text
                                else:
                                    item = item + ' ' + w.text
                            else:
                                if item == '':
                                    item = w.lemma_
                                else:
                                    item = item + ' ' + w.lemma_

                    if item != '' and item != ' ':
                        entity.append(item)
            else:
                item = ''
                for w in chunk:
                    if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP')
                            and w.text != 'the' and w.text != 'a'):
                        if plural:
                            if item == '':
                                item = w.text
                            else:
                                item = item + ' ' + w.text
                        else:
                            if item == '':
                                item = w.lemma_
                            else:
                                item = item + ' ' + w.lemma_
                if item != '' and item != ' ':
                    entity.append(item)
    else:
        for j in caption:
            j = j.lower()
            doc = nlp(j)
            for chunk in doc.noun_chunks:
                # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
                if ' and ' in str(chunk):
                    l = re.split(r' and ', str(chunk))

                    for i in l:
                        doc1 = nlp(i)
                        item = ''
                        for w in doc1:
                            if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP')
                                    and w.text != 'the' and w.text != 'a'):
                                if plural:
                                    if item == '':
                                        item = w.text
                                    else:
                                        item = item + ' ' + w.text
                                else:
                                    if item == '':
                                        item = w.lemma_
                                    else:
                                        item = item + ' ' + w.lemma_

                        if item != '' and item != ' ':
                            entity.append(item)
                else:
                    item = ''
                    for w in chunk:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP')
                                and w.text != 'the' and w.text != 'a'):
                            if plural:
                                if item == '':
                                    item = w.text
                                else:
                                    item = item + ' ' + w.text
                            else:
                                if item == '':
                                    item = w.lemma_
                                else:
                                    item = item + ' ' + w.lemma_
                    if item != '' and item != ' ':
                        entity.append(item)


# #####读取数据###########
# print('flickr30k For vocab')


# path='/data1/yangzhenbang_new/blip/BLIP/annotation/flickr30k_train.json'
# path1='/data1/yangzhenbang_new/blip/BLIP/annotation/flickr30k_val.json'


# with open(path,'r',encoding='utf8')as fp:
#     json_data = json.load(fp)

# with open(path1,'r',encoding='utf8')as fp:
#     json_data1 = json.load(fp)

# nlp = spacy.load("en_core_web_sm")

# json_data.extend(json_data1)
# # entities = []
# entity = []
# for idx,i in enumerate(tqdm(json_data)):
#     # if idx == 5:
#     #     break
#     caption = i['caption']
#     # image_id = i['image_id']
#     image_id = i['image']
#     if type(caption) == str:
#         doc = nlp(caption.lower())
#         for chunk in doc.noun_chunks:
#             # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
#             if ' and ' in str(chunk):
#                 l = re.split(r' and ', str(chunk))

#                 for i in l:
#                     doc1 = nlp(i)
#                     item = ''
#                     for w in doc1:
#                         if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
#                                 and w.text != 'the' and w.text != 'a'):
#                             if plural:
#                                 if item == '':
#                                     item = w.text
#                                 else:
#                                     item = item + ' ' + w.text
#                             else:
#                                 if item == '':
#                                     item = w.lemma_
#                                 else:
#                                     item = item + ' ' + w.lemma_

#                     if item != '' and item != ' ':
#                         entity.append(item)
#             else:
#                 item = ''
#                 for w in chunk:
#                     if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
#                             and w.text != 'the' and w.text != 'a'):
#                         if plural:
#                             if item == '':
#                                 item = w.text
#                             else:
#                                 item = item + ' ' + w.text
#                         else:
#                             if item == '':
#                                 item = w.lemma_
#                             else:
#                                 item = item + ' ' + w.lemma_
#                 if item != '' and item != ' ':
#                     entity.append(item)
#     else:
#         for j in caption:
#             doc = nlp(j.lower())
#             for chunk in doc.noun_chunks:
#                 # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
#                 if ' and ' in str(chunk):
#                     l = re.split(r' and ', str(chunk))

#                     for i in l:
#                         doc1 = nlp(i)
#                         item = ''
#                         for w in doc1:
#                             if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
#                                     and w.text != 'the' and w.text != 'a'):
#                                 if plural:
#                                     if item == '':
#                                         item = w.text
#                                     else:
#                                         item = item + ' ' + w.text
#                                 else:
#                                     if item == '':
#                                         item = w.lemma_
#                                     else:
#                                         item = item + ' ' + w.lemma_

#                         if item != '' and item != ' ':
#                             entity.append(item)
#                 else:
#                     item = ''
#                     for w in chunk:
#                         if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
#                                 and w.text != 'the' and w.text != 'a'):
#                             if plural:
#                                 if item == '':
#                                     item = w.text
#                                 else:
#                                     item = item + ' ' + w.text
#                             else:
#                                 if item == '':
#                                     item = w.lemma_
#                                 else:
#                                     item = item + ' ' + w.lemma_
#                     if item != '' and item != ' ':
#                         entity.append(item)


# #########vg
# print('VG For vocab')


# path='/data1/yangzhenbang_new/blip/BLIP/annotation/vg_caption.json'


# with open(path,'r',encoding='utf8')as fp:
#     json_data = json.load(fp)

# # with open(path1,'r',encoding='utf8')as fp:
# #     json_data1 = json.load(fp)

# nlp = spacy.load("en_core_web_sm")


# # entities = []
# entity = []
# for idx,i in enumerate(tqdm(json_data)):
#     # if idx == 5:
#     #     break
#     caption = i['caption']
#     # image_id = i['image_id']
#     image_id = i['image']
#     if type(caption) == str:
#         doc = nlp(caption.lower())
#         for chunk in doc.noun_chunks:
#             # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
#             if ' and ' in str(chunk):
#                 l = re.split(r' and ', str(chunk))

#                 for i in l:
#                     doc1 = nlp(i)
#                     item = ''
#                     for w in doc1:
#                         if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
#                                 and w.text != 'the' and w.text != 'a'):
#                             if plural:
#                                 if item == '':
#                                     item = w.text
#                                 else:
#                                     item = item + ' ' + w.text
#                             else:
#                                 if item == '':
#                                     item = w.lemma_
#                                 else:
#                                     item = item + ' ' + w.lemma_

#                     if item != '' and item != ' ':
#                         entity.append(item)
#             else:
#                 item = ''
#                 for w in chunk:
#                     if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
#                             and w.text != 'the' and w.text != 'a'):
#                         if plural:
#                             if item == '':
#                                 item = w.text
#                             else:
#                                 item = item + ' ' + w.text
#                         else:
#                             if item == '':
#                                 item = w.lemma_
#                             else:
#                                 item = item + ' ' + w.lemma_
#                 if item != '' and item != ' ':
#                     entity.append(item)
#     else:
#         for j in caption:
#             doc = nlp(j.lower())
#             for chunk in doc.noun_chunks:
#                 # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
#                 if ' and ' in str(chunk):
#                     l = re.split(r' and ', str(chunk))

#                     for i in l:
#                         doc1 = nlp(i)
#                         item = ''
#                         for w in doc1:
#                             if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
#                                     and w.text != 'the' and w.text != 'a'):
#                                 if plural:
#                                     if item == '':
#                                         item = w.text
#                                     else:
#                                         item = item + ' ' + w.text
#                                 else:
#                                     if item == '':
#                                         item = w.lemma_
#                                     else:
#                                         item = item + ' ' + w.lemma_

#                         if item != '' and item != ' ':
#                             entity.append(item)
#                 else:
#                     item = ''
#                     for w in chunk:
#                         if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
#                                 and w.text != 'the' and w.text != 'a'):
#                             if plural:
#                                 if item == '':
#                                     item = w.text
#                                 else:
#                                     item = item + ' ' + w.text
#                             else:
#                                 if item == '':
#                                     item = w.lemma_
#                                 else:
#                                     item = item + ' ' + w.lemma_
#                     if item != '' and item != ' ':
#                         entity.append(item)

count = Counter(entity)
count = count.most_common()
count = dict(count)
res = list(count.keys())
# print(res)
# print(count)

if plural:
    with open("/data1/yangzhenbang_new/datasets/blip_caption/cc12m_vocab.json", 'w', encoding='utf-8') as fw:
        json.dump(res, fw, indent=4)
    print("vocab")

    with open("/data1/yangzhenbang_new/datasets/blip_caption/cc12m_vocab_count.json", 'w',
              encoding='utf-8') as fw:
        json.dump(count, fw, indent=4)
    print("vocab_count")

else:
    with open("/data1/yangzhenbang_new/datasets/blip_caption/cc12m_vocab_noplural.json", 'w',
              encoding='utf-8') as fw:
        json.dump(res, fw, indent=4)
    print("vocab_noplural")

    with open("/data1/yangzhenbang_new/datasets/blip_caption/cc12m_vocab_count_noplural.json", 'w',
              encoding='utf-8') as fw:
        json.dump(count, fw, indent=4)
    print("vocab_noplural_count")


#
# path='/data1/yangzhenbang_new/datasets/Flickr30k/results_20130124.token'
#
# annotations=pd.read_table(path, sep='\t', header=None,names=['image', 'caption'])
# data = annotations['caption']
# image = annotations['image']
#
#
# for idx,i in enumerate(tqdm(data)):
#     # if idx == 5:
#     #     break
#
#     # caption = i['caption']
#     i.lower()
#     doc = nlp(i)
#     for chunk in doc.noun_chunks:
#         # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
#         if ' and ' in str(chunk):
#             l = re.split(r' and ', str(chunk))
#
#             for i in l:
#                 doc1 = nlp(i)
#                 item = ''
#                 for w in doc1:
#                     if((w.tag_ == 'NN' or w.tag_== 'NNS' or w.tag_== 'NNPS' or w.tag_== 'NNP')\
#                             and w.text != 'the' and w.text != 'a'):
#                         if plural:
#                             if item == '':
#                                 item = w.text
#                             else:
#                                 item = item + ' ' + w.text
#                         else:
#                             if item == '':
#                                 item = w.lemma_
#                             else:
#                                 item = item + ' ' + w.lemma_
#                 if item != '' and item != ' ':
#                     entity.append(item.lower())
#         else:
#             item = ''
#             for w in chunk:
#                 if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
#                         and w.text != 'the' and w.text != 'a'):
#                     if plural:
#                         if item == '':
#                             item = w.text
#                         else:
#                             item = item + ' ' + w.text
#                     else:
#                         if item == '':
#                             item = w.lemma_
#                         else:
#                             item = item + ' ' + w.lemma_
#             if item != '' and item != ' ':
#                 entity.append(item.lower())
