import spacy
from tqdm import tqdm
import json
import re
import pandas as pd
from collections import Counter
plural = False
nlp = spacy.load("en_core_web_sm")
print('Vocab of COCO，Flickr,CC3M,CC12M,SBU')
######构建CC3M、CC12M、SBU的实体表##
#加载数据
# print("ccs_filtered")
path1='/data1/yangzhenbang_new/datasets/blip_caption/entities_ccs_filtered.json'

# print("ccs_synthetic_filtered")
path2='/data1/yangzhenbang_new/datasets/blip_caption/entities_ccs_synthetic_filtered.json'

# print("ccs_synthetic_filtered_large")
path3='/data1/yangzhenbang_new/datasets/blip_caption/entities_ccs_synthetic_filtered_large.json'




entity = []

with open(path1,'r',encoding='utf8')as fp:
    ccs_filtered = json.load(fp)

for i in tqdm(ccs_filtered) :
    if plural:
        w = i['entity']
        entity.extend(w)
    else:
        for temp in i['entity']:
            doc = nlp(temp)
            for w in doc:
                entity.append(w.lemma_)



with open(path2,'r',encoding='utf8')as fp:
    ccs_synthetic_filtered = json.load(fp)

for i in tqdm(ccs_synthetic_filtered) :
    if plural:
        w = i['entity']
        entity.extend(w)
    else:
        for temp in i['entity']:
            doc = nlp(temp)
            for w in doc:
                entity.append(w.lemma_)


with open(path3,'r',encoding='utf8')as fp:
    ccs_synthetic_filtered_large = json.load(fp)

for i in tqdm(ccs_synthetic_filtered_large) :
    if plural:
        w = i['entity']
        entity.extend(w)
    else:
        for temp in i['entity']:
            doc = nlp(temp)
            for w in doc:
                entity.append(w.lemma_)


#####加载COCO和Flickr的实体表######
if plural:
    print('COCO captions_train2014 and captions_val2014 ')
else:
    print('COCO captions_train2014 and captions_val2014 For vocab no plural')
path='/data1/yangzhenbang_new/datasets/coco/annotations/captions_train2014.json'
path1='/data1/yangzhenbang_new/datasets/coco/annotations/captions_val2014.json'

with open(path,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
    json_data = json_data['annotations']

with open(path1,'r',encoding='utf8')as fp:
    json_data1 = json.load(fp)
    json_data1 = json_data1['annotations']

# entities = []
entity = []
for idx,i in enumerate(tqdm(json_data)):
    # if idx == 5:
    #     break
    caption = (i['caption']).lower()
    image_id = i['image_id']
    doc = nlp(caption)
    for chunk in doc.noun_chunks:
        # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
        if ' and ' in str(chunk):
            l = re.split(r' and ', str(chunk))

            for i in l:
                doc1 = nlp(i)
                item = ''
                for w in doc1:
                    if((w.tag_ == 'NN' or w.tag_== 'NNS' or w.tag_== 'NNPS' or w.tag_== 'NNP')\
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
                if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
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


#####读取数据###########
print('flickr30k For vocab')
path='/data1/yangzhenbang_new/datasets/Flickr30k/results_20130124.token'

annotations=pd.read_table(path, sep='\t', header=None,names=['image', 'caption'])
data = annotations['caption']
image = annotations['image']


for idx,i in enumerate(tqdm(data)):
    # if idx == 5:
    #     break

    # caption = i['caption']
    i.lower()
    doc = nlp(i)
    for chunk in doc.noun_chunks:
        # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
        if ' and ' in str(chunk):
            l = re.split(r' and ', str(chunk))

            for i in l:
                doc1 = nlp(i)
                item = ''
                for w in doc1:
                    if((w.tag_ == 'NN' or w.tag_== 'NNS' or w.tag_== 'NNPS' or w.tag_== 'NNP')\
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
                    entity.append(item.lower())
        else:
            item = ''
            for w in chunk:
                if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
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
                entity.append(item.lower())

count = Counter(entity)
count = count.most_common()
count = dict(count)
res = list(set(entity))
# print(res)
# print(count)

if plural:
    with open("/data1/yangzhenbang_new/datasets/blip_caption/vocab_large.json", 'w', encoding='utf-8') as fw:
        json.dump(res, fw, indent=4)
    print("vocab")

    with open("/data1/yangzhenbang_new/datasets/blip_caption/vocab_large_count.json", 'w',
              encoding='utf-8') as fw:
        json.dump(count, fw, indent=4)
    print("vocab_count")

else:
    with open("/data1/yangzhenbang_new/datasets/blip_caption/vocab_large_noplural.json", 'w',
              encoding='utf-8') as fw:
        json.dump(res, fw, indent=4)
    print("vocab_noplural")

    with open("/data1/yangzhenbang_new/datasets/blip_caption/vocab_large_count_noplural.json", 'w',
              encoding='utf-8') as fw:
        json.dump(count, fw, indent=4)
    print("vocab_noplural_count")

