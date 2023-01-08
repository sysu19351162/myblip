import spacy
from tqdm import tqdm
import json
import re
import pandas as pd

###########data format###########
#idx, entity, caption, image_id



#####读取数据###########
print('Visual Genome')
path='/mnt/data/yangzhenbang/BLIP/annotation/vg_caption.json'

with open(path,'r',encoding='utf8')as fp:
    json_data = json.load(fp)



nlp = spacy.load("en_core_web_sm")

index = []
entities = []
idx = 0
for _,i in enumerate(tqdm(json_data)):
    # if idx == 5:
    #     break
    # print(i)
    entity = []
    captions = i['caption']
    image_id = i['image']
    if type(captions) == str:
        caption = captions.lower()

        doc = nlp(caption)
        for chunk in doc.noun_chunks:
            # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
            if ' and ' in str(chunk):
                list = re.split(r' and ', str(chunk))

                for i in list:
                    doc1 = nlp(i)
                    item = ''
                    for w in doc1:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                and w.text != 'the' and w.text != 'a'):
                            if item == '':
                                item = w.text
                            else:
                                item = item + ' ' + w.text

                    if item != '' and item != ' ':
                        entity.append(item)
            else:
                item = ''
                for w in chunk:
                    if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                            and w.text != 'the' and w.text != 'a'):
                        if item == '':
                            item = w.text
                        else:
                            item = item + ' ' + w.text
                if item != '' and item != ' ':
                    entity.append(item)
        # for w in doc:
        #     if(w.tag_ == 'NN' or w.tag_ =='NNS'):
        #         entity.append(w.text)
        # entities.append([['idx',idx],['entity',entity],['caption',i],['image_id',image_id]]])
        # index.append(idx)
        # entities.append({'entity': entity, 'caption': i, 'image_id': image_id})
        entities.append({'idx': idx, 'entity': entity, 'caption': caption, 'image_id': image_id})
        idx = idx + 1
    else:
        for caption in captions:
            caption = caption.lower()
            doc = nlp(caption)
            for chunk in doc.noun_chunks:
                # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
                if ' and ' in str(chunk):
                    list = re.split(r' and ', str(chunk))

                    for i in list:
                        doc1 = nlp(i)
                        item = ''
                        for w in doc1:
                            if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                    and w.text != 'the' and w.text != 'a'):
                                if item == '':
                                    item = w.text
                                else:
                                    item = item + ' ' + w.text

                        if item != '' and item != ' ':
                            entity.append(item)
                else:
                    item = ''
                    for w in chunk:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                and w.text != 'the' and w.text != 'a'):
                            if item == '':
                                item = w.text
                            else:
                                item = item + ' ' + w.text
                    if item != '' and item != ' ':
                        entity.append(item)
            # for w in doc:
            #     if(w.tag_ == 'NN' or w.tag_ =='NNS'):
            #         entity.append(w.text)
            # entities.append([['idx',idx],['entity',entity],['caption',i],['image_id',image_id]]])
            # index.append(idx)
            # entities.append({'entity': entity, 'caption': i, 'image_id': image_id})
            entities.append({'idx': idx, 'entity': entity, 'caption': caption, 'image_id': image_id})
            idx = idx + 1


# print(entities)
# print(entities)
# res = dict(zip(index,entities))

with open("/mnt/data/yangzhenbang/datasets/blip_caption/vg_entity.json", 'w', encoding='utf-8') as fw:
    json.dump(entities, fw, indent=4)
print("vg_entity")