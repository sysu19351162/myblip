import json
from tqdm import tqdm

result_f = '/mnt/data/yangzhenbang/BLIP/annotation/vg_qa.json'
with open(result_f) as f:
    result = json.load(f,strict=False)
# d = {}
for i in tqdm(result):
    image = i['image']
    head,seq,tail = image.partition('/')
    i['image'] = tail
    # i['image'] = '/data1/yangzhenbang_new/datasets/coco/' + image
    # d.update(i)

# print(result)
#
# print(type(result))
with open("/mnt/data/yangzhenbang/BLIP/annotation/vg_qa.json", 'w',
            encoding='utf-8') as fw:
    json.dump(result, fw, indent=4)