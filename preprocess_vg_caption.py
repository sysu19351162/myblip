import json
from tqdm import tqdm

# result_f = '/mnt/data/yangzhenbang/BLIP/annotation/vg_caption.json'
result_f = '/mnt/data/yangzhenbang/BLIP/annotation/coco_karpathy_train.json'
with open(result_f) as f:
    result = json.load(f,strict=False)
d = {}
for i in tqdm(result):
    image = i['image']
    # head,seq,tail = image.partition('image')
    # i['image'] = '/mnt/data/linbingqian/beifen/lbq/cv02/data2/all/pkt/data/VG_100K'+tail
    i['image'] = '/mnt/data/datasets/coco/coco2014/' + image
    i['img_name'] = i['image']
    d.update(i)

# print(result)
#
# print(type(result))
with open("/mnt/data/yangzhenbang/BLIP/annotation/coco_karpathy_train_pretrain.json", 'w', encoding='utf-8') as fw:
        json.dump(result, fw, indent=4)

# result_f = '/mnt/data/yangzhenbang/BLIP/annotation/vg_caption.json'
# # result_f = '/data1/yangzhenbang_new/blip/BLIP/annotation/coco_karpathy_train.json'
# with open(result_f) as f:
#     result = json.load(f,strict=False)
# d = {}
# for i in tqdm(result):
#     # image = i['image']
#     # head,seq,tail = image.partition('image')
#     # i['img_name'] = '/data1/yangzhenbang_new/datasets/VisualGenome/image'+tail
#     # i['image'] = '/data1/yangzhenbang_new/datasets/coco/' + image
#     i['img_name'] = i['image']
#     d.update(i)
#
# # print(result)
# #
# # print(type(result))
# with open("/mnt/data/yangzhenbang/BLIP/annotation/vg_caption.json", 'w',
#             encoding='utf-8') as fw:
#     json.dump(result, fw, indent=4)