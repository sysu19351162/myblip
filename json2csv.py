import json
import pandas as pd
json_path='/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/vg_example_visual_concept.json'
csv_path='/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/vg_example_visual_concept.csv'
# df = pd.read_csv("json_path")
data = json.load(open(json_path,'r'))
# print(type(data))
# r = list(data)
imgs = data.keys()
entitys = data.values()
vcs = []
for i in entitys:
    vc = []
    for j in i:
        item = j.keys()
        # print(type(list(item)[0]))
        vc.append(list(item)[0])
    vcs.append(vc)
# df = pd.read_json(open(json_path,'r'),orient='records')
# sort_knowlegde = []
df = pd.DataFrame()
df['images'] = imgs
df['vcs'] = vcs

# for img_file in df['dir']
#     sort_knowledge = data['img_file']
# df['knowledge'] = sort_knowledge
df.to_csv(csv_path, index=None)
print(df)