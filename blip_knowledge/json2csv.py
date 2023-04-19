import json
import pandas as pd
json_path='/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/vg_example_visual_concept.json'
csv_path='/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/vg_example_visual_concept.csv'
# df = pd.read_csv("json_path")
data = json.load(open(json_path,'r'))
df = pd.read_csv(data,orient='records')
# sort_knowlegde = []
# for img_file in df['dir']
#     sort_knowledge = data['img_file']
# df['knowledge'] = sort_knowledge
df.to_csv(csv_path, index=None)