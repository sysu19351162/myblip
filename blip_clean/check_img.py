import os
import json
from tqdm import tqdm

ann = json.load(open('/data2/datasets/cc3m/cc3m.json','r'))
ann = ann.values()

result = {}

for i in ann:
    if os.access('/data2/datasets/cc3m/images/'+i['img_name'], os.F_OK):
        result[i['img_name']]=i

with open('/data2/yangzhenbang/dataset/cc3m_new.json','wt') as file_obj:
  json.dump(result, file_obj)