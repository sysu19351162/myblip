import json
from tqdm import tqdm
# def collect_entities(visual_concept,text_entities_path):

def collect_visual_concept(train_p,val_p,test_p):

    with open(train_p) as f:
        train = json.load(f)

    with open(val_p) as f:
        val = json.load(f)

    with open(test_p) as f:
        test = json.load(f)

    for k,v in val.items():
        train[k] = v

    for k,v in test.items():
        train[k] = v


    return train

if __name__ == "__main__":
    annotation = json.load(open('/mnt/data/yangzhenbang/datasets/blip_caption/data_coco_vg_flickr/kg_data/vg_has_knowledge.json', 'r'))
    flag = -1
    target = ["/mnt/data/linbingqian/beifen/lbq/cv02/data2/all/pkt/data/VG_100K/1.jpg",
    "/mnt/data/linbingqian/beifen/lbq/cv02/data2/all/pkt/data/VG_100K/10.jpg",
    "/mnt/data/linbingqian/beifen/lbq/cv02/data2/all/pkt/data/VG_100K/1000.jpg"]
    l = []
    for img_id, ann in tqdm(enumerate(annotation)):
        caption = ann['caption']
        if type(caption) != str:
            print(img_id)
            print(caption)
            flag = 1
            break
    if flag == -1:
        print(len(annotation))



    ##############flickr
    # test_path = '/mnt/data/yangzhenbang/datasets/blip_caption/flickr_test_visual_concept.json'
    # val_path = '/mnt/data/yangzhenbang/datasets/blip_caption/flickr_val_visual_concept.json'
    # train_path = '/mnt/data/yangzhenbang/datasets/blip_caption/flickr_train_visual_concept.json'
    # text_entities_path = '/mnt/data/yangzhenbang/datasets/blip_caption/flickr_entity.json'
    #
    #
    # visual_concept = collect_visual_concept(train_path,val_path,test_path)
    #
    # with open("/mnt/data/yangzhenbang/datasets/blip_caption/flickr_visual_concept.json", 'w',
    #           encoding='utf-8') as fw:
    #     json.dump(visual_concept, fw, indent=4)
    # print("/mnt/data/yangzhenbang/datasets/blip_caption/flickr_visual_concept.json")


    # ############coco
    # test_path = '/mnt/data/yangzhenbang/datasets/blip_caption/coco_test_visual_concept.json'
    # val_path = '/mnt/data/yangzhenbang/datasets/blip_caption/coco_val_visual_concept.json'
    # train_path = '/mnt/data/yangzhenbang/datasets/blip_caption/coco_train_visual_concept.json'
    #
    # visual_concept = collect_visual_concept(train_path, val_path, test_path)
    #
    # with open("/mnt/data/yangzhenbang/datasets/blip_caption/coco_visual_concept.json", 'w',
    #           encoding='utf-8') as fw:
    #     json.dump(visual_concept, fw, indent=4)
    # print("/mnt/data/yangzhenbang/datasets/blip_caption/coco_visual_concept.json")