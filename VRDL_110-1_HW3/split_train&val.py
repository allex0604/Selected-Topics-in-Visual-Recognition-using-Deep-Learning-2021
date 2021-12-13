import os
import json

json_list = os.listdir("annotations")

# 记录json文件
annotations_list = []

dict = {}
image = []
annotations = []
total_img = 0
# 将所有的json文件读入
for file in json_list:
    if (file.endswith(".json")):
        annotations_list.append(file)


for anno_name in annotations_list:
    anno = json.loads(open("annotations/"+anno_name,encoding='utf-8').read())
    for img in anno["images"]:
        img["id"] = int(img["id"])
        # 处理完的img存入image列表
        image.append(img)
        break
    for a in anno['annotations']:
        # 计算对应的图片编号
        a['id'] = total_img
        a['category_id'] = 1
        # 处理完的annotation存入annotations列表
        annotations.append(a)
        total_img += 1

# 将字典对应的部分放入
dict['images']=image
dict['annotations']=annotations
# 类别直接写
dict['categories']=[{
    "id": 1,
    "name": "Nuclei",
    "supercategory": "Nuclei"
  }]
json.dump(dict, open('instances_train2017.json', 'w'), indent=4, ensure_ascii=True)

json_list = os.listdir("annotations_val")

# 记录json文件
annotations_list = []

dict = {}
image = []
annotations = []
total_img = 0
# 将所有的json文件读入
for file in json_list:
    if (file.endswith(".json")):
        annotations_list.append(file)


for anno_name in annotations_list:
    img_cnt = 0
    anno = json.loads(open("annotations_val/"+anno_name,encoding='utf-8').read())
    for img in anno["images"]:
        img["id"] = int(img["id"])
        # 处理完的img存入image列表
        image.append(img)
        break
    for a in anno['annotations']:
        # 计算对应的图片编号
        a['id'] = total_img
        a['category_id'] = 1
        # 处理完的annotation存入annotations列表
        annotations.append(a)
        total_img += 1

# 将字典对应的部分放入
dict['images']=image
dict['annotations']=annotations
# 类别直接写
dict['categories']=[{
    "id": 1,
    "name": "Nuclei",
    "supercategory": "Nuclei"
  }]
json.dump(dict, open('instances_val2017.json', 'w'), indent=4, ensure_ascii=True)

