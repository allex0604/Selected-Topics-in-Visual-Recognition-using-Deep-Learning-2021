import json


def get_coco_test():
    json_fp = open('dataset\\test_img_ids.json', 'r')
    data = json.load(json_fp)

    coco_test = {
      "images": data,
      "annotations": [],
      "categories": [{
        "id": 1,
        "name": "Nuclei",
        "supercategory": "Nuclei"}]
    }

    return coco_test


outfile = get_coco_test()
json.dump(outfile,
          open("instances_test2017.json", 'w'),
          indent=4, ensure_ascii=True)
