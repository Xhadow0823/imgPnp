"""coco 檔案轉yolo 檔案工具。
將目標影像與目標coco 格式json 放在dataset 中，執行本檔案。
"""
import json
from typing import List
import os

class Image:
    def __init__(self, image_dict: dict):
        for k, v in image_dict.items():
            setattr(self, k, v)
    id: int
    width: int
    height: int
    file_name: str

class Annotation:
    def __init__(self, image_dict: dict):
        for k, v in image_dict.items():
            setattr(self, k, v)
    image_id: int
    id: int
    segmentation: list

def load_label_file(filename: str) -> tuple[dict[int, Image], List[Annotation]]:
    coco = {}
    with open(filename, "rt") as f:
        coco = json.load(f)
    image_list = coco['images']
        #     {
        #     "id": 1,
        #     "width": 4032,
        #     "height": 3024,
        #     "file_name": "20230110-20.jpg"
        # },
    annotation_list = coco['annotations']
        #     {
        #     "id": 0,
        #     "iscrowd": 0,
        #     "image_id": 1,
        #     "segmentation": [
        #         [
        #             3188.659793814432,
        #             1597.731958762886,
        #             3693.4020618556688,
        #             1707.2164948453603,
        #             3687.8350515463903,
        #             2104.3298969072157,
        #             3188.659793814432,
        #             1781.4432989690715
        #         ]
        #     ],
        #     "bbox": [
        #         3188.659793814432,
        #         1597.731958762886,
        #         504.7422680412369,
        #         506.59793814432965
        #     ],
        #     "area": 146376.78818152845
        # },
    # images = list(map(Image, image_list))
    image_dict = {}
    for d in image_list:
        img = Image(d)
        image_dict[img.id] = img
    annotations = list(map(Annotation, annotation_list))
    return image_dict, annotations
    
def gen_label_file(img: Image, segs: list):
    line = '0'
    assert(len(segs[0])%2 == 0)
    for idx, cord in enumerate(segs[0]):
        ratio = cord/(img.height if idx%2 else img.width)
        assert(ratio <= 1)
        assert(ratio >= 0)
        line += f" {ratio:.6}"
    line+'\n'
    label_file_name = os.path.splitext(img.file_name)[0] + '.txt'
    label_file_path = os.path.join("dataset", label_file_name)
    with open(label_file_path, "wt") as f:
        f.write(line)

def transform(target_filename: str = "labels_my-project-name_2023-10-06-11-25-21.json"):
    target_path = os.path.join("dataset", target_filename)
    images, annotations = load_label_file(target_path)
    for anno in annotations:
        img = images[anno.image_id]
        gen_label_file(img, anno.segmentation)

def list_all_file(file_list):
    for idx, name in enumerate(file_list):
        print(f"[{idx:2}] | {name}")

if __name__ == "__main__":
    potential_json = list(filter(lambda x: x.endswith('.json'), next(os.walk("./dataset/"))[2]))
    target = None
    list_all_file(potential_json)
    if len(potential_json) <= 1:
        target = potential_json[0]
    else:
        target = potential_json[int(input("which json file ? (input the index)"))]
    transform(target)