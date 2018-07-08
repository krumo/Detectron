from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import h5py
import json
import os
import scipy.misc
import sys

import cityscapesscripts.evaluation.instances2dict_with_polygons as cs

import detectron.utils.segms as segms_util
import detectron.utils.boxes as bboxs_util


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--dataset', help="foggy cityscapes", default='foggy_cityscapes_instance_only', type=str)
    parser.add_argument(
        '--outdir', help="output dir for json files", default=None, type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted",
        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

# for Cityscapes
def getLabelID(self, instID):
    if (instID < 1000):
        return instID
    else:
        return int(instID / 1000)


def convert_foggy_cityscapes_instance_only(
        data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = [
        'leftImg8bit_foggy/val',
        'leftImg8bit_foggy/train',
        'leftImg8bit_foggy/test',
        # 'gtFine_train',
        # 'gtFine_test',

        # 'gtCoarse_train',
        # 'gtCoarse_val',
        # 'gtCoarse_train_extra'
    ]
    ann_dirs = [
        'gtFine/val',
        'gtFine/train',
        'gtFine/test',
        # 'gtFine_trainvaltest/gtFine/train',
        # 'gtFine_trainvaltest/gtFine/test',

        # 'gtCoarse/train',
        # 'gtCoarse/train_extra',
        # 'gtCoarse/val'
    ]
    json_name = 'instancesonly_filtered_%s.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = [
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
    ]
    for cls in category_instancesonly:
        category_dict[cls] = cat_id
        cat_id+=1

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        data_dirs = os.path.join(data_dir, data_set)
        ann_dir = os.path.join(data_dir, ann_dir)
        for root, _, files in os.walk(data_dirs):
            for filename in files:
                if filename.endswith('.png'):
                    ann_name = filename.split('_leftImg8bit')[0]+'_gtFine_polygons.json'
                    # print('ann_name: '+ann_name)
                    if len(images) % 50 == 0:
                        print("Processed %s images, %s annotations" % (
                            len(images), len(annotations)))
                    json_ann = json.load(open(os.path.join(ann_dir, filename.split('_')[0], ann_name)))
                    image = {}
                    image['id'] = img_id
                    img_id += 1

                    image['width'] = json_ann['imgWidth']
                    image['height'] = json_ann['imgHeight']
                    image['file_name'] = filename
                    image['seg_file_name'] = filename.split('_leftImg8bit')[0] + \
                        '_gtFine_instanceIds.png'
                    images.append(image)

                    fullname = os.path.join(ann_dir, filename.split('_')[0], image['seg_file_name'])
                    # print('seg_file_name '+image['seg_file_name'])
                    objects = cs.instances2dict_with_polygons(
                        [fullname], verbose=False)[fullname]

                    for object_cls in objects:
                        if object_cls not in category_instancesonly:
                            print('unknown class: '+object_cls)
                            continue  # skip non-instance categories

                        for obj in objects[object_cls]:
                            if obj['contours'] == []:
                                # print('Warning: empty contours.')
                                continue  # skip non-instance categories

                            len_p = [len(p) for p in obj['contours']]
                            if min(len_p) <= 4:
                                # print('Warning: invalid contours.')
                                continue  # skip non-instance categories

                            ann = {}
                            ann['id'] = ann_id
                            ann_id += 1
                            ann['image_id'] = image['id']
                            ann['segmentation'] = obj['contours']

                            if object_cls not in category_dict:
                                category_dict[object_cls] = cat_id
                                cat_id += 1
                            ann['category_id'] = category_dict[object_cls]
                            ann['iscrowd'] = 0
                            ann['area'] = obj['pixelCount']
                            ann['bbox'] = bboxs_util.xyxy_to_xywh(
                                segms_util.polys_to_boxes(
                                    [ann['segmentation']])).tolist()[0]

                            annotations.append(ann)

        ann_dict['images'] = images
        categories = [{"id": i+1, "name": name} for i, name in enumerate(category_instancesonly)]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print(categories)
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir, json_name % data_set.split('/')[1]), 'wb') as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "foggy_cityscapes_instance_only":
        convert_foggy_cityscapes_instance_only(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
