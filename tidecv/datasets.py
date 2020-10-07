from .data import Data
from . import functions as f

import zipfile
from pathlib import Path
from appdirs import user_data_dir
import urllib.request
from collections import defaultdict
import shutil
import json
import os
from .kittidataset.kitti_dataset import KittiDataset
import math

from .kitti_object_eval_python import kitti_common as kitti
# from eval import get_official_eval_result, get_coco_eval_result
def _read_imageset_file_kitti(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

# det_path = "/path/to/your_result_folder"
# dt_annos = kitti.get_label_annos(det_path)
# gt_path = "/path/to/your_gt_label_folder"
# gt_split_file = "/path/to/val.txt"
# val_image_ids = _read_imageset_file_kitti(gt_split_file)
# gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
# print(get_official_eval_result(gt_annos, dt_annos, 0)) # 6s in my computer
# print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer



def sigmoid(score: float) -> float:
    return 1. / (1 + math.exp(-score))


def default_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def get_tide_path():
    if 'TIDE_PATH' in os.environ:
        tide_path = os.environ['TIDE_PATH']
    else:
        tide_path = user_data_dir('tidecv', appauthor=False)

    if not os.path.exists(tide_path):
        os.makedirs(tide_path)

    return tide_path


def download_annotations(name: str, url: str, force_download: bool = False) -> str:
    tide_path = get_tide_path()
    candidate_path = os.path.join(tide_path, name)
    finished_file_path = os.path.join(candidate_path, '_finished')
    zip_file_path = os.path.join(candidate_path, '_tmp.zip')

    # Check if the file has already been downloaded
    # If there isn't a file called _finished, that means we didn't finish downloading last time, so try again
    already_downloaded = os.path.exists(candidate_path) and os.path.exists(finished_file_path)

    if not force_download and already_downloaded:
        return candidate_path
    else:
        print('{} annotations not found. Downloading...'.format(name))

        if os.path.exists(candidate_path):
            shutil.rmtree(candidate_path)
        os.makedirs(candidate_path)

        urllib.request.urlretrieve(url, zip_file_path)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(candidate_path)

        os.remove(zip_file_path)
        open(finished_file_path, 'a').close()  # Make an empty _finished file to mark that we were successful

        print('Successfully downloaded {} to "{}"'.format(name, candidate_path))
        return candidate_path

# def KITTI2(split_path:str = None, gt_path: str = None, name: str = None, use_box2d=False, vaild_class=['Car']) -> Data:
#     assert gt_path is not None, 'path should not be None.'
#     sample_idx_list = _read_imageset_file_kitti(split_path)
#     gt_annos = kitti.get_label_annos(gt_path, sample_idx_list)
#
#     if name is None: name = default_name(gt_path)
#
#     # Add everything from the coco json into our data structure
#     data = Data(name, max_dets=100)
#
#     for idx in range(len(sample_idx_list)):
#         sample_id = int(sample_idx_list[idx])
#         data.add_image(sample_id, sample_idx_list[idx])
#
#     type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
#     for cat in type_to_id:
#         data.add_class(type_to_id[cat], cat)
#
#     for i,image_idx in enumerate(sample_idx_list) :
#         sample_id = int(image_idx)
#         object_list = gt_annos[i]
#         level = kitti.add_difficulty_to_annos({"anno":object_list})
#         for i in range(len(object_list['name'])):
#             if object_list['name'][i] in vaild_class:
#                 cls_id = type_to_id(object_list['name'][i])
#                 ignore = level ==-1
#                 box =
#                 data._add(sample_id, , obj if not use_box2d else obj.box2d, None, ignore=obj.get_obj_level() == 4)
#
#     return data

def KITTI(split='val', path: str = None, name: str = None, use_box2d=False,vaild_class=['Car']) -> Data:
    assert path is not None, 'path should not be None.'

    dataset = KittiDataset(root_dir=path, split=split)
    if name is None: name = default_name(path)

    # Add everything from the coco json into our data structure
    data = Data(name, max_dets=100)

    for idx in range(len(dataset.image_idx_list)):
        sample_id = int(dataset.image_idx_list[idx])
        data.add_image(sample_id, dataset.image_idx_list[idx])

    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    for cat in type_to_id:
        data.add_class(type_to_id[cat], cat)

    for image_idx in dataset.image_idx_list:
        sample_id = int(image_idx)
        object_list = dataset.get_label(sample_id)
        for obj in object_list:
            if obj.cls_type in vaild_class:
                data._add(sample_id, obj.cls_id, obj if not use_box2d else obj.box2d, None, ignore=obj.get_obj_level() == 4)

    return data


def KITTIResult(split='val', path: str = None, name: str = None, pred_label_path: str = None, use_box2d=False) -> Data:
    """ Loads predictions from kitti resultv files. """
    assert path is not None, 'path should not be None.'
    dataset = KittiDataset(root_dir=path, split=split)
    dataset.label_dir = os.path.join(pred_label_path)
    if name is None: name = default_name(path)

    # Add everything from the coco json into our data structure
    data = Data(name, max_dets=100)

    for idx in range(len(dataset.image_idx_list)):
        sample_id = int(dataset.image_idx_list[idx])
        data.add_image(sample_id, dataset.image_idx_list[idx])

    for image_idx in dataset.image_idx_list:
        sample_id = int(image_idx)
        object_list = dataset.get_label(sample_id)
        for obj in object_list:
            score = sigmoid(obj.score)
            data.add_detection(image_id=sample_id, class_id=obj.cls_id, box=obj if not use_box2d else obj.box2d,
                               score=score, mask=None)
    return data


def COCO(path: str = None, name: str = None, year: int = 2017, ann_set: str = 'val',
         force_download: bool = False) -> Data:
    """
    Loads ground truth from a COCO-style annotation file.

    If path is not specified, this will download the COCO annotations for the year and ann_set specified.
    Valid years are 2014, 2017 and valid ann_sets are 'val' and 'train'.
    """
    if path is None:
        path = download_annotations(
            'COCO{}'.format(year),
            'http://images.cocodataset.org/annotations/annotations_trainval{}.zip'.format(year),
            force_download)

        path = os.path.join(path, 'annotations', 'instances_{}{}.json'.format(ann_set, year))

    if name is None: name = default_name(path)

    with open(path, 'r') as json_file:
        cocojson = json.load(json_file)

    images = cocojson['images']
    anns = cocojson['annotations']
    cats = cocojson['categories'] if 'categories' in cocojson else None

    # Add everything from the coco json into our data structure
    data = Data(name, max_dets=100)

    image_lookup = {}

    for idx, image in enumerate(images):
        image_lookup[image['id']] = image
        data.add_image(image['id'], image['file_name'])

    if cats is not None:
        for cat in cats:
            data.add_class(cat['id'], cat['name'])

    for ann in anns:
        image = ann['image_id']
        _class = ann['category_id']
        box = ann['bbox']
        mask = f.toRLE(ann['segmentation'], image_lookup[image]['width'], image_lookup[image]['height'])

        if ann['iscrowd']:
            data.add_ignore_region(image, _class, box, mask)
        else:
            data.add_ground_truth(image, _class, box, mask)

    return data


def COCOResult(path: str, name: str = None) -> Data:
    """ Loads predictions from a COCO-style results file. """
    if name is None: name = default_name(path)

    with open(path, 'r') as json_file:
        dets = json.load(json_file)

    data = Data(name)

    for det in dets:
        image = det['image_id']
        _cls = det['category_id']
        score = det['score']
        box = det['bbox'] if 'bbox' in det else None
        mask = det['segmentation'] if 'segmentation' in det else None

        data.add_detection(image, _cls, score, box, mask)

    return data


def LVIS(path: str = None, name: str = None, version_str: str = 'v1', force_download: bool = False) -> Data:
    """
    Load an LVIS-style dataset.
    The version string is used for downloading the dataset and should be one of the versions of LVIS (e.g., v0.5, v1).

    Note that LVIS evaulation is special, but we can emulate it by adding ignore regions.
    The detector isn't punished for predicted class that LVIS annotators haven't guarenteed are in
    the image (i.e., the sum of GT annotated classes in the image and those marked explicitly not
    in the image.) In order to emulate this behavior, add ignore region labels for every class not
    found to be in the image. This is not that inefficient because ignore regions are separate out
    during mAP calculation and error processing, so adding a bunch of them doesn't hurt.

    The LVIS AP numbers are slightly lower than what the LVIS API reports because of these workarounds.
    """
    if path is None:
        path = download_annotations(
            'LVIS{}'.format(version_str),
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_{}_val.json.zip'.format(version_str),
            force_download)

        path = os.path.join(path, 'lvis_{}_val.json'.format(version_str))

    if name is None: name = default_name(path)

    with open(path, 'r') as json_file:
        lvisjson = json.load(json_file)

    images = lvisjson['images']
    anns = lvisjson['annotations']
    cats = lvisjson['categories'] if 'categories' in lvisjson else None

    data = Data(name, max_dets=300)
    image_lookup = {}
    classes_in_img = defaultdict(lambda: set())

    for image in images:
        image_lookup[image['id']] = image
        data.add_image(image['id'], image['coco_url'])  # LVIS has no image names, only coco urls

        # Negative categories are guarenteed by the annotators to not be in the image.
        # Thus we should care about them during evaluation.
        for cat_id in image['neg_category_ids']:
            classes_in_img[image['id']].add(cat_id)

    if cats is not None:
        for cat in cats:
            data.add_class(cat['id'], cat['synset'])

    for ann in anns:
        image = ann['image_id']
        _class = ann['category_id']
        box = ann['bbox']
        mask = f.toRLE(ann['segmentation'], image_lookup[image]['width'], image_lookup[image]['height'])

        data.add_ground_truth(image, _class, box, mask)

        # There's an annotation for this class, so we should consider the class for evaluation.
        classes_in_img[image].add(_class)

    all_classes = set(data.classes.keys())

    # LVIS doesn't penalize the detector for detecting classes that the annotators haven't guarenteed to be in/out of
    # the image. Here we simulate that property by adding ignore regions for all such classes.
    for image in images:
        ignored_classes = all_classes.difference(classes_in_img[image['id']])

        # LVIS doesn't penalize the detector for mistakes made on classes explicitly marked as not exhaustively annoted
        # We can emulate this by adding ignore regions for every category listed, so add them to the ignored classes.
        ignored_classes.update(set(image['not_exhaustive_category_ids']))

        for _cls in ignored_classes:
            data.add_ignore_region(image['id'], _cls)

    return data


def LVISResult(path: str, name: str = None) -> Data:
    """ Loads predictions from a LVIS-style results file. Note that this is the same as a COCO-style results file. """
    return COCOResult(path, name)


def Pascal(path: str = None, name: str = None, year: int = 2007, ann_set: str = 'val',
           force_download: bool = False) -> Data:
    """
    Loads the Pascal VOC 2007 or 2012 data from a COCO json.

    Valid years are 2007 and 2012, and valid ann_sets are 'train' and 'val'.
    """
    if path is None:
        path = download_annotations(
            'Pascal',
            'https://s3.amazonaws.com/images.cocodataset.org/external/external_PASCAL_VOC.zip',
            force_download)

        path = os.path.join(path, 'PASCAL_VOC', 'pascal_{}{}.json'.format(ann_set, year))

    return COCO(path, name)


def Cityscapes(path: str, name: str = None):
    """
    Loads the fine cityscapes annotations as instance segmentation masks, and also generates bounding boxes for them.

    Note that we can't automatically download Cityscapes because it requires registration and an agreement to the ToS.
    You can get cityscapes here: https://www.cityscapes-dataset.com/

    Path should be to gtFine/<ann_set>. E.g., <path_to_cityscapes>/gtFine/val.
    """
    if name is None: name = default_name(path)
    data = Data(name)

    instance_classes = {
        'person': 24,
        'rider': 25,
        'car': 26,
        'truck': 27,
        'train': 31,
        'motorcycle': 32,
        'bicycle': 33,
        'bus': 28,
        'caravan': 29,
        'trailer': 30,
    }

    ignored_classes = set([29, 30])

    for class_name, class_id in instance_classes.items():
        data.add_class(class_id, class_name)

    for ann in Path(path).glob('*/*.json'):
        with open(ann) as json_file:
            ann_json = json.load(json_file)

        # Note: a string for an image ID is okay
        image_id = os.path.basename(ann).replace('_gtFine_polygons.json', '')
        objs = ann_json['objects']

        data.add_image(image_id, image_id)  # The id in this case is just the name of the image

        # Caravan and Trailer should be ignored from all evaluation
        for _cls in ignored_classes:
            data.add_ignore_region(image_id, _cls)

        for obj in objs:
            class_label = obj['label']
            is_crowd = False

            # Cityscapes labelers can label objects without defined boundaries as 'group'. In COCO-land this would be
            # a crowd annotation. So in this case, let's make it an ignore region.
            if class_label.endswith('group'):
                is_crowd = True
                class_label = class_label[:-5]  # Remove the group at the end

            # We are only considering instance classes
            if not class_label in instance_classes:
                continue

            class_id = instance_classes[class_label]

            # If the class is not used in evaluation, don't include it
            if class_id in ignored_classes:
                continue

            poly = [sum(obj['polygon'],
                        [])]  # Converts a list of points to a list of lists of ints, where every 2 ints represents a point
            box = f.polyToBox(poly)

            if is_crowd:
                data.add_ignore_region(image_id, class_id, box, poly)
            else:
                data.add_ground_truth(image_id, class_id, box, poly)

    return data
