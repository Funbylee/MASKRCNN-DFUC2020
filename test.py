# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
#
#
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
#
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster.

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.color
import skimage.io
import skimage.transform
import csv

# Root directory of the project

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import json
from collections import defaultdict
import itertools
import mask as maskUtils

ROOT_DIR = os.path.abspath('./')
DATA_DIR = ROOT_DIR + '/dfuc/'


# ## Configurations

# ## Notebook Preferences
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class WoundConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "wounds"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax



class DFUCDataset(utils.Dataset):

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if
                                                   ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def load_dfuc(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        #
        # if auto_download is True:
        #     self.auto_download(dataset_dir, subset, year)

        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        jsonfile = dataset_dir + 'annotations/instances_' + subset + '2017.json'

        if not jsonfile == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(jsonfile, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            print('creating index...')
            anns, cats, imgs = {}, {}, {}
            imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
            if 'annotations' in self.dataset:
                for ann in self.dataset['annotations']:
                    imgToAnns[ann['image_id']].append(ann)
                    anns[ann['id']] = ann

            if 'images' in self.dataset:
                for img in self.dataset['images']:
                    imgs[img['id']] = img

            if 'categories' in self.dataset:
                for cat in self.dataset['categories']:
                    cats[cat['id']] = cat

            if 'annotations' in self.dataset and 'categories' in self.dataset:
                for ann in self.dataset['annotations']:
                    catToImgs[ann['category_id']].append(ann['image_id'])

            print('index created!')

            # create class members
            self.anns = anns
            self.imgToAnns = imgToAnns
            self.catToImgs = catToImgs
            self.imgs = imgs
            self.cats = cats

        # coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        # if subset == "minival" or subset == "valminusminival":
        #     subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, '2017')

        # Load all classes or a subset?

        if not class_ids:
            # All classes
            class_ids = 1

        # All images or a subset?
        if class_ids:
            image_ids = []

            image_ids.extend(list(self.getImgIds(catIds=1)))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(self.imgs.keys())

        # Add classes
        for i in [class_ids]:
            self.add_class("coco", i, self.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, self.imgs[i]['file_name']),
                width=self.imgs[i]["width"],
                height=self.imgs[i]["height"],
                annotations=self.loadAnns(self.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        # if image_info["source"] != "coco":
        #     return super(DFUCDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue

                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(DFUCDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(DFUCDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


# ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# get_ipython().magic('matplotlib inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

TEST_DIR = os.path.join(ROOT_DIR,'DFU2020_Testing_Release/')#testimage/')

#
# config = WoundConfig()
# config.display()
#

# Validation dataset
dataset_val = DFUCDataset()
dataset_val.load_dfuc(DATA_DIR, subset='val')
dataset_val.prepare()

#image_name_all = dataset_val.dataset['images'][1]['file_name'][0:-4]


class InferenceConfig(WoundConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


f = open('preresult.csv','w',encoding='utf-8',newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(["filename","xmin","ymin","xmax","ymax","score"])
for filename in os.listdir(TEST_DIR):
    #filename = "00831.jpg"
    #image = dataset_val.load_image(image_id)
    # Load image and ground truth data

    image = skimage.io.imread(TEST_DIR + filename)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]

    # image = np.array(image)
    # H = image.shape[0]
    # W = image.shape[1]
    # mh = np.arange(1, H + 1, 1)
    # mw = np.arange(1, W + 1, 1)
    # hx, wy = np.meshgrid(mw, mh)
    # hw = np.abs(hx - W / 2) + np.abs(wy - H / 2)
    # maxh = np.max(np.max(hw))
    # hw = np.array(hw * 255 / maxh, dtype='uint8')
    #
    # hw = np.expand_dims(hw, axis=-1)
    # image = np.append(image, hw, axis=-1)

    image_name_all = filename[0:-4]
    #image_name_p = image_name_all
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    pre_boxes = list(r["rois"])
    scores = list(r['scores'])
    # si = 0
    # # pre_boxes_r = []
    # # score_r = []
    # if len(pre_boxes) > 1:
    #     for box in pre_boxes:
    #         lab = 0
    #         blab = 1
    #         xmin = box[1]
    #         ymin = box[0]
    #         xmax = box[3]
    #         ymax = box[2]
    #
    #         while blab == 1:
    #             if si >= len(pre_boxes) - 1:
    #                 # pre_boxes_r.append([ymin, xmin, ymax, xmax])
    #                 # score_r.append(scores[si])
    #                 blab = 0
    #             else:
    #                 for ri in range(si+1, len(pre_boxes), 1):
    #                     xmin1 = pre_boxes[ri][1]
    #                     ymin1 = pre_boxes[ri][0]
    #                     xmax1 = pre_boxes[ri][3]
    #                     ymax1 = pre_boxes[ri][2]
    #
    #                     if (xmin1 - xmin) * (xmin1 - xmax) <= 0 and (ymin1 - ymin) * (ymin1 - ymax) <= 0:
    #                         lab = 1
    #                     if (xmin1 - xmin) * (xmin1 - xmax) <= 0 and (ymax1 - ymin) * (ymax1 - ymax) <= 0:
    #                         lab = 1
    #                     if (xmax1 - xmin) * (xmax1 - xmax) <= 0 and (ymax1 - ymin) * (ymax1 - ymax) <= 0:
    #                         lab = 1
    #                     if (xmax1 - xmin) * (xmax1 - xmax) <= 0 and (ymin1 - ymin) * (ymin1 - ymax) <= 0:
    #                         lab = 1
    #                     if lab == 1:
    #                         xmin = min(xmin, xmin1)
    #                         xmax = max(xmax, xmax1)
    #                         ymin = min(ymin, ymin1)
    #                         ymax = max(ymax, ymax1)
    #                         pre_boxes[si] = [ymin, xmin, ymax, xmax]
    #                         scores[si] = max(scores[si],scores[ri])
    #                         lab = 0
    #
    #                         del(pre_boxes[ri])
    #                         del(scores[ri])
    #                         blab = 1
    #                         if ri >= len(pre_boxes) - 1:
    #                             break
    #
    #                     else:
    #                         blab = 0
    #
    #                 # pre_boxes_r.append([ymin,xmin,ymax,xmax])
    #                 # score_r.append(scores[si])
    #                 si += 1
    # # else:
    # #     pre_boxes_r = pre_boxes
    # #     score_r = scores
    #
    pre_boxes = np.array(pre_boxes)
    scores = np.array(scores)


    si = 0
    for box in pre_boxes:
        csv_writer.writerow([str(filename), box[1], box[0], box[3], box[2], scores[si]])
        si += 1
    print(filename)


    visualize.display_instances(image, pre_boxes, r['masks'], r['class_ids'],
                                dataset_val.class_names, scores, figsize=(5.12, 5.12), image_name=image_name_all +'re')
f.close()



# In[ ]:



