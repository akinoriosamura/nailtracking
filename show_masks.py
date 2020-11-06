# -*- coding: utf-8 -*-

# import the necessary packages
# from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import os
import cv2
import find_finger as ff


def save_sample(img, mask, tag):
    try:
        dst = img + mask
        dst = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    except:
        import pdb; pdb.set_trace()
    cv2.imwrite("./sample_masked_" + tag + ".jpg", dst)


if __name__ == '__main__':
            dataset = 'nails_augment'
            mask_files = os.listdir(os.path.join(dataset, 'mask'))
            raw_files = os.listdir(os.path.join(dataset, 'raw'))
            # find intersection of two lists
            files = list(set(raw_files).intersection(mask_files))
            for f in files:
                image_p = os.path.join(dataset, 'raw/'+f)
                mask_p = os.path.join(dataset, 'mask/'+f)
                image = cv2.imread(image_p)
                mask = cv2.imread(mask_p)
                save_sample(image, mask, "1")
                import pdb; pdb.set_trace()