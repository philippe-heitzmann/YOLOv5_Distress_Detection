#data manipulation
import numpy as np
import pandas as pd

#file processing / requests
from pathlib import Path
from tqdm import tqdm
import json
import urllib

#computer vision
import cv2
import torch
import torchvision

#ML 
from sklearn.model_selection import train_test_split


#display
from IPython.display import display
from pylab import rcParams

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import PIL.Image as Image


import time 
from os import walk

def tdec(func):
    def inner(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print('{} sec to complete'.format(np.round(time.time() - start_time, 1)))
        return result
    return inner 


@tdec
def get_files_in_dir(path = './data/train/Japan/labels', show_folders = True, **kwargs):
    'Function to return all files in a directory with option of filtering for specific filetype extension'
    
    files = next(walk(path), (None, None, []))[2]
    if 'extension' in kwargs:
        files = [f for f in files if f.endswith(kwargs['extension'])]
    
    if show_folders:
        folders = next(walk(path), (None, None, []))[1]
        return files, folders, path
    return files, path


@tdec
def check_img_labels_match(imgspath, labelspath, imgsextension, labelsextension, **kwargs):
    labels_files, labels_folders, labels_path = get_files_in_dir(path = labelspath, extension = labelsextension, **kwargs)
    images_files, images_folders, images_path = get_files_in_dir(path = imgspath, extension = imgsextension, **kwargs)
    imageids, labelids = [], []
    for imageid, labelid in zip(images_files, labels_files):
        imageids.append(imageid.split('.')[0])
        labelids.append(labelid.split('.')[0])
    duplicate_images = set([x for x in imageids if imageids.count(x) > 1])
    duplicate_labels = set([x for x in labelids if labelids.count(x) > 1])
    if len(duplicate_images) > 0: print(duplicate_images, ' duplicate image IDs in images folder')
    else: print('No duplicate image IDs')
    if len(duplicate_labels) > 0: print(duplicate_labels, ' duplicate label IDs in images folder')
    else: print('No duplicate label IDs')
    
    images_not_in_labels = list(set(imageids) - set(labelids))
    labels_not_in_images = list(set(labelids) - set(imageids))
    if len(images_not_in_labels) > 0: print(images_not_in_labels, ' image IDs not found in labels folder')
    else: print('All specified image IDs found in labels folder')
    if len(labels_not_in_images) > 0: print(labels_not_in_images, ' label IDs not found in images folder')
    else: print('All specified image IDs found in labels folder')
    
    return images_not_in_labels, labels_not_in_images

            