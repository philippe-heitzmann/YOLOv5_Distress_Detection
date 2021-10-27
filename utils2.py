

#data manipulation
import numpy as np
import pandas as pd

#file processing / requests
from pathlib import Path
from tqdm import tqdm
import json
import urllib
from xml.dom import minidom
import os
import glob
from os import walk


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
from matplotlib import rc, patches, patheffects
import PIL.Image as Image

#styling
from typing import List, Dict

import time 

def tdec(func):
    def inner(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print('{} sec to complete'.format(np.round(time.time() - start_time, 1)))
        return result
    return inner 


@tdec
def get_files_in_dir(path = './data/train/Japan/labels', show_folders = False, **kwargs):
    'Function to return all files in a directory with option of filtering for specific filetype extension'
    
    files = next(walk(path), (None, None, []))[2]
    if 'extension' in kwargs:
        files = [f for f in files if f.endswith(kwargs['extension'])]
    
    if show_folders:
        folders = next(walk(path), (None, None, []))[1]
        return files, folders
    return files


@tdec
def check_img_labels_match(imgspath, labelspath, imgsextension, labelsextension, **kwargs):
    if 'show_folders' in kwargs and kwargs['show_folders']:
        labels_files, labels_folders = get_files_in_dir(path = labelspath, extension = labelsextension, **kwargs)
        images_files, images_folders = get_files_in_dir(path = imgspath, extension = imgsextension, **kwargs)
    else:
        labels_files = get_files_in_dir(path = labelspath, extension = labelsextension, **kwargs)
        images_files = get_files_in_dir(path = imgspath, extension = imgsextension, **kwargs)
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
    else: print('All specified label IDs found in images folder')
    
    return images_not_in_labels, labels_not_in_images






def convert_coordinates(size, box, normalize = True):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    if normalize:
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
    return (x,y,w,h)

# labelsdict,
@tdec
def convert_xml_to_yolo(path, labelsdict, extension = 'xml', xmlsdir = 'xmls', normalize = True):
    '''Inputs:
    extension: make sure to pass without any punctuation, i.e. India_0001.xml should be extension = 'xml' '''

    files = [f for f in next(walk(path), (None, None, []))[2] if f.endswith(extension)] 
    
    for file in files:
        xmldoc = minidom.parse(path + os.sep + file)
        idx = len(extension) + 1
        fname_out = (file[:-idx]+'.txt')
        outpath = path.replace(xmlsdir, 'labels', 1) + os.sep + fname_out

        with open(outpath, "w") as f:

            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                # get class label
                try:    
                    classid =  labelsdict[(item.getElementsByTagName('name')[0]).firstChild.data]
                except: classid = 0
                # get bbox coordinates
                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                if normalize:
                    bb = convert_coordinates((width,height), b, normalize = True)
                else: bb = convert_coordinates((width,height), b, normalize = False)
                f.write(classid + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')

        print ("wrote %s" % fname_out)
    return 'Done creating txts from xmls in {}'.format(path.replace(xmlsdir, 'labels', 1))


@tdec
def show_preds(dict_preds: Dict, labeldict, rootpath = './data/test2/Japan/images/images/'):
    '''Function to show bounding box predictions on a set of images'''
    
    for img, labels in dict_preds.items():
        img_path = rootpath + img
        img = cv2.imread(img_path)
        color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(color)
        plt.title('Image')
        fig, ax = plt.subplots(figsize = (6,9))

        for l in labels:
            indlabels = l.split(' ')
            l, x, y, w, h = labeldict[indlabels[0]], int(indlabels[1]), int(indlabels[2]), int(indlabels[3]), int(indlabels[4])
            ax.xaxis.tick_top()
            ax.add_patch(patches.Rectangle((x,y),w,h, fill=False, edgecolor='red', lw=2))
            ax.text(x,(y-50),str(l),verticalalignment='top', color='white',fontsize=10,
                    weight='bold').set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])
        print(img_path)
        ax.imshow(img)

    return 'Done Outputting Bounding Box Image Visualizations'
