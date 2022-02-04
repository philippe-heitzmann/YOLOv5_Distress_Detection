

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

import PIL

#computer vision
import cv2
import torch
import torchvision
import torchvision.ops.boxes as bops

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
import typing
from typing import List, Dict

import time 

from matplotlib.pyplot import figure

from collections import defaultdict
import csv
import requests
import xml.etree.ElementTree as ET

#string manipulation
import re
import string
  
import sys
sys.path.append('/Users/Administrator/DS/IEEE-Big-Data-2020')
sys.path.append('/Users/phil0/DS/IEEE')


def tdec(func):
    def inner(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print('{} sec to complete {}'.format(np.round(time.time() - start_time, 1), func))
        return result
    return inner 


def show_im(path, **kwargs):
    img = cv2.imread(path)
    figure(figsize=(10, 10), dpi=80)
    color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    plt.imshow(color)
    
    
def get_lines(labelpath):
    with open(labelpath) as labelfile:
        lines = [line.rstrip() for line in labelfile.readlines()]
    return lines
    
    
@tdec
def get_files_in_dir(*extensions, path = './data/train/Japan/labels', show_folders = False, **kwargs):
    'Function to return all files in a directory with option of filtering for specific filetype extension'
    
    result = []
    for extension in list(extensions):
        files = next(walk(path), (None, None, []))[2]
        if 'fullpath' in kwargs and kwargs['fullpath']:
            files = [path + '/' + f for f in files if f.endswith(extension)]
            result.extend(files)
        else: 
            files = [f for f in files if f.endswith(extension)]
            result.extend(files)
    if show_folders:
        folders = next(walk(path), (None, None, []))[1]
        folders = [folder for folder in folders if folder[0] != '.']
        return result, folders
    return result


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


def convert_xml_to_yolo(path, labelsdict, extension = 'xml', xmlsdir = 'xmls', normalize = True):
    '''Inputs:
    extension: make sure to pass without any punctuation, i.e. India_0001.xml should be extension = 'xml' '''

    files = [f for f in next(walk(path), (None, None, []))[2] if f.endswith(extension)] 
    unrecognized_classes = []

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
                except: 
                    # print('Error')
                    # print('No entry for classID ', (item.getElementsByTagName('name')[0]).firstChild.data)
                    # unrecognized_classes.append((item.getElementsByTagName('name')[0]).firstChild.data)
                    # classid = 0
                    continue
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
    print('Done creating txts from xmls in {}'.format(path.replace(xmlsdir, 'labels', 1)))
    return unrecognized_classes


def get_xml_info(xmlpath):
    xmldoc = minidom.parse(xmlpath)
    filename = xmldoc.getElementsByTagName('filename')[0].firstChild.nodeValue
    itemlist = xmldoc.getElementsByTagName('object')
    size = xmldoc.getElementsByTagName('size')[0]
    width = int((size.getElementsByTagName('width')[0]).firstChild.data)
    height = int((size.getElementsByTagName('height')[0]).firstChild.data)
    boxes = []
    for item in itemlist:
        classid = item.getElementsByTagName('name')[0].firstChild.data
        # get bbox coordinates
        xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
        ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
        xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
        ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
        boxes.append([classid, xmin, ymin, xmax, ymax])
    return filename, width, height, boxes


def xml_to_csv(path, labelsdict):
    xml_list = []
    
    for xml_file in glob.glob(path + '/*.xml'):
        xmldoc = minidom.parse(xml_file)
        filename = xml_file.split('/')[-1].replace('xml','jpg')
        #xmldoc.getElementsByTagName('filename')[0].text
        itemlist = xmldoc.getElementsByTagName('object')
        size = xmldoc.getElementsByTagName('size')[0]
        width = int((size.getElementsByTagName('width')[0]).firstChild.data)
        height = int((size.getElementsByTagName('height')[0]).firstChild.data)
        for item in itemlist:
            try:    
                classid = item.getElementsByTagName('name')[0].firstChild.data
                if classid not in labelsdict:
                    continue
                classid = labelsdict[classid]
            except: 
                continue
            # get bbox coordinates
            xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
            ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
            xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
            ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
        
            value = (filename,
                     width,
                     height,
                     classid,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

@tdec
def jpg_to_csv(path):
    imgs = get_files_in_dir(path = path, extension = 'jpg', show_folders = False, fullpath = False)
    outlist = []
    for img in imgs:
        pil_image = PIL.Image.open(path + os.sep + img)
        imgwidth, imgheight = pil_image.size
        classid, xmin, ymin, xmax, ymax = 1,0,0,1,1
        value = (img, imgwidth, imgheight, classid, xmin, ymin, xmax, ymax)
        outlist.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    outdf = pd.DataFrame(outlist, columns=column_name)
    return outdf    

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
            ax.text(x,(y-20),str(l),verticalalignment='top', color='white',fontsize=10,
                    weight='bold').set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])
        print(img_path)
        ax.imshow(img)

    return 'Done Outputting Bounding Box Image Visualizations'

@tdec
def get_all_class_ids(path, extension, show_country = False):
    classes = set()
    classesdict = defaultdict(int)
    paths = get_files_in_dir(path = path, extension = extension, show_folders = False, fullpath = True)
    empty_images = 0
    for file in paths:
        if show_country:
            country = file.split('/')[-1].split('_')[0]
        if extension == 'txt':
            with open(file) as labelfile:
                lines = [line.rstrip() for line in labelfile.readlines()]
                if len(lines) == 0:
                    empty_images += 1
                for line in lines:
                    classid = line.split(' ')[0]
                    classes.add(classid)
                    if show_country:
                        keyn = country + '_' + str(classid)
                        classesdict[keyn] += 1
                        continue
                    classesdict[classid] += 1
        elif extension == 'xml':
            filename, width, height, boxes = get_xml_info(file)
            if len(boxes) == 0:
                empty_images += 1
            for box in boxes:
                classes.add(box[0])
                if show_country:
                    keyn = country + '_' + str(box[0])
                    classesdict[keyn] += 1
                    continue
                classesdict[classid] += 1
        
    return classes, classesdict, empty_images


def show_labels(imagepath, **kwargs):
    
    labelpath = imagepath.replace('images','labels',1)        
    labelpath = labelpath.replace('jpg','txt',1)

    pil_image = PIL.Image.open(imagepath)
    imgwidth, imgheight = pil_image.size
    fig, ax = plt.subplots(figsize = (9,9))
    #show the image
    ax.imshow(pil_image)
    with open(labelpath) as labelfile:
        #labelpath are .txt yolo annotation format label files in format <predicted object-class> <xcenter of bounding box (normalized by image width)> <ycenter of bounding box (normalized by image height)> <box width (normalized by image width)> <box height (normalized by image height)>
        #convert to PASCAL VOC format xmin, ymin, xmax, ymax
        lines = [line.rstrip() for line in labelfile.readlines()] 
        if len(lines) > 0:
            for label in lines:
                lb = label.split(' ')[0]
                if 'labelsdict' in kwargs and lb in kwargs['labelsdict']:
                    try:
                        lb = kwargs['labelsdict'][lb]
                    except: continue
                xcenter, ycenter, box_width, box_height = [float(i) for i in label.split(' ')[1:]] 
                xmin = (xcenter - box_width / 2) * imgwidth
                xmax = (xcenter + box_width / 2) * imgwidth
                ymin = (ycenter - box_height / 2) * imgheight
                ymax = (ycenter + box_height / 2) * imgheight
                xmin, ymin, xmax, ymax = [int(i) for i in [xmin, ymin, xmax, ymax]]
                box_width, box_height = int(np.round(box_width * imgwidth,0)), int(np.round(box_height * imgheight,0))

                ax.add_patch(patches.Rectangle((xmin,ymin),box_width, box_height, fill=False, edgecolor='red', lw=3))
                ax.text(xmin,(ymin-5),str(lb),verticalalignment='top', color='white',fontsize=15,
                                     weight='bold').set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])
            plt.show()


def wrapper_show_labels(imagedir, start = 0, end = 10, **kwargs):
    imagepaths = get_files_in_dir(path = imagedir, extension = 'jpg', **kwargs)
    
    for imagepath in imagepaths[start:end]:
        print(imagepath)
        #print(imagepath)
        show_labels(imagepath = imagepath, **kwargs)



def wrapper_get_iou(labelpath1, labelpath2):
    '''Function to get TP, FP, FN between set of prediction and ground truth txt files in YOLO format 
    Inputs:
    labelpath1 = path to ground truth label
    labelpath2 = path to prediction label'''
    classesdict = defaultdict(list)
    TP = 0; FP = 0; FN = 0; 
    with open(labelpath1) as labelfile1:
        #labelpath1 is path to ground truth file
        lines1 = [line.rstrip() for line in labelfile1]
        imagepath1 = labelpath1.replace('labels','images',1)        
        imagepath1 = imagepath1.replace('txt','jpg',1)
        pil_image1 = PIL.Image.open(imagepath1)
        imgwidth1, imgheight1 = pil_image1.size
        for label in lines1:
            lb = label.split(' ')[0]
            xcenter, ycenter, box_width, box_height = [float(i) for i in label.split(' ')[1:]] 
            xmin = (xcenter - box_width / 2) * imgwidth1
            xmax = (xcenter + box_width / 2) * imgwidth1
            ymin = (ycenter - box_height / 2) * imgheight1
            ymax = (ycenter + box_height / 2) * imgheight1
            xmin, ymin, xmax, ymax = [int(i) for i in [xmin, ymin, xmax, ymax]]
            classesdict[lb].append([xmin, ymin, xmax, ymax])
    try:
        #try except statement in case prediction labelfile does not exist in cases where model predicted no objects in image 
        with open(labelpath2) as labelfile2:
            #labelpath2 is path to prediction file
            lines2 = [line.rstrip() for line in labelfile2]
            imagepath2 = labelpath2.replace('labels/','',1)        
            imagepath2 = imagepath2.replace('txt','jpg',1)
            pil_image2 = PIL.Image.open(imagepath2)
            imgwidth2, imgheight2 = pil_image2.size
            for label in lines2:
                lb = label.split(' ')[0]
                if len(classesdict[lb]) == 0:
                    FP += 1
                    continue
                else:
                    xcenter, ycenter, box_width, box_height = [float(i) for i in label.split(' ')[1:]] 
                    xmin1 = (xcenter - box_width / 2) * imgwidth2
                    xmax1 = (xcenter + box_width / 2) * imgwidth2
                    ymin1 = (ycenter - box_height / 2) * imgheight2
                    ymax1 = (ycenter + box_height / 2) * imgheight2
                    xmin1, ymin1, xmax1, ymax1 = [int(i) for i in [xmin1, ymin1, xmax1, ymax1]]
                    for bb in classesdict[lb]:
                        xmin2, ymin2, xmax2, ymax2 = bb[0], bb[1], bb[2], bb[3]
                        box1 = torch.tensor([[xmin1, ymin1, xmax1, ymax1]], dtype=torch.float)
                        box2 = torch.tensor([[xmin2, ymin2, xmax2, ymax2]], dtype=torch.float)
                        iou = bops.box_iou(box1, box2).item()
                        if iou >= 0.5:
                            TP += 1
                            classesdict[lb].remove(bb)
        print('No Error')
    except:
        print('Error: Could not find labelpath2 ', labelpath2)
        for k, v in classesdict.items():
            FN += len(v)
        return TP, FP, FN 
    for k, v in classesdict.items():
        FN += len(v)
    return TP, FP, FN


def get_performance_metrics(gtbasepath, predbasepath, idxs):
    groundtruths = get_files_in_dir(path = gtbasepath, extension = 'txt', fullpath = False)
    TPe = 0; FPe = 0; FNe = 0;
    for gt in groundtruths:
        gtpath = gtbasepath + gt
        predpath = predbasepath + gt
        TP2, FP2, FN2 = wrapper_get_iou(labelpath1 = gtpath, labelpath2 = predpath)
        TPe += TP2
        FPe += FP2
        FNe += FN2
        for idx in idxs:
            predpath2 = predbasepath + gt.replace('.txt',f'''_aug{idx}.txt''',1)
            TP2, FP2, FN2 = wrapper_get_iou(labelpath1 = gtpath, labelpath2 = predpath2)
            TPe += TP2
            FPe += FP2
            FNe += FN2

    re = recall(TPe, FNe)
    pe = precision(TPe, FPe) 
    F1e = f1score(re, pe)
    return TPe, FPe, FNe, re, pe, F1e


@tdec
def aug_pipeline(*seqs, imagepaths, start = 0, end = 100000, **kwargs):
    #make sure to create /aug folder in imagespath folder to write augmented images
    
    imagepaths = get_files_in_dir(path = imagepaths, **kwargs)[start:end]
    images = [imageio.imread(imagepath) for imagepath in imagepaths]
    indices = [i for i in range(0,len(images))]
    for _, image_io, imagepath in zip(indices, images, imagepaths):
        for idx, seq in enumerate(seqs):
            images_aug = [seq(image=image_io) for _ in range(1)]
            #ia.imshow(np.hstack([images[_]]))
            #ia.imshow(ia.draw_grid(images_aug, cols=1, rows=1))
            filename = imagepath.split('/')[-1].replace('.jpg',f'''_aug{idx}.jpg''',1)
            outpath = '/'.join(imagepath.split('/')[:-1]) + os.sep + 'aug/' + filename 
            for image_aug in images_aug:
                imageio.imwrite(outpath, image_aug)
        print(f'''Processed image batch #{_}''')
#     for augmented_image in augmented_images:
    return f'''Done exporting images'''


        
def recall(tp, fn):
    return tp / (tp + fn)

def precision(tp, fp):
    return tp / (tp + fp)

def f1score(r, p):
    return (2 * r * p) / (r + p)

@tdec
def get_preds_txt(path, confidence = False):

    dictdf = {}
    files = get_files_in_dir(path = path, extension = 'jpg', show_folders = False)
    errorcount = 0
    for file in files:
        labelpath = path + '/labels' + os.sep + file.replace('jpg','txt',1)
        imagepath = path + os.sep + file
        #print(imagepath)
        try:
            with open(labelpath) as labelfile:
                lines = [line.rstrip() for line in labelfile.readlines()]
                newline = ''
                for idx, line in enumerate(lines):
#                     if idx > 4:
#                        continue
#                     else:
                    if confidence:
                        classid, xcenter, ycenter, box_width, box_height, conf = line.split(' ')
                    else: 
                        classid, xcenter, ycenter, box_width, box_height = line.split(' ')
                        conf = ''
                    if int(classid) > 3:
                        continue
                    else: classid = str(int(classid) + 1)
                    try:
                        xcenter, ycenter, box_width, box_height = float(xcenter), float(ycenter), float(box_width), float(box_height)
                    except:
                        #print('Error')
                        errorcount += 1
                    imwidth, imheight, colorchannels = cv2.imread(imagepath).shape
                    xmin = np.round((xcenter - box_width / 2) * imwidth, 0)
                    xmax = np.round((xcenter + box_width / 2) * imwidth, 0)
                    ymin = np.round((ycenter - box_height / 2) * imheight, 0)
                    ymax = np.round((ycenter + box_height / 2) * imheight, 0)
                    newline += classid + ' ' + str(int(xmin)) + ' ' + str(int(ymin)) + ' ' + str(int(xmax)) + ' ' + str(int(ymax)) + ' '
                    dictdf[file] = newline
        except:
            #print('No Labels')
            dictdf[file] = ''
            continue
    print('Errorcount: ', errorcount)
    outdf = pd.DataFrame.from_dict(dictdf, orient = 'index').reset_index()
    return outdf


@tdec
def get_preds_df(testpath, extension = 'txt'):
    
    predsdf = pd.DataFrame(columns = ['filename','class','xmin','ymin','xmax','ymax'])
    files = [f for f in next(walk(testpath), (None, None, []))[2] if f.endswith(extension)] 
    for filename in files:
        labelpath = testpath + os.sep + filename
        print(labelpath)
        newdf = pd.read_csv(labelpath, header = None)
        newdf = newdf[0].str.split(' ', expand = True)
        newdf.columns = ['class','xcenter','ycenter','box_width','box_height']
        newdf[['class','xcenter','ycenter','box_width','box_height']] = newdf[['class','xcenter','ycenter','box_width','box_height']].astype(float)
        imagename = filename.replace('txt','jpg',1)
        imagepath = testpath.replace('/labels','',1) + os.sep + imagename
        imwidth, imheight, colorchannels = cv2.imread(imagepath).shape
        newdf['xmin'] = np.round((newdf['xcenter'] - newdf['box_width'] / 2) * imwidth, 0)   
        newdf['xmax'] = np.round((newdf['xcenter'] + newdf['box_width'] / 2) * imwidth, 0)
        newdf['ymin'] = np.round((newdf['ycenter'] - newdf['box_height'] / 2) * imheight, 0)  
        newdf['ymax'] = np.round((newdf['ycenter'] + newdf['box_height'] / 2) * imheight, 0)
        newdf[['class','xmin','ymin','xmax','ymax']] = newdf[['class','xmin','ymin','xmax','ymax']].astype(int)
        newdf.insert(0, 'filename', imagename) #df.insert(loc, column, value)
        newdf.drop(['xcenter','ycenter','box_width','box_height'], axis = 1, inplace = True)
        predsdf = pd.concat([predsdf, newdf], axis = 0)
    return predsdf



def parseXML(xmlfile):
  
    # create element tree object
    tree = ET.parse(xmlfile)  
    # get root element
    root = tree.getroot()
    # create empty list for news items
    coords = []
    coordsdict = {}
    print(root)
    for item in root.findall('./node'):
        try:
            lat = item.attrib['lat']
            long = item.attrib['lon']
            id_ = item.attrib['id']
            coords.append((lat,long,id_))
            coordsdict[id_] = (lat,long)
        except:
            print('Could not retrieve one or more elements')

    return coords, coordsdict


def get_results(path):
    files, folders = get_files_in_dir(path, show_folders = True)
    outdf = pd.DataFrame({'modeltype':[], 'model_name':[], 'max_train_f1':[], 'bestepoch':[], 'maxepoch':[], 'percbest':[], 'epochs':[], 'batch_size':[],'imgsz':[], 'max_train_recall':[], 'max_train_precision':[],'metrics/mAP_0.5':[], 'metrics/mAP_0.5:0.95':[], 'lr0':[], 'lrf':[], 'momentum':[], 'weight_decay':[], 'warmup_epochs':[],
                          'warmup_momentum':[], 'warmup_bias_lr':[], 'box':[], 'cls':[], 'cls_pw':[], 'obj':[], 'obj_pw':[], 'iou_t':[], 'anchor_t':[], 'fl_gamma':[], 'hsv_h':[], 'hsv_s':[], 'hsv_v':[], 'degrees':[], 'translate':[], 
                          'scale':[], 'shear':[], 'perspective':[], 'flipud':[], 'fliplr':[], 'mosaic':[], 'mixup':[], 'copy_paste':[], 'rect':[],'resume':[],'nosave':[],'noval':[],'noautoanchor':[],'evolve':[],'bucket':[],'cache':[],'image_weights':[],'device':[],'multi_scale':[],'single_cls':[],'adam':[],'sync_bn':[],'workers':[],'project':[],'name':[],'exist_ok':[],'quad':[],'linear_lr':[],'label_smoothing':[],'patience':[],'freeze':[],'save_period':[],'local_rank':[],'entity':[],'upload_dataset':[],'bbox_interval':[],'artifact_alias':[],'save_dir':[]}) # 'weights':[],
    for folder in folders:
        resultspath = path + os.sep + folder + '/results.csv'
        if not os.path.exists(resultspath):
            print(f'''{folder} results does not exist''')
            continue
        hyppath = path + os.sep + folder + '/hyp.yaml'
        optpath = path + os.sep + folder + '/opt.yaml'
        df = pd.read_csv(resultspath, sep = ',')
        df.columns = [col.strip() for col in df.columns]
        df['f1'] = (2 * df['metrics/precision'] * df['metrics/recall']) / (df['metrics/precision'] + df['metrics/recall'])
        maxrecall = df['metrics/recall'].max()
        maxprecision = df['metrics/precision'].max()
        maxf1 = df['f1'].max()
        maxmAP_50 = df['metrics/mAP_0.5'].max()
        maxmAP_5095 = df['metrics/mAP_0.5:0.95'].max()
        bestepoch = df.loc[df['f1'] == maxf1].index[0]
        maxepoch = df['epoch'].max()
        
        #hypyaml
        hypyaml = pd.read_csv(hyppath, sep = ':', header = None)
        hypyaml.columns = ['hyperparameter','value']
        lr0 = hypyaml.loc[hypyaml['hyperparameter'] == 'lr0'][['value']].iloc[0,0]
        lrf = hypyaml.loc[hypyaml['hyperparameter'] == 'lrf'][['value']].iloc[0,0]
        momentum = hypyaml.loc[hypyaml['hyperparameter'] == 'momentum'][['value']].iloc[0,0]
        weight_decay = hypyaml.loc[hypyaml['hyperparameter'] == 'weight_decay'][['value']].iloc[0,0]
        warmup_epochs = hypyaml.loc[hypyaml['hyperparameter'] == 'warmup_epochs'][['value']].iloc[0,0]
        warmup_momentum = hypyaml.loc[hypyaml['hyperparameter'] == 'warmup_momentum'][['value']].iloc[0,0]
        warmup_bias_lr = hypyaml.loc[hypyaml['hyperparameter'] == 'warmup_bias_lr'][['value']].iloc[0,0]
        box = hypyaml.loc[hypyaml['hyperparameter'] == 'box'][['value']].iloc[0,0]
        cls = hypyaml.loc[hypyaml['hyperparameter'] == 'cls'][['value']].iloc[0,0]
        cls_pw = hypyaml.loc[hypyaml['hyperparameter'] == 'cls_pw'][['value']].iloc[0,0]
        obj = hypyaml.loc[hypyaml['hyperparameter'] == 'obj'][['value']].iloc[0,0]
        obj_pw = hypyaml.loc[hypyaml['hyperparameter'] == 'obj_pw'][['value']].iloc[0,0]
        iou_t = hypyaml.loc[hypyaml['hyperparameter'] == 'iou_t'][['value']].iloc[0,0]
        anchor_t = hypyaml.loc[hypyaml['hyperparameter'] == 'anchor_t'][['value']].iloc[0,0]
        fl_gamma = hypyaml.loc[hypyaml['hyperparameter'] == 'fl_gamma'][['value']].iloc[0,0]
        hsv_h = hypyaml.loc[hypyaml['hyperparameter'] == 'hsv_h'][['value']].iloc[0,0]
        hsv_s = hypyaml.loc[hypyaml['hyperparameter'] == 'hsv_s'][['value']].iloc[0,0]
        hsv_v = hypyaml.loc[hypyaml['hyperparameter'] == 'hsv_v'][['value']].iloc[0,0]
        degrees = hypyaml.loc[hypyaml['hyperparameter'] == 'degrees'][['value']].iloc[0,0]
        translate = hypyaml.loc[hypyaml['hyperparameter'] == 'translate'][['value']].iloc[0,0]
        scale = hypyaml.loc[hypyaml['hyperparameter'] == 'scale'][['value']].iloc[0,0]
        shear = hypyaml.loc[hypyaml['hyperparameter'] == 'shear'][['value']].iloc[0,0]
        perspective = hypyaml.loc[hypyaml['hyperparameter'] == 'perspective'][['value']].iloc[0,0]
        flipud = hypyaml.loc[hypyaml['hyperparameter'] == 'flipud'][['value']].iloc[0,0]
        fliplr = hypyaml.loc[hypyaml['hyperparameter'] == 'fliplr'][['value']].iloc[0,0]
        mosaic = hypyaml.loc[hypyaml['hyperparameter'] == 'mosaic'][['value']].iloc[0,0]
        mixup = hypyaml.loc[hypyaml['hyperparameter'] == 'mixup'][['value']].iloc[0,0]
        copy_paste = hypyaml.loc[hypyaml['hyperparameter'] == 'copy_paste'][['value']].iloc[0,0]
        
        #optyaml
        optyaml = pd.read_csv(optpath, sep = ':', header = None)
        optyaml.columns = ['hyperparameter','value']
        weights = optyaml.loc[optyaml['hyperparameter'] == 'weights'][['value']].iloc[0,0]
        modeltype = weights.split('/')[-1].split('.')[0]
        cfg = optyaml.loc[optyaml['hyperparameter'] == 'cfg'][['value']].iloc[0,0]
        #data = optyaml.loc[optyaml['hyperparameter'] == 'data'][['value']].iloc[0,0]
        #hyp = optyaml.loc[optyaml['hyperparameter'] == 'hyp'][['value']].iloc[0,0]
        epochs = optyaml.loc[optyaml['hyperparameter'] == 'epochs'][['value']].iloc[0,0]
        batch_size = optyaml.loc[optyaml['hyperparameter'] == 'batch_size'][['value']].iloc[0,0]
        imgsz = optyaml.loc[optyaml['hyperparameter'] == 'imgsz'][['value']].iloc[0,0]
        rect = optyaml.loc[optyaml['hyperparameter'] == 'rect'][['value']].iloc[0,0]
        resume = optyaml.loc[optyaml['hyperparameter'] == 'resume'][['value']].iloc[0,0]
        nosave = optyaml.loc[optyaml['hyperparameter'] == 'nosave'][['value']].iloc[0,0]
        noval = optyaml.loc[optyaml['hyperparameter'] == 'noval'][['value']].iloc[0,0]
        noautoanchor = optyaml.loc[optyaml['hyperparameter'] == 'noautoanchor'][['value']].iloc[0,0]
        evolve = optyaml.loc[optyaml['hyperparameter'] == 'evolve'][['value']].iloc[0,0]
        bucket = optyaml.loc[optyaml['hyperparameter'] == 'bucket'][['value']].iloc[0,0]
        cache = optyaml.loc[optyaml['hyperparameter'] == 'cache'][['value']].iloc[0,0]
        image_weights = optyaml.loc[optyaml['hyperparameter'] == 'image_weights'][['value']].iloc[0,0]
        device = optyaml.loc[optyaml['hyperparameter'] == 'device'][['value']].iloc[0,0]
        multi_scale = optyaml.loc[optyaml['hyperparameter'] == 'multi_scale'][['value']].iloc[0,0]
        single_cls = optyaml.loc[optyaml['hyperparameter'] == 'single_cls'][['value']].iloc[0,0]
        adam = optyaml.loc[optyaml['hyperparameter'] == 'adam'][['value']].iloc[0,0]
        sync_bn = optyaml.loc[optyaml['hyperparameter'] == 'sync_bn'][['value']].iloc[0,0]
        workers = optyaml.loc[optyaml['hyperparameter'] == 'workers'][['value']].iloc[0,0]
        project = optyaml.loc[optyaml['hyperparameter'] == 'project'][['value']].iloc[0,0]
        name = optyaml.loc[optyaml['hyperparameter'] == 'name'][['value']].iloc[0,0]
        exist_ok = optyaml.loc[optyaml['hyperparameter'] == 'exist_ok'][['value']].iloc[0,0]
        quad = optyaml.loc[optyaml['hyperparameter'] == 'quad'][['value']].iloc[0,0]
        linear_lr = optyaml.loc[optyaml['hyperparameter'] == 'linear_lr'][['value']].iloc[0,0]
        label_smoothing = optyaml.loc[optyaml['hyperparameter'] == 'label_smoothing'][['value']].iloc[0,0]
        patience = optyaml.loc[optyaml['hyperparameter'] == 'patience'][['value']].iloc[0,0]
        freeze = optyaml.loc[optyaml['hyperparameter'] == 'freeze'][['value']].iloc[0,0]
        save_period = optyaml.loc[optyaml['hyperparameter'] == 'save_period'][['value']].iloc[0,0]
        local_rank = optyaml.loc[optyaml['hyperparameter'] == 'local_rank'][['value']].iloc[0,0]
        entity = optyaml.loc[optyaml['hyperparameter'] == 'entity'][['value']].iloc[0,0]
        upload_dataset = optyaml.loc[optyaml['hyperparameter'] == 'upload_dataset'][['value']].iloc[0,0]
        bbox_interval = optyaml.loc[optyaml['hyperparameter'] == 'bbox_interval'][['value']].iloc[0,0]
        artifact_alias = optyaml.loc[optyaml['hyperparameter'] == 'artifact_alias'][['value']].iloc[0,0]
        save_dir = optyaml.loc[optyaml['hyperparameter'] == 'save_dir'][['value']].iloc[0,0]

        newdf = pd.DataFrame({'modeltype':[modeltype], 'model_name':[folder], 'max_train_f1':[maxf1], 'bestepoch':[bestepoch], 'maxepoch':[maxepoch], 'percbest': [bestepoch / maxepoch], 'epochs':[epochs],'batch_size':[batch_size],'imgsz':[imgsz], 'max_train_recall':[maxrecall], 'max_train_precision':[maxprecision], 'metrics/mAP_0.5':[maxmAP_50], 'metrics/mAP_0.5:0.95':[maxmAP_5095], 
                              'lr0':[lr0], 'lrf':[lrf], 'momentum':[momentum], 'weight_decay':[weight_decay], 'warmup_epochs':[warmup_epochs], 'warmup_momentum':[warmup_momentum], 'warmup_bias_lr':[warmup_bias_lr], 
                              'box':[box], 'cls':[cls], 'cls_pw':[cls_pw], 'obj':[obj], 'obj_pw':[obj_pw], 'iou_t':[iou_t], 'anchor_t':[anchor_t], 'fl_gamma':[fl_gamma], 'hsv_h':[hsv_h], 'hsv_s':[hsv_s], 'hsv_v':[hsv_v], 
                              'degrees':[degrees], 'translate':[translate], 'scale':[scale], 'shear':[shear], 'perspective':[perspective], 'flipud':[flipud], 'fliplr':[fliplr], 'mosaic':[mosaic], 'mixup':[mixup], 'copy_paste':[copy_paste],
                             'rect':[rect],'resume':[resume],'nosave':[nosave],'noval':[noval],'noautoanchor':[noautoanchor],'evolve':[evolve],'bucket':[bucket],'cache':[cache],'image_weights':[image_weights],'device':[device],'multi_scale':[multi_scale],'single_cls':[single_cls],'adam':[adam],'sync_bn':[sync_bn],'workers':[workers],'project':[project],'name':[name],'exist_ok':[exist_ok],'quad':[quad],'linear_lr':[linear_lr],'label_smoothing':[label_smoothing],'patience':[patience],'freeze':[freeze],'save_period':[save_period],'local_rank':[local_rank],'entity':[entity],'upload_dataset':[upload_dataset],'bbox_interval':[bbox_interval],'artifact_alias':[artifact_alias],'save_dir':[save_dir]}) #'weights':[weights],
        outdf = pd.concat([outdf, newdf], axis = 0)
        
    return outdf.sort_values(by = 'max_train_f1', ascending = False)



def get_cmds(*tests, weights, nms_thresholds, conf_thresholds, counter = 0, m = 'm1'):
    newcmdlist = []
    for nms in nms_thresholds:
        for conf in conf_thresholds:
            for test in tests:
                if m == 'm1':
                    detect = '/Users/Administrator/DS/IEEE-Big-Data-2020/yolov5/detect.py'
                    path = f'''\'/Users/Administrator/DS/IEEE-Big-Data-2020/yolov5/runs/detect/exp{counter}\''''
                else:  
                    detect = '/Users/phil0/DS/IEEE/yolov5/detect.py'
                    path = f'''\'/Users/phil0/DS/IEEE/yolov5/runs/detect/exp{counter}\''''
                newcmd = f''' #pass {counter - 1} nms = {nms} conf = {conf} {test}
!python {detect} --weights {weights} --img 416 --source ./{test}/All_Countries/images --save-txt --save-conf --conf-thres {conf} --iou-thres {nms}  --agnostic-nms --augment \n
import sys
sys.path.append('/Users/Administrator/DS/IEEE-Big-Data-2020')
sys.path.append('/Users/phil0/DS/IEEE/Object_Detection')
import utils2
from importlib import reload
reload(utils2)
from utils2 import * \n
dictdf = get_preds_txt(path = {path}, confidence=True)
dictdf.to_csv('results/{test}_nms{int(np.round(nms * 100, 0))}_conf{int(np.round(conf*100,0))}_pass{counter - 1}.txt', sep = ',', index = False, header = False)
print(\'Completed pass {counter - 1} on {test}\')
\n'''
                newcmdlist.append(newcmd)
                counter += 1
                
    return newcmdlist



#### General Torch Functions ####

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T

def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.\
           fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
   
    return model


def get_transform(train = False):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def load_torch_model(weights_path, num_classes):    
    loaded_model = get_model(num_classes = num_classes)
    loaded_model.load_state_dict(torch.load(weights_path))
    return loaded_model




def parse_one_annot(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data["filename"] == filename][["xmin", "ymin","xmax", "ymax"]].values
    return boxes_array

def parse_labels(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    labels_array = data[data["filename"] == filename][["class"]].values
    labels_array = [item for sublist in labels_array for item in sublist]
    return labels_array

class IEEEDataset(torch.utils.data.Dataset):
   
    def __init__(self, root, data_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.path_to_data_file = data_file

    def __getitem__(self, idx):
          # load images and bounding boxes
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
#         pil_to_tensor = T.ToTensor()(img).unsqueeze_(0)
#         print(pil_to_tensor.shape) 
#         print(type(pil_to_tensor))
#         print(pil_to_tensor)
# tensor_to_pil = transforms.ToPILImage()(pil_to_tensor.squeeze_(0))
# print(tensor_to_pil.size)
#         print(img)

        box_list = parse_one_annot(self.path_to_data_file, self.imgs[idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)    
        num_objs = len(box_list)
        # there is only one class
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        labels_list = parse_labels(self.path_to_data_file, self.imgs[idx])
        labels = torch.as_tensor(labels_list, dtype=torch.int64)   
        print('Labels are', labels)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["imagefile"] = self.imgs[idx]
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    
    
from PIL import ImageDraw

@tdec
def get_rcnn_preds(dataset_test, loaded_model):
    dictdf = {}
    len_dataset_test = len(dataset_test)
    for idx in range(len_dataset_test):
        start_time = time.time()
        img, _ = dataset_test[idx]
        imagefile = _['imagefile']
        label_boxes = np.array(dataset_test[idx][1]["boxes"])
        #put the model in evaluation mode
        loaded_model.eval()
        with torch.no_grad():
            prediction = loaded_model([img])
            #predstr = ''
            counter = 0
            predictions = 0
            for box, label, conf in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
                xmin = str(int(np.round(box[0].item(), 0)))
                ymin = str(int(np.round(box[1].item(), 0)))
                xmax = str(int(np.round(box[2].item(), 0)))
                ymax = str(int(np.round(box[3].item(), 0)))
                label = str(label.item())
                conf = float(conf.item())
                outputstr = label + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' '
                if outputstr is None: outputstr = ''
                #predstr += outputstr
                keyn = imagefile + '_' + str(counter)
                dictdf[keyn] = [outputstr, conf]
                counter += 1
                predictions += 1
        print('{}s to output {} predictions for image {} / {}'.format(np.round(time.time() - start_time, 1), predictions, idx, len_dataset_test))
    outdf = pd.DataFrame.from_dict(dictdf, orient = 'index').reset_index()
    outdf.columns = ['filename', 'prediction', 'conf'] 
    return outdf


@tdec
def inters_counter(list1, list2):
    counter = 0
    for item in list1:
        if item in list2:
            counter += 1
    return counter


@tdec
def get_dataset_filenames(*args, datasetobj, **kwargs):
    filenames = []
    print(len(datasetobj))
    for i in range(len(datasetobj)):
        img, target, filename = datasetobj[i]
        print(filename)
        filenames.append(filename)
    return filenames


def filter_preds(df, conf_threshold, testu):
    df = df.loc[df['conf'] > conf_threshold]
    df['filename2'] = df['filename'].str.split('_')
    df['filename2'] = df['filename2'].str[0:2]
    df['filename2'] = ['_'.join(map(str, l)) for l in df['filename2']]
    df['preds2'] = df.groupby(['filename2'])['prediction'].transform(lambda x: ''.join(x))
    df = df.drop_duplicates(subset = ['filename2', 'preds2'], keep = 'first')
    df = df.drop(['filename', 'prediction', 'conf', 'labels'], axis = 1)
    filenameu = df['filename2'].unique()
    dictdf = {}
    for fileu in testu:
        if fileu not in filenameu:
            dictdf[fileu] = ''
    newdf = pd.DataFrame.from_dict(dictdf, orient = 'index').reset_index()
    newdf.columns = ['filename2','preds2']
    df = pd.concat([df, newdf], axis = 0)        
    return df


def visualize_rcnn(idxs, dataset_test, loaded_model, conf_threshold = 0.01, groundtruth = False):
    for idx in idxs:
        img, target, filename = dataset_test[idx]
        print(filename)
        loaded_model.eval()
        with torch.no_grad():
            prediction = loaded_model([img])
        image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        draw = ImageDraw.Draw(image)
        if groundtruth:
            label_boxes = dataset_test[idx][1]
            for i in range(len(label_boxes['boxes'])):
                draw.rectangle([(label_boxes['boxes'][i][0].item(), label_boxes['boxes'][i][1].item()), (label_boxes['boxes'][i][2].item(), label_boxes['boxes'][i][3].item())], outline ="green", width = 3)
                draw.text((label_boxes['boxes'][i][0].item(), label_boxes['boxes'][i][3].item() + 5), text = str(label_boxes['labels'][i].item()), fill = 'green')
        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().numpy(),
                            decimals= 4)
            if score > conf_threshold:
               draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline ="red", width =3)
               pred_label = str(prediction[0]['labels'][element].item()) + ' ' + str(score) 
               draw.text((boxes[0], boxes[1] - 10), text = pred_label, fill = 'red')
        image.show()
    return image

def dropcol(df):
    try:
        df = df.drop(['Unnamed: 0'], axis = 1)
    except: print('Could not drop Unnamed: 0 col')
    return df


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.path_to_data_file = data_file
    def __getitem__(self, idx):
       # load images and bounding boxes
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        box_list = parse_one_annot(self.path_to_data_file, self.imgs[idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        num_objs = len(box_list)
        # pulling labels for each image 
        labels_list = parse_labels(self.path_to_data_file, self.imgs[idx])
        labels = torch.as_tensor(labels_list, dtype=torch.int64) 
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        if self.transforms is not None:
            img, target = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])(img, target)
        return transform(img), target
    def __len__(self):
        return len(self.imgs)

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T

def get_dataloaders(idx_cutoff, train_batch_size, root, data_file, num_workers = 0):
    # use our dataset and defined transformations
    dataset = TorchDataset(root=root,  data_file=data_file, transforms = None)
    dataset_test = TorchDataset(root=root, data_file=data_file, transforms = None)
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    # valindices = torch.randperm(valmask).tolist()
    # trainindices = torch.randperm(trainmask).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-idx_cutoff])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-idx_cutoff:])
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
                  dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                  collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
             dataset_test, batch_size=1, shuffle=False, num_workers=num_workers,
             collate_fn=utils.collate_fn)
    print("We have: {} examples, {} are training and {} testing".format(len(indices), len(dataset), len(dataset_test)))
    return dataset, dataset_test, data_loader, data_loader_test


def train_fastrcnn(num_epochs, model, optimizer, lr_scheduler, data_loader_train, data_loader_val, device, notequal = [0], print_freq = 50, save_every = 10, output_weights_file = 'train/weights/model_1218_fastrcnn_{}e'):
    
    for idx, epoch in enumerate(range(num_epochs)):
        if (idx not in notequal) & (idx % save_every == 0):
            torch.save(model.state_dict(), output_weights_file.format(idx))
        # train for one epoch, printing every print_freq iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)
        
        
#import google_streetview.api
from utils2 import *

def get_multiple_gsv_im(locations, size, heading, pitch, key, outfolder = 'gsv_downloads', show_image = False):
    
     # Define parameters for street view api
    params = {
        'size': size, # max 640x640 pixels
        'location': locations,
        'heading': heading,
        'pitch': pitch,
        'key': key
    }
    api_list = google_streetview.helpers.api_list(params)
    results = google_streetview.api.results(api_list)
    results.download_links(outfolder)
    if show_image:
        for i in range(len(api_list)):
            show_im(f'''{outfolder}/gsv_{i}.jpg''')

        
@tdec
def get_road_section_scores(*paths, frequency_factor = 0.5):
    '''Function to produce scores of road sections by weighting road damage frequency and severity along road section'''
    intermediate_score = {}
    for idx, path in enumerate(list(paths)):
        files = get_files_in_dir(path = path, extension = 'jpg', show_folders = False)
        distress_ct = 0
        rsconfidences = []
        for file in files:
            labelpath = path + '/labels' + os.sep + file.replace('jpg','txt',1)
            try:
                with open(labelpath) as labelfile:
                    lines = [line.rstrip() for line in labelfile.readlines()]
                    distress_ct += len(lines)
                    for line in lines:
                        classid, xcenter, ycenter, box_width, box_height, conf = line.split(' ')
                        rsconfidences.append(float(conf))
            except:
                continue
        if len(rsconfidences) != 0:  
            average_conf = sum(rsconfidences) / len(rsconfidences)
        else:
            average_conf = 0.5
        intermediate_score[idx] = {'distress_ct':distress_ct, 'average_conf':average_conf}
    return intermediate_score


def show1(*objs):
    for obj in objs:
        try: print(obj.shape, type(obj))
        except: print(type(obj))
            
def display_np_array(array):
    #function to display numpy array using matplotlib 
    plt.imshow(array, interpolation='nearest')
    plt.show()
    
    
# import the necessary packages
import cv2
import imutils
import argparse
import numpy as np
from imutils import contours
    
    
def read_ocr(ocr_path):
    # load the reference OCR-A image from disk, convert it to grayscale,
    # and threshold it, such that the digits appear as *white* on a
    # *black* background
    # and invert it, such that the digits appear as *white* on a *black*
    ref = cv2.imread(ocr_path)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
    
    # find contours in the OCR-A image (i.e,. the outlines of the digits)
    # sort them from left to right, and initialize a dictionary to map
    # digit name to the ROI
    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refCnts = imutils.grab_contours(refCnts)
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
    return ref, refCnts


def get_digits(ref, refCnts):
    digits = {}
    # loop over the OCR-A reference contours
    for (i, c) in enumerate(refCnts):
        # Compute the bounding box for the digit, 
        # extract it, and resize it to a fixed size. 
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # update the digits dictionary, mapping the digit name to the ROI
        digits[i] = roi
    return digits

def get_kernel(size):
    #create kernel of size size
    return cv2.getStructuringElement(cv2.MORPH_RECT, size)

def get_grayscale(imagepath):
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(imagepath)
    image = imutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def cleanAndRead(img,contours):
    """Takes the extracted contours and once it passes the rotation
    and ratio checks it passes the potential license plate to PyTesseract for OCR reading"""
    for i,cnt in enumerate(contours):
        min_rect = cv2.minAreaRect(cnt)

        if validateRotationAndRatio(min_rect):

            x,y,w,h = cv2.boundingRect(cnt)
            plate_img = img[y:y+h,x:x+w]

            if(isMaxWhite(plate_img)):
                clean_plate, rect = cleanPlate(plate_img)
                
                if rect:
                    row, col = 1, 2
                    fig, axs = plt.subplots(row, col, figsize=(15, 10))
                    fig.tight_layout()
                    
                    x1,y1,w1,h1 = rect
                    x,y,w,h = x+x1,y+y1,w1,h1
                    
                    axs[0].imshow(cv2.cvtColor(clean_plate, cv2.COLOR_BGR2RGB))
                    axs[0].set_title('Cleaned Plate')
                    cv2.imwrite('cleaned_plate.jpg', clean_plate)
                    
                    plate_im = Image.fromarray(clean_plate)
                    text = tess.image_to_string(plate_im, lang='eng')
                    print("Detected Text : ", text)

                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    axs[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    axs[1].set_title('Detected Plate')
                    cv2.imwrite('detected_plate.jpg', img)
                    
                    plt.show()
                    
def cleanPlate(plate):
    """This function gets the countours that most likely resemeber the shape
    of a license plate"""    
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thresh = cv2.dilate(gray, kernel, iterations = 1)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)

        max_cnt = contours[max_index]
        max_cntArea = areas[max_index]
        x,y,w,h = cv2.boundingRect(max_cnt)

#         if not ratioCheck(max_cntArea,w,h):
#             return plate,None

        cleaned_final = thresh[y:y+h, x:x+w]
        plt.imshow(cv2.cvtColor(cleaned_final, cv2.COLOR_BGR2RGB))
        plt.title('Function Test'); plt.show()
        
        return cleaned_final,[x,y,w,h]

    else:
        return plate, None

import sys
sys.path.append(r'C:/Users/phil0/Tesseract-OCR/tesseract.exe')
import pytesseract as tess
# tess.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
tess.pytesseract.tesseract_cmd = r'C:/Users/phil0/Tesseract-OCR/tesseract.exe'

def check_locs(bb, min_x, max_x, min_y, max_y):
    #bb in format xmin, ymin, width, height
    if bb[0] > min_x and bb[0] + bb[2] < max_x and bb[1] > min_y and bb[1] + bb[3] < max_y:
        return True
    return False

@tdec
def read_cc_nums(ocr_path, imagepath, char_min_y):
    ref, refCnts = read_ocr(ocr_path)
    digits = get_digits(ref, refCnts)
    rectKernel = get_kernel((9,3))
    sqKernel = get_kernel((5,5)) 
    image, gray = get_grayscale(imagepath)
    # apply a tophat (whitehat) morphological operator to find light
    # regions against a dark background (i.e., the credit card numbers)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")    
    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  
    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    # find contours in the thresholded image, then initialize the
    # list of digit locations
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitlocs = []
    charlocs = []
    #bbs in format xmin, ymin, width, height
    digitbbs = []
    # loop over the contours
    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        #digits
        if check_locs((x, y, w, h), min_x = 15, max_x = 285, min_y = 85, max_y = 145):
            digitbbs.append((x, y, w, h))
            ar = w / float(h)
            # since credit cards used a fixed size fonts with 4 groups
            # of 4 digits, we can prune potential contours based on the
            # aspect ratio
            if ar > 2.5 and ar < 4:
                # contours can further be pruned on minimum/maximum width
                # and height
                if (w > 40 and w < 55) and (h > 10 and h < 20):
                    # append the bounding box region of the digits group
                    # to our locations list
                    digitlocs.append((x, y, w, h))      
    #chars
    min_x, max_x, min_y, max_y = 15, 200, char_min_y, 1000
    group = gray[min_y - 5:max_y, min_x - 5:max_x]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img = Image.fromarray(group)
    display(img)
    text = get_chars(group, lang='eng')
    # sort the digit locations from left-to-right, then initialize the
    # list of classified digits
    digitlocs = sorted(digitlocs, key=lambda x:x[0])
    output = [] 
    
    #digits
    # loop over the 4 groupings of 4 digits
    for (i, (gX, gY, gW, gH)) in enumerate(digitlocs):
        # initialize the list of group digits
        groupOutput = []
        # extract the group ROI of 4 digits from the grayscale image,
        # then apply thresholding to segment the digits from the
        # background of the credit card
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        img3 = Image.fromarray(group)
        display(img3)
        # detect the contours of each individual digit in the group,
        # then sort the digit contours from left to right
        digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = imutils.grab_contours(digitCnts)
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
        # loop over the digit contours
        for c in digitCnts:
            # compute the bounding box of the individual digit, extract
            # the digit, and resize it to have the same fixed size as
            # the reference OCR-A images
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            # initialize a list of template matching scores
            scores = []
            # loop over the reference digit name and digit ROI
            for (digit, digitROI) in digits.items():
                # apply correlation-based template matching, take the
                # score, and update the scores list
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            # The classification for the digit ROI will be the reference
            # digit name with the *largest* template matching score
            groupOutput.append(str(np.argmax(scores)))

        # draw the digit classifications around the group
        cv2.rectangle(image, (gX - 5, gY - 5),
            (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # update the output digits list
        output.extend(groupOutput)
    print(imagepath)
    newim = Image.fromarray(image)
    display(newim)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return charlocs, digitlocs, cnts, thresh, digitbbs, ''.join(output), text, image
        
def get_chars(img, show_image = False, **kwargs):
    if isinstance(img, str):
        img = cv2.imread(img)
    if show_image:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Test Image'); plt.show()
    image = Image.fromarray(img)
    text = tess.image_to_string(image, **kwargs)
    print("PyTesseract Detected the following text: ", text)
    return text

@tdec
def readcsv(path, **kwargs):
    return pd.read_csv(path, sep = ',', **kwargs)


@tdec
def show_bb(image, bbs: List, pascal_voc = False, resize = None):
    '''Function to show bounding box predictions on an image
    PASCAL VOC: bb in format <xmin><ymin><xmax><ymax>
    YOLO: bb in format <xcenter><ycenter><half height><half width>
    bb is normally passed in YOLO format'''
    
    if pascal_voc:
        pass
    if isinstance(image, str):
        if resize:
            img = cv2.imread(image)
            img = cv2.resize(img, dsize = resize, interpolation = cv2.INTER_AREA)
            print(img.shape)
        else: img = cv2.imread(image)
    else:
        img = Image.fromarray(image)
    fig, ax = plt.subplots(figsize = (6,9))
    for idx, bb in enumerate(bbs):
        x, y, w, h = bb[0], bb[1], bb[2], bb[3]
        ax.xaxis.tick_top()
        ax.add_patch(patches.Rectangle((x,y),w,h, fill=False, edgecolor='red', lw=2))
        ax.text(x,(y-20),str(idx),verticalalignment='top', color='white',fontsize=10,
                weight='bold').set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])
    ax.imshow(img)
    return 

def getimsize(imagepath):
    img = cv2.imread(imagepath)
    return img.shape


def process_str(string):
    string = string.upper()
    string = re.sub(r'[^\w\s]','',string)
    string = string.splitlines()
    if len(string) == 0:
        return string
    return string[0]


@tdec
def test_cc_thresholds(df, range_thresholds):
    resultsdict = {}
    for charminy in range_thresholds:
        print(charminy)
        digit_tp = 0
        char_tp = 0
        for idx, row in df.iterrows():
            charlocs, digitlocs, cnts, thresh, digitbbs, output, text, image = read_cc_nums(ocr_path = 'ocr_a_reference.png', imagepath = row['filename'], char_min_y = charminy)
            show_bb(imagepath = row['filename'], bbs = digitbbs)
            print('Credit card number:', output)
            print('Digit Groundtruth:', row['cc_number'])
            if row['cc_number'] == str(output):
                digit_tp += 1
                print('Correct digits: ', digit_tp)
            else:
                print('Digits False Positive')
            print('\n')
            print('Text Groundtruth:', row['cardholder'])
            chartext = process_str(text)
            print('Final detected text:', chartext)
            if row['cardholder'] == chartext:
                char_tp += 1
                print('Correct chars: ', char_tp)
            else:
                print('Text False Positive')
            if row['cc_number'] != str(output) and row['cardholder'] != chartext:
                print('####')
                print('No matches')
                print('####')
            print('-----------------------')
        digits_recall = digit_tp / float(len(df))
        char_recall = char_tp / float(len(df))
        print('Digits recall:', digits_recall)
        print('Char recall:', char_recall)
        resultsdict[charminy] = (digits_recall, char_recall)
    return resultsdict 


# Unsupervised ML

# import hdbscan
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import plotly.graph_objects as go
# import plotly.express as px 

# def scale(df, cols, **kwargs):
#     df = df.loc[:, cols]
#     df = df.to_numpy()
#     if 'scaletype' in kwargs and kwargs['scaletype'] == 'minmax':
#         scaler = MinMaxScaler()
#     else:
#         scaler = StandardScaler()
#     scaler.fit(df)
#     X = scaler.transform(df)
#     dff = pd.DataFrame(X, columns = cols)
#     return dff, scaler

# @tdec
# def get_kmeans(data, cols, n_clusters = 3, random_state = 42, **kwargs):
#     data, scaler = scale(df = data, cols = cols, **kwargs)
#     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init = 10, tol = 1e-04, random_state = random_state)
#     kmeans.fit(data)
#     clusters = pd.DataFrame(data, columns = cols)
#     clusters['label'] = kmeans.labels_
#     polars = clusters.groupby('label').mean().reset_index()
#     polar = pd.melt(polar, id_vars = ['label'])
#     fig = px.line_polar(polar, r='value', theta = 'variable', color = 'label', line_close = True, height = 800, width = 1400)
#     return fig, polar, clusters, scaler

# def get_clusters(data, cols, cluster_labels):
#     data = data[cols]
#     clusters = pd.DataFrame(data, columns = cols)
#     clusters['label'] = cluster_labels
#     polar = clusters.groupby('label').mean().reset_index()
#     polar = pd.melt(polar, id_vars = ['label'])
#     fig = px.line_polar(polar, r = 'value', theta = 'variable', color = 'label', line_close = True, height = 800, width = 1400)
#     return fig, polar, clusters, scaler

# def inv_transform(data, scaler, cols, clusters):
#     data_inv = scaler.inverse_transform(data)
#     data_inv = pd.DataFrame(data_inv, columns = cols)
#     data_inv['labels'] = clusters['label']
#     cols = [col for col in data_inv.columns]
#     groupby_dict = {}
#     for idx, col in enumerate(cols):
#         if idx == 0:
#             groupby_dict[col] = ['mean', 'count']
#             continue
#         groupby_dict[col] = ['mean']
#     df = data_inv.groupby('labels').agg(groupby_dict).reset_index()
#     df_cols = ['_'.join(col) for col in [c for c in df.columns]]
#     df.columns = df_cols
#     df['perc'] = df.iloc[:,2] / df.iloc[:,2].sum()
#     return df, data_inv

# def get_hdbscan_cluster_stats_unscaled(data, scaler, cols, clusters):
#     data_inv['labels'] = clusters['label']
#     cols = [col for col in data_inv.columns]
#     groupby_dict = {}
#     for idx, col in enumerate(cols):
#         if idx == 0:
#             groupby_dict[col] = ['mean', 'count']
#             continue
#         groupby_dict[col] = ['mean']
#     df = data_inv.groupby('labels').agg(groupby_dict).reset_index()
#     df_cols = ['_'.join(col) for col in [c for c in df.columns]]
#     df.columns = df_cols
#     df['perc'] = df.iloc[:,2] / df.iloc[:,2].sum()
#     return df, data_inv
    
# def inv_transform2(data, scaler):
#     data_inv = scaler.inverse_transform(data)
#     return data_inv

# @tdec 
# def HDBSCAN_hyperparameter_search(df, min_cluster_sizes, min_sample_sizes, scaler, cols):
#     dfs =[]
#     for min_cluster_size in min_cluster_sizes:
#         for min_samples in min_samples_sizes:
#             start_time = time.time()
#             clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, min_samples = min_samples)
#             cluster_labels = clusterer.fit_predict(df)
#             clusters_unique = np.unique(cluster_labels)
#             print(clusters_unique)
#             clusters = pd.DataFrame(cluster_labels, columns = ['label'])
#             dfg, data_inv = get_hdbscan_cluster_stats_unscaled(df, scaler, cols = cols, clusters = clusters)
#             dfs.append((dfg, data_inv, clusters, min_cluster_size, min_samples))
#             print('{} sec to complete HDBSCAN with {} min_cluster_size and {} min_samples'.format(np.round(time.time() - start_time, 0), min_cluster_size, min_samples))
#     return dfs

# def filter_hdbscan_dfs(dfs, noise_threshold):
#     for dfg, data_inv, clusters, min_cluster_size, min_samples in dfs:
#         if dfg.iloc[0,-1] < noise_threshold:
#             print(min_cluster_size, min_samples, '\n', dfg.head())
            
# def get_hdbscan_polar(dfg, data, clusters, min_cluster_size, min_samples):
#     print('Min Cluster size:', min_cluster_size, '| Min Sample Size:', min_samples)
#     dfg = dfg[['labels_','tenure_mean','tenure_count','chdep_ind_mean','onln_signon_days_mean','di_bal_mean','atmdays_mean','mobile_signon_days_mean']]
#     tenure_count = dfg['tenure_count']
#     dfg = dfg.drop('tenure_count', axis = 1)
#     polar = pd.melt(dfg, id_vars = ['labels_'])
#     fig = px.line_polar(polar, r='value', theta = 'variable', color = 'labels_', line_close = True, height = 800, width = 1400)
#     dfg['perc'] = tenure_count / tenure_count.sum()
#     print(dfg)
#     fig.show()
#     return dfg
            
            
# def get_elbow_plot(X, start = 1, end = 11, elbow_annot = None, random_state = 42):
#     inertia = []
#     for i in range(start, end):
#         kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init = 10, tol = 1e-04, random_state = random_state)
#         kmeans.fit(X)
#         inertia.append(kmeans.inertia_)
#     fig = go.Figure(data = go.Scater(x=np.arange(1,11), y= inertia))
#     if elbow_annot is not None:
#         annotation = dict(x=elbow_annot, y = inertia[eblow_annot - 1], xref = 'x', yref = 'y', text = 'Elbow!', showarrow = True, arrowhead = 7, ax = 20, ay = -40)
#         fig.update_layout(annotations =[annotation])
#     fig.update_layout(title='Inertia vs Cluster Number', xaxis = dict(range=[0,11], title = 'Cluster Number'), yaxis = {'title':'Inertia'})
#     return fig, inertia
#     fig
           
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

@tdec
def remove_stopwords(words):
    stopwords = nltk.corpus.stopwords.words("english")
    words = [w for w in words if w.lower() not in stopwords]
    return words


@tdec
def get_freq_dist(text: str):
    fdist = nltk.FreqDist(word.lower() for word in nltk.word_tokenize(text))
    return fdist


@tdec
def get_concordances(corpus: str, target_word, n_lines):
    text = nltk.Text(nltk.corpus.state_union.words())
    concordances = text.concordance_list(target_word, lines=n_lines)
    return concordances

@tdec
def tokenize_text(text: List[str], granularity = 'words'):
    '''Use to tokenize list of multi-worded strings to create a text corpus
    Inputs: granularity == \'words\' for tokenization to words-level of granularity, == \'sentences' for sentence-level'''
    if granularity == 'words':
        #tokenize text to the word level
        words = [nltk.word_tokenize(word) for word in text]
    else:
        #tokenize text to the sentence level
        words = [nltk.sent_tokenize(word) for word in text]
    return words


@tdec
def get_collocations(gramtype, words):
    '''Used to return most common collections of n words where n in set [2-4] inclusive'''
    if gramtype not in ['bigram','trigram','quadgram']:
        raise ValueError('Collocation gram type should be one of \'bigram\', \'trigram\' , \'quadgram\'')
    if gramtype == 'bigram': 
        finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
    elif gramtype == 'trigram':
        finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
    else:
        finder = nltk.collocations.QuadgramCollocationFinder.from_words(words)
    return finder

@tdec
def get_vader_polarity_score(text):    
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores

@tdec
def is_text_positive(text: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)["compound"] > 0
