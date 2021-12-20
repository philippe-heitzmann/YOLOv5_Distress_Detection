

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
from typing import List, Dict

import time 

from matplotlib.pyplot import figure

from collections import defaultdict
import csv
import requests
import xml.etree.ElementTree as ET
  
import sys
sys.path.append('/Users/Administrator/DS/IEEE-Big-Data-2020')

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
def get_files_in_dir(path = './data/train/Japan/labels', show_folders = False, **kwargs):
    'Function to return all files in a directory with option of filtering for specific filetype extension'
    
    files = next(walk(path), (None, None, []))[2]
    if 'fullpath' in kwargs and kwargs['fullpath']:
        files = [path + os.sep + f for f in files if f.endswith(kwargs['extension'])]
    else: files = [f for f in files if f.endswith(kwargs['extension'])]
    if show_folders:
        folders = next(walk(path), (None, None, []))[1]
        folders = [folder for folder in folders if folder[0] != '.']
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
                ax.text(xmin,(ymin-15),str(lb),verticalalignment='top', color='white',fontsize=10,
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
                    else: classid, xcenter, ycenter, box_width, box_height = line.split(' ')
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
        #target["imagefile"] = self.imgs[idx]
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    
    
from PIL import ImageDraw

@tdec
def get_rcnn_preds(dataset_test, loaded_model):
    dictdf = {}
    for idx in range(len(dataset_test)):
        start_time = time.time()
        img, _, file = dataset_test[idx]
        label_boxes = np.array(dataset_test[idx][1]["boxes"])
        #put the model in evaluation mode
        loaded_model.eval()
        with torch.no_grad():
            prediction = loaded_model([img])
            #predstr = ''
            counter = 0
            for box, label, conf in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
                xmin = str(int(np.round(box[0].item(), 0)))
                ymin = str(int(np.round(box[1].item(), 0)))
                xmax = str(int(np.round(box[2].item(), 0)))
                ymax = str(int(np.round(box[3].item(), 0)))
                label = str(label.item())
                conf = float(conf.item())
                outputstr = label + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' '
                if outputstr is None: outputstr = ''
                print(file, outputstr, conf)
                #predstr += outputstr
                keyn = file + '_' + str(counter)
                dictdf[keyn] = [outputstr, conf]
                counter += 1
        print('{}s to predict'.format(np.round(time.time() - start_time, 1)))
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
        image = Image.fromarray(img.mul(255).permute(1, 2,0).byte().numpy())
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


def train_fastrcnn(num_epochs, model, optimizer, lr_scheduler, data_loader_train, data_loader_val, device, print_freq = 50, save_every = 10, output_weights_file = 'train/weights/model_1218_fastrcnn_{}e'):
    
    for idx, epoch in enumerate(range(num_epochs)):
        if (idx != 0) & (idx % save_every == 0):
            torch.save(model.state_dict(), output_weights_file.format(idx))
        # train for one epoch, printing every print_freq iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)