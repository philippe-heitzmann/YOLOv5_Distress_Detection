## Object Detection Next Steps & Roadmap

#### Philippe Next Steps:
1. Investigate remaining imgaug augmentation transformations available and select subset most appropriate for task 
1. Investigate setting up Multicore image augmentation processing to speed up sequential augmentations 
1. Finalize augmentation pipeline and augment full image set to start model training 
1. Push final XML to TXT annotation file conversion pipeline code to shared repo - **DONE**

#### Data Pipeline Structure 

1. XML_to_TXT_Annotation_Conversion_Pipeline.ipynb to convert XML annotation files to TXT for YOLOv5 use 
1. A01 - Load and Augment an Image.ipynb to define augmentations to apply to input images

### Development Roadmap:
1. [October 2021] Build desktop app that can take image or video files with embedded EXIF metadata as input and return geolocated map showing where distresses occured 
2. [October 2021] Run Inference on 
3. Train and Deploy object detection models to 
4. [After October 2021] Build iPhone app that 

![image](https://user-images.githubusercontent.com/8759492/128956163-ebbb5a5c-5426-4594-b5c3-9d65149fc78b.png)


#### Background Information

IEEE road distress page https://rdd2020.sekilab.global/


Example App View of Detected Road Distresses:

![image](https://user-images.githubusercontent.com/8759492/128953853-c481f587-efc5-4a58-a0b2-7d2ed1f987e1.png)

