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
![image](https://user-images.githubusercontent.com/8759492/128956163-ebbb5a5c-5426-4594-b5c3-9d65149fc78b.png)
1. [October 2021] Run Inference using iPhone taking pictures / video from dashboard mount inside car to develop initial proof of concept 
1. [October 2021] Optional - Publish paper showing object detection results & model performance thus far
1. [After October 2021] Build iPhone app that can show real-time object detection in top half of screen and real-time mapping of identified distress relative to current user location as user is driving in bottom half of screen 

![image](https://user-images.githubusercontent.com/8759492/128957315-8cae1c19-deac-4e65-9ed9-5fca93acd62a.png)


#### Background Information

IEEE road distress page https://rdd2020.sekilab.global/


Example App View of Detected Road Distresses:

![image](https://user-images.githubusercontent.com/8759492/128953853-c481f587-efc5-4a58-a0b2-7d2ed1f987e1.png)

