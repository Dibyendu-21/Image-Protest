# Detection of Protest Leader and Perceived Violence Estimation from Social Media Images using Deep Learning
Implementation of the model used in the paper Protest Activity Detection and Perceived Violence Estimation from Social Media Images
(Springers) by Dibyendu Biswas, Jyoti Prakash Singh.

## Table of Contents
* [Motivation](#motivation)
* [Technologies](#technologies) 
* [Protest Image Dataset](#protest-image-dataset )
* [](#models)

## Motivation


![](Images/caught_1.png)

## Technologies

* OpenCV 
* Numpy
* Keras
* Pandas
* Matplotlib
* Scikit-Learn

## Protest Image Dataset 
The dataset consist of 23,294 negative Images and 11,659 positive Images of Protest. This dataset was created by UCLA and is in their private repository. A new dataset consisting of 491 images of protest containg protest Leader and Non-Protest Leader was created by me. Images with protest Leader was lablled with a bounding box drawn around the protest leader using a manual annaotaing tool. 

### Dataset Statistics
No. of images: 35,444
No. of protest images: 12,150

### Protest & Visual Attributes

![](Images/Positive_Rate.png)

### Data Preprocessing

Since there was significant imbalance ine the dataset, various Image Augmentation techniques were employed to increase the count of the positive classes by 2. With this Augmentation the total count of positive classes increased to 24,300 and negative classes were maintained at 23,294.


![](Images/Augmentation.PNG)
