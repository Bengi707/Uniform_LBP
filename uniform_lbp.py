# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:07:13 2022

@author: Bengi
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def get_pixel(img, center, x, y):
     
    new_value = 0
     
    if img[x][y] >= center:
        new_value = 1
             
    return new_value

 
# Function for calculating LBP
def lbp_calculate(image):
    
    img = np.zeros_like(image)
    
    for x in range(0, image.shape[0]-2):
        for y in range(0, image.shape[1]-2):
    
            center = image[x+1][y+1]
           
            val_ar = []
              
            # top_left
            
            val_ar.append(get_pixel(image, center, x-1, y-1))
            
            # top
            val_ar.append(get_pixel(image, center, x-1, y))
              
            # top_right
            val_ar.append(get_pixel(image, center, x-1, y+1))
              
            # right
            val_ar.append(get_pixel(image, center, x, y+1))
              
            # bottom_right
            val_ar.append(get_pixel(image, center, x+1, y+1))
              
            # bottom
            val_ar.append(get_pixel(image, center, x+1, y))
              
            # bottom_left
            val_ar.append(get_pixel(image, center, x+1, y-1))
              
            # left
            val_ar.append(get_pixel(image, center, x, y-1))
               
            # Now, we need to convert binary
            # values to decimal   
            
            count = 0
            #check if it's uniform
            for i in range(0,len(val_ar)-1):
                if(val_ar[i] != val_ar[i+1]):
                    count+=1
            
            if count>2:# not uniform
                val = 51
                
            else:
                
                val=0
                for i in range(len(val_ar)):
                    val += (val_ar[i] * (pow(2,7-i)))
            
            img[x+1][y+1]= val
        
        
    return img

def calcHistogram(img):
    
    # calculate histogram here
    img_height = img.shape[0]
    img_width = img.shape[1]
    histogram = np.zeros(256, np.int32) 
    
    for i in range(0, img_height):
        for j in range(0, img_width):
            histogram[img[i][j]] +=1
    
    return histogram  

def normalize_img(hist,height,width):
    
    total_pixels = height*width
    normalized= np.zeros(256,np.float) 
    
    for i in range(256):
        normalized[i] = hist[i] / total_pixels

    return normalized 

def manhattan_dist(hist_1,hist_2):
    
    sum_ = 0
    
    for i in range(256):
        
        sum_ += abs(hist_1[i]-hist_2[i])
    
    return sum_

def most_similar_3(list1):
    
    
    MIN=1000
    min_values = []
    
    
    for i in range(3):
        
        MIN=1000
        
        for j in range(len(list1)):
            
            if list1[j][0] < MIN:
                
                MIN=list1[j][0]
                list1[j][0]=1000
                path = list1[j][1]
        
                
        min_values.append([MIN,path])
        
    return min_values
    

train = []
directory = './train'

for file_name in os.listdir(directory):
        sub_dir_path = directory + '/' + file_name
        if (os.path.isdir(sub_dir_path)):
            for image_name in os.listdir(sub_dir_path):
                if image_name[-4:] == '.jpg':
                    path = sub_dir_path + '/' + image_name
                    print(path)
                    
                    img_bgr = cv2.imread(path)
                       
                    height, width, _ = img_bgr.shape
                       
                    # We need to convert RGB image 
                    # into gray one because gray 
                    # image has one channel only.
                    
                    
                    img_gray = img_bgr[:,:,0]*0.11 + img_bgr[:,:,1]*0.59 + img_bgr[:,:,2]*0.3
                    
                    img_gray = img_gray.astype(int)
                    
                    # Create a numpy array as 
                    # the same height and width 
                    # of RGB image
                    
                    img_lbp=lbp_calculate(img_gray)
            
                
                    #Calculating the histogram of original
                    # and lbp uniform images
                
                    #hist =calcHistogram(img_gray)
                    hist2 =calcHistogram(img_lbp)
                    
                    """
                    Intensity = np.arange(0,256,1)       
                
                    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
                    fig.tight_layout()
                
                    plt.subplot(2, 1, 1)
                    plt.bar(Intensity,hist,color="maroon",width = 1)
                    plt.xlabel("Intensity")
                    plt.title('Original Image')
                    plt.ylabel('Frequency')
                
                
                    plt.subplot(2, 1, 2)
                    plt.bar(Intensity,hist2,color="maroon",width = 1)
                    plt.title('LBP Uniform Image')
                    plt.xlabel('Intensity')
                    plt.ylabel('Frequency')
                
                    plt.show()
                    """ 
                    #normalize image in range (0,1)
                    normalized=normalize_img(hist2,height,width)
                    
                    train.append([normalized,path])
                    
             
directory = './testRaporaEklenecek'                
for file_name in os.listdir(directory):
        
    
    if file_name[-4:] == '.jpg':

        path = directory + '/' + file_name
        img_bgr = cv2.imread(path)
           
        height, width, _ = img_bgr.shape
           
        # We need to convert RGB image 
        # into gray one because gray 
        # image has one channel only.
        img_gray = img_bgr[:,:,0]*0.11 + img_bgr[:,:,1]*0.59 + img_bgr[:,:,2]*0.3
           
        # Create a numpy array as 
        # the same height and width 
        # of RGB image
        img_gray = img_gray.astype(int)
        img_lbp=lbp_calculate(img_gray)
        
        
        #Calculating the histogram of original
        # and lbp uniform images
        
        hist =calcHistogram(img_lbp)
        normalized_test=normalize_img(hist,height,width)
        
        dist = []
        
        for i in range(len(train)):
            #distance ve path tut
            dist.append([manhattan_dist(train[i][0],normalized_test),train[i][1]])
            
        min_values = most_similar_3(dist)
        
        
        print("---")
        print("Original:",path)
        print("Similar 1:",min_values[0][1], " Distance:",min_values[0][0])
        print("Similar 2:",min_values[1][1], " Distance:",min_values[1][0])
        print("Similar 3:",min_values[2][1], " Distance:",min_values[2][0])
        print("---")
        
        
        img_first = cv2.imread(min_values[0][1], 1)
        img_sec = cv2.imread(min_values[1][1], 1)
        img_third = cv2.imread(min_values[2][1], 1)

        
        fig = plt.figure(figsize=(5, 5))
        # Adds a subplot at the 1st position
        fig.add_subplot(2, 2, 1)
          
        # showing image
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB )
        plt.imshow(img_bgr)
        plt.axis('off')
        plt.title("Original")
        
          
        # Adds a subplot at the 2nd position
        fig.add_subplot(2, 2, 2)
          
        # showing image
        img_first = cv2.cvtColor(img_first, cv2.COLOR_BGR2RGB )
        plt.imshow(img_first)
        plt.axis('off')
        plt.title("Similar 1")

          
        # Adds a subplot at the 3rd position
        fig.add_subplot(2, 2, 3)
         
        # showing image
        img_sec = cv2.cvtColor(img_sec, cv2.COLOR_BGR2RGB )
        plt.imshow(img_sec)
        plt.axis('off')
        plt.title("Similar 2")
          
        # Adds a subplot at the 4th position
        fig.add_subplot(2, 2, 4)
          
        # showing image
        img_third = cv2.cvtColor(img_third, cv2.COLOR_BGR2RGB )
        plt.imshow(img_third)
        plt.axis('off')
        plt.title("Similar 3")
    

                    