# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 22:04:31 2024

@author: kaueu
"""

import numpy as np
from matplotlib import pyplot as plt
from rasterio.plot import reshape_as_image
import tensorflow as tf
import rasterio
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tf_keras_vis.utils import num_of_gpus

import cv2

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore, BinaryScore

import os
print(tf.__version__)


# list to store files
out_label = []
images = []
out_path = []
dir_path = 'C://Users//kaueu//Desktop//LC2500_filtered//'
# Iterate directory
# for path in os.listdir(dir_path+'Crop_dataset_X//'):
#     # check if current path is a file
#     if os.path.isfile(os.path.join(dir_path+'Crop_dataset_X//', path)):
#         out_label.append(float(path.replace('.tif','').split('x')[-1])>0.5)
#         out_path.append(dir_path+'output_gradcam//'+path.replace('.tif','.png'))
#         with rasterio.open(dir_path+'Crop_dataset_X//'+path) as src:
#             img = src.read()
#             image = reshape_as_image(img).astype('float64')
#             images.append(image)

fold = 5
Xa = np.load(dir_path+'X.npy')
y = np.load(dir_path+'y.npy')
test_indices = np.load(dir_path+'indices/test_indices_fold'+str(fold)+'.npy')
paths = np.load(dir_path+'paths.npy')
model = load_model('C://Users//kaueu//Desktop//LC2500_filtered//models/model_fold'+str(fold)+'.h5',compile=False) 
model.summary()
replace2linear = ReplaceToLinear()
gradcam = Gradcam(model,
                  model_modifier=replace2linear,
                  clone=True)
predx = np.load(dir_path+'y_pred_'+str(fold)+'.npy')
for i in range(0,len(test_indices),10):
    
    X         = Xa[test_indices][i:i+10]
    out_label = y[test_indices][i:i+10]
    out_label = [bool(bit) for bit in out_label]
    out_path  = paths[test_indices][i:i+10]
    pred = predx[i:i+10]

    
    score = BinaryScore(out_label)
    
    cam = gradcam(score,
                  X,
                  penultimate_layer=-1)
    
    for i, title in enumerate(pred):
        plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
    
        # Plot the original image on the left
        plt.subplot(1, 2, 1)
        plt.title("Original Image", fontsize=16)
        plt.imshow(X[i])
        plt.axis('off')
        grayscale_image = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)
        
        #grayscale_image = rgb2gray(X[i])
        # Plot the image with heatmap on the right
        plt.subplot(1, 2, 2)
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        plt.title(str(title) + ' - ' + str(out_label[i]), fontsize=16)
        #plt.imshow(grayscale_image)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)  # overlay
        plt.axis('off')
    
        plt.tight_layout()
        #plt.show()
        plt.savefig(dir_path+'Plots/'+out_path[i])
