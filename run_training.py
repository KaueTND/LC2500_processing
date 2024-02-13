# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 13:04:53 2024

@author: kaueu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import sys

fold = sys.argv[1]

train_indices = np.load('indices/train_indices_fold'+fold+'.npy') 
val_indices   = np.load('indices/val_indices_fold'+fold+'.npy') 
test_indices  = np.load('indices/test_indices_fold'+fold+'.npy') 

X = np.load('X.npy')
y = np.load('y.npy')

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
#    Dropout(0.2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
 #   Dropout(0.2),
    Conv2D(256, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
  #  Dropout(0.3), 
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])
checkpoint_filepath = f"models/model_fold{fold}.h5"
model_checkpoint = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
# Define a ReduceLROnPlateau callback to drop the learning rate if val_loss gets stuck
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1)

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X[train_indices], 
                    y[train_indices],
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(X[val_indices], y[val_indices]),
                    callbacks=[model_checkpoint, reduce_lr]
                   )

model = load_model(checkpoint_filepath)

# Save the best model
#best_model.save("best_model.h5")


#model.save("models/model_cancer_"+fold+'.h5')
#model.save("models/model_cancer_"+fold)

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
#plt.show()
plt.savefig('plots/acc_'+fold+'.png')


plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
#plt.show()
plt.savefig('plots/loss_'+fold+'.png')

y_pred = model.predict(X[test_indices])
y_pred_binary = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y[test_indices], y_pred_binary)
conf_matrix = confusion_matrix(y[test_indices], y_pred_binary)

np.save('y_pred_'+fold,y_pred)

fpr, tpr, _ = roc_curve(y[test_indices], y_pred)
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
#plt.show()
plt.savefig('plots/roc_'+fold+'.png')

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')

