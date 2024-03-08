
# changes : !
#  changes to 'binary' not categorical
# epochs = 3 not 30


# Basic
import os
import random
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
import numpy as np
import pandas as pd

# visuals
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay

# Tensorflow
import tensorflow as tf

from keras.utils import to_categorical

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,MaxPooling2D,Dropout,Flatten,BatchNormalization,Conv2D
from keras.callbacks import ReduceLROnPlateau,EarlyStopping



# 1 ---------------------------------------------------Loading Images in a Dataframe
# region

all_entities_path = "../dogs-vs-cats/data set 2/"

# Get the list of subfolders (each subfolder name is considered as a label)
all_entities_names = os.listdir(all_entities_path)

filenames = []

# the data set folder contains subfolders, each subfolder in named cat,dog, etc
# the images in each subfolder are named as cat.1.jpg, cat.2.jpg, etc

percentage = 1  # Use only a percentage of the data to speed up the process

# Loop over each subfolder
for entity_name in all_entities_names:
    entity_path = os.path.join(all_entities_path, entity_name)
    entity_filenames = os.listdir(entity_path)
    # Shuffle the filenames
    random.shuffle(entity_filenames)
    # Select a percentage of the filenames
    entity_filenames = entity_filenames[:int(len(entity_filenames) * percentage)]
    # Add the filenames to the main list, prepend the subfolder name to each filename
    filenames.extend([os.path.join(entity_name, image) for image in entity_filenames])

file_labels = [x.split(os.sep)[0] for x in filenames]
data = pd.DataFrame({"filename": filenames, "label": file_labels})

# endregion

# ------------------------------------------------------- visualize the data
# region

# #  ---------------dogs photos
# plt.figure(figsize=(20,20)) # specifying the overall grid size
# plt.subplots_adjust(hspace=0.4)


# for i in range(10):
    
#     plt.subplot(1,10,i+1)    # the number of images in the grid is 10*10 (100)
#     filename = os.path.join(train_path, 'dog.' + str(i+1) + '.jpg')
#     image = imread(filename)
#     plt.imshow(image)
#     plt.title('Dog',fontsize=12)
#     plt.axis('off')

# plt.show(block = False)

# #  ---------------cats photos
# plt.figure(figsize=(20,20)) # specifying the overall grid size
# plt.subplots_adjust(hspace=0.4)


# for i in range(10):
    
#     plt.subplot(1,10,i+1)    # the number of images in the grid is 10*10 (100)
#     filename = os.path.join(train_path, 'cat.' + str(i+1) + '.jpg')
#     image = imread(filename)
#     plt.imshow(image)
#     plt.title('Cat',fontsize=12)
#     plt.axis('off')

# plt.show(block = False)

#  endregion
#

# 2 --------------------------------------------------- Train Test Split
# region

all_entities_names = data['label']
X_train, X_temp = train_test_split(data, test_size=0.2, stratify=all_entities_names, random_state = 42)
label_test_val = X_temp['label']
X_test, X_val = train_test_split(X_temp, test_size=0.5, stratify=label_test_val, random_state = 42)

print(" ")
print('The shape of train data',X_train.shape)
print('The shape of test data',X_test.shape)
print('The shape of validation data',X_val.shape)
print(" ")

# endregion

# ----------------------------------------------------visualize the test-train percentage
# region
# labels = ['Cat','Dog']

# label1,count1 = np.unique(X_train.label,return_counts=True)
# label2,count2 = np.unique(X_val.label,return_counts=True)
# label3,count3 = np.unique(X_test.label,return_counts=True)

# uni1 = pd.DataFrame(data=count1,index=labels,columns=['Count1'])
# uni2 = pd.DataFrame(data=count2,index=labels,columns=['Count2'])
# uni3 = pd.DataFrame(data=count3,index=labels,columns=['Count3'])


# plt.figure(figsize=(20,6),dpi=200)
# sns.set_style('darkgrid')

# font_size = 15

# plt.subplot(131)
# sns.barplot(data=uni1,x=uni1.index,y='Count1',palette='icefire',width=0.2).set_title('Training set',fontsize=font_size)
# plt.xlabel('Labels',fontsize=12)
# plt.ylabel('Count',fontsize=12)

# plt.subplot(132)
# sns.barplot(data=uni2,x=uni2.index,y='Count2',palette='icefire',width=0.2).set_title('validation set',fontsize= font_size)
# plt.xlabel('Labels',fontsize=12)
# plt.ylabel('Count',fontsize=12)


# plt.subplot(133)
# sns.barplot(data=uni3,x=uni3.index,y='Count3',palette='icefire',width=0.2).set_title('Testing set',fontsize= font_size)
# plt.xlabel('Labels',fontsize=12)
# plt.ylabel('Count',fontsize=12)

# plt.show(block = False)
# endregion 
# 3 --------------------------------------------------- Data Preparation
# region

image_size = 128
image_channel = 3
bat_size = 32

# Creating image data generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range = 15,
                                    horizontal_flip = True,
                                    zoom_range = 0.2,
                                    shear_range = 0.1,
                                    fill_mode = 'reflect',
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1)
test_datagen = ImageDataGenerator(rescale=1./255)

# Applying image data gernerator to train and test data

train_generator = train_datagen.flow_from_dataframe(X_train,
                                                directory = all_entities_path ,
                                                x_col= 'filename',
                                                y_col= 'label',
                                                batch_size = bat_size,
                                                target_size = (image_size,image_size),
                                                class_mode='categorical')
val_generator = test_datagen.flow_from_dataframe(X_val, 
                                                directory = all_entities_path ,
                                                x_col= 'filename',
                                                y_col= 'label',
                                                batch_size = bat_size,
                                                target_size = (image_size,image_size),
                                                shuffle=False,
                                                class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(X_test, 
                                                directory = all_entities_path ,
                                                x_col= 'filename',
                                                y_col= 'label',
                                                batch_size = bat_size,
                                                target_size = (image_size,image_size),
                                                shuffle=False,
                                                class_mode='categorical')

# endregion

# 4 --------------------------------------------------- Deep Learning Model
# region

model = Sequential()

# Input Layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape = (image_size,image_size,image_channel))) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Bloack 1 
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
# Block 2
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
# Block 3
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Fully Connected layers 
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Output layer
model.add(Dense(2, activation='softmax'))
# model.summary()

# endregion

# 5 --------------------------------------------------- Callbacks
# region
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',
                                            patience=2,
                                            factor=0.5,
                                            min_lr = 0.00001,
                                            verbose = 1)

early_stoping = EarlyStopping(monitor='val_loss',patience= 3,restore_best_weights=True,verbose=0)

# endregion

# 6 --------------------------------------------------- Model Compilation
# region
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

# endregion

# 7 --------------------------------------------------- Model Fitting
# region

print("x_train length: ",len(X_train))
print("x_test length: ",len(X_test))
print("batch size: ",bat_size)
print("steps_per_epoch: ",len(X_train) , " // " , bat_size , " = " , len(X_train) // bat_size)
print("validation_steps: ",len(X_test) , " // " , bat_size , " = " , len(X_test) // bat_size)
cat_dog = model.fit(train_generator,
                    validation_data = val_generator,          
                    callbacks=[early_stoping,learning_rate_reduction],
                    epochs = 30,
                    # data generator must generate at least steps_per_epochs * epochs batches
                    
                    steps_per_epoch = len(X_train) // bat_size,
                    validation_steps = len(X_test) // bat_size,
                   )

# endregion

# 8 --------------------------------------------------- Plot the results
# region
# plots for accuracy and Loss with epochs

error = pd.DataFrame(cat_dog.history)

plt.figure(figsize=(18,5),dpi=200)
sns.set_style('darkgrid')

plt.subplot(121)
plt.title('Cross Entropy Loss',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.plot(error['loss'])
plt.plot(error['val_loss'])

plt.subplot(122)
plt.title('Classification Accuracy',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.plot(error['accuracy'])
plt.plot(error['val_accuracy'])

plt.show(block=False)  # hosain : prevent the popup 


# endregion

# 9 --------------------------------------------------- Evaluation
# region 
# Evaluvate for train generator
loss,acc = model.evaluate(train_generator,batch_size = bat_size, verbose = 0)

print('The accuracy of the model for training data is:',acc*100)
print('The Loss of the model for training data is:',loss)

# Evaluvate for validation generator
loss,acc = model.evaluate(val_generator,batch_size = bat_size, verbose = 0)

print('The accuracy of the model for validation data is:',acc*100)
print('The Loss of the model for validation data is:',loss)

#  endregion

# 10 --------------------------------------------------- save the model
# region
model.save("model.keras")
# endregion

# 11 --------------------------------------------------- Prediction
# region
result = model.predict(test_generator,batch_size = bat_size,verbose = 0)

y_pred = np.argmax(result, axis = 1)

y_true = test_generator.labels

# Evaluvate
loss,acc = model.evaluate(test_generator, batch_size = bat_size, verbose = 0)

print('The accuracy of the model for testing data is:',acc*100)
print('The Loss of the model for testing data is:',loss)

# endregion

# 12 --------------------------------------------------- Classification Report
# region
all_entities_names =['Cat','Dog']
print(classification_report(y_true, y_pred,target_names=all_entities_names))

# endregion

# 13 --------------------------------------------------- Confusion Matrix
# region
confusion_mtx = confusion_matrix(y_true,y_pred) 
print("Confusion Matrix: \n",confusion_mtx)

f,ax = plt.subplots(figsize = (8,4),dpi=200)
sns.heatmap(confusion_mtx, annot=True, linewidths=0.1, cmap = "gist_yarg_r", linecolor="black", fmt='.0f', ax=ax,cbar=False, xticklabels=all_entities_names, yticklabels=all_entities_names)

plt.xlabel("Predicted Label",fontsize=10)
plt.ylabel("True Label",fontsize=10)
plt.title("Confusion Matrix",fontsize=13)

plt.show()
# endregion

