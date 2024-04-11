import json
import os
import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import efficientnet.tfkeras as enet
from sklearn import metrics
import itertools

# Define file paths
file_path = 'C:\\Users\\Lenovo\\Downloads\\Kaggle Plant Leaf Dataset (1)\\Kaggle Plant Leaf Dataset\\data'
splitted_folder = '../output/corn-or-maize-leaf-disease-dataset/splitted_folder'

# Function to split data into train, test, and validation sets
def train_test_valid(train_size=0.6, test_size=0.2, val_size=0.2, images_folder=file_path, splitted_folder=splitted_folder):
    train_size = train_size
    test_size = test_size
    val_size = val_size
    input_folder = images_folder
    output_folder = splitted_folder
    splitfolders.ratio(input_folder, output_folder, seed=1337, ratio=(train_size, test_size, val_size), group_prefix=None)

# Function for data preprocessing
def data_pre_processing(valid_split=0, input_size=(260, 260), image_color='rgb', batch_size=32, shuffle=True):
    train_gen = ImageDataGenerator(rescale=1/255.0, validation_split=valid_split, fill_mode='nearest', rotation_range=40, horizontal_flip=True)
    validation_gen = ImageDataGenerator(rescale=1/255.0, validation_split=valid_split)
    test_gen = ImageDataGenerator(rescale=1/255.0)

    train_data = train_gen.flow_from_directory(directory='../output/corn-or-maize-leaf-disease-dataset/splitted_folder/train', target_size=input_size, color_mode=image_color, batch_size=batch_size, shuffle=shuffle, class_mode='categorical')
    test_data = test_gen.flow_from_directory(directory='../output/corn-or-maize-leaf-disease-dataset/splitted_folder/test', target_size=input_size, color_mode=image_color, batch_size=batch_size, shuffle=shuffle, class_mode='categorical')
    valid_data = validation_gen.flow_from_directory(directory='../output/corn-or-maize-leaf-disease-dataset/splitted_folder/val', target_size=input_size, color_mode=image_color, batch_size=batch_size, shuffle=shuffle, class_mode='categorical')

    return train_data, test_data, valid_data

# Define the hybrid model architecture
def configure_model():
    inputs_1 = tf.keras.Input(shape=(260, 260, 3))
    mymodel = enet.EfficientNetB2(input_shape=(260, 260, 3), include_top=False, weights='imagenet')
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(mymodel.output)
    x = tf.keras.layers.Flatten()(x)
    predictors = tf.keras.layers.Dense(4, activation='softmax', name='Predictions')(x)
    final_model = Model(mymodel.input, outputs=predictors)
    return final_model

# Function to configure the model
def model(new_model, layers_num=1, trainable=False):
    for layer in new_model.layers[:layers_num]:
        layer.trainable = trainable
    return new_model

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 3:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Compile the final model
final_model = configure_model()
final_model = model(final_model)
opt = tf.keras.optimizers.Adam(0.0001)
final_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
def callbacks(patience=2):
    checkpoint = tf.keras.callbacks.ModelCheckpoint('my_model.weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.001)
    lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callbacks_list = [checkpoint, early, lr]
    return callbacks_list

callbacks = callbacks()

# Class weights calculation
train, _, _ = data_pre_processing(valid_split=0.2)
counter = Counter(train.classes)
max_val = float(max(counter.values()))
class_weights1 = {class_id: max_val/num_images for class_id, num_images in counter.items()}

# Train the model
train_data, _, validation = data_pre_processing(valid_split=0.2)
hist = final_model.fit(train_data, epochs=1000, validation_data=validation, callbacks=callbacks, class_weight=class_weights1)

# Plot accuracy
plt.figure(figsize=(10, 10))
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["accuracy", "Validation Loss"])
plt.show()
