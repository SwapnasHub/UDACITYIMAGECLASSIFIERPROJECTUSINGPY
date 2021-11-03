# Perform necessary imports
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from PIL import Image
import json

IMAGE_SIZE = 224

# Implementation of the process_image function. The image returned by the process_image function is a NumPy
# array with shape (224, 224, 3) but the model expects the input images to be of shape (1, 224, 224, 3)
#
# @param image
# The image to be processed
def process_image(image): 
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
    return image.numpy()

# Implementation of the predict function
#
# @param image_path
# Path of the image file
#
# @param model
# Saved Kerasmodel
# 
#  @param top_k
# Number of top matching probabilities
#
# @return
# Array of top_k matching probablities
# Array of top_k matching classes
#
def predict(image_path, model, top_k):
    
    if top_k < 1:
        top_k = 1
    
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    expanded_image = np.expand_dims(image, axis=0)
    probabilities = model.predict(expanded_image)
    
    top_values, top_indices = tf.nn.top_k(probabilities, k=top_k)
    #print("These are the top probabilities: ", top_values.numpy()[0])
    #print("These are the top indices: ", top_indices.numpy()[0])
    
    # retrieve the classes from the corresponding indices
    top_classes = []
    for value in top_indices.numpy()[0]:
        top_classes.append(class_names[str(value + 1)])
    
    # return the top_k probabilities and the top_k classes
    return top_values.numpy()[0], top_classes


if __name__ == '__main__':
    print('Running predict.py')
    
    # Bulding the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type = str)
    parser.add_argument('saved_model', type = str)
    parser.add_argument('--top_k', default = 5, type = int)
    parser.add_argument('--category_names', default = 'label_map.json', type = str)
    
    # Load the arguments
    args = parser.parse_args()
    
    # Check and print the arguments
    print(args)
    
    # Print individual arguments
    print('arg1 =', args.image_path)
    print('arg2 =', args.saved_model)
    print('top_k =', args.top_k)
    print('category_names =', args.category_names)
    
    # Load the given Keras model
    model = load_model(args.saved_model, custom_objects={'KerasLayer':hub.KerasLayer})
    #model.summary()
    
    # Load top_k. If top_k is not supplied, set its default value as 3
    top_k = args.top_k
    if top_k is None: 
        top_k = 3
    
    # Load category_names, if provided in the arguments
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        
    # Predict the given image using the supplied model
    probs, classes = predict(args.image_path, model, top_k)
    print(probs)
    print(classes)