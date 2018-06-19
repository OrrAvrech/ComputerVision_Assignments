# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:22:58 2018

@author: orrav
"""

#%% Setup
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
import numpy as np

import os
import math

# Main slim library
from tensorflow.contrib import slim

#%% Displaying bird images
bird0 = mpimg.imread(os.path.join('..', 'Data', 'birds', 'bird_0.jpg'))
bird1 = mpimg.imread(os.path.join('..', 'Data', 'birds', 'bird_1.jpg'))
plt.figure()
plt.subplot(121); plt.imshow(bird0); plt.axis('off')

plt.subplot(122); plt.imshow(bird1); plt.axis('off')
plt.show()

#%% Build graph and predictions
# from slim library
import vgg_preprocessing
import imagenet
import vgg

# global
image_size  = vgg.vgg_16.default_image_size

def build_graph():    
    image_placeholder = tf.placeholder(tf.float32, shape=[None, None, 3])
    processed_image = vgg_preprocessing.preprocess_image(image_placeholder, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
    probabilities = tf.nn.softmax(logits)
    return probabilities, image_placeholder, end_points

def init_variables():
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join('..', 'Data', 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
    return init_fn

def predict(images, probabilities, image_placeholder):
    np_probabilities = sess.run(probabilities, feed_dict={image_placeholder: images})
    return np_probabilities

def showImage(np_image_raw, np_probabilities):
    names = imagenet.create_readable_names_for_imagenet_labels()
    np_probability = np_probabilities[0, :]
    sorted_inds = [j[0] for j in sorted(enumerate(-np_probability), key=lambda x:x[1])]
    
    plt.figure()
    plt.imshow(np_image_raw.astype(np.uint8))
    plt.axis('off')
    plt.show()

    for k in range(5):
            index = sorted_inds[k]
            # Shift the index of a class name by one. 
            print('Probability %0.2f%% => [%s]' % (np_probability[index] * 100, names[index+1]))
            
tf.reset_default_graph()
probabilities, image_placeholder, end_points = build_graph()
init_fn = init_variables()
sess = tf.InteractiveSession()
init_fn(sess)
images = [bird0, bird1]
batch_size = len(images)
for i in range(batch_size):
    np_probabilities = predict(images[i], probabilities, image_placeholder)
    showImage(images[i], np_probabilities)
    
#%% Burrito image and transformations
from skimage import transform
from skimage import color
from skimage import filters
    
# image and some transformations
orig_burrito          = mpimg.imread(os.path.join('..', 'Data', 'burrito.jpg'))
g_tform_burrito       = transform.rotate(orig_burrito, angle=90)*255
color_tform_burrito   = color.convert_colorspace(orig_burrito, 'RGB', 'HSV')*255
filtered_burrito      = filters.gaussian(orig_burrito, sigma=3, multichannel=False)*255

burrito_images = [orig_burrito, g_tform_burrito, color_tform_burrito, filtered_burrito]
batch_size = len(burrito_images)
for i in range(batch_size):
    np_probabilities = predict(burrito_images[i], probabilities, image_placeholder)
    showImage(burrito_images[i], np_probabilities)
    
#%% Visualize filters and their response
def normalize_image(arr):
    arr = arr.astype('float')
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def plot_conv_weights(weights, indices, images):
    """plots given filters and their responses to a given input.
    Args:
        weights: variable name
        indices: vector of indices smaller than num_filters
        images: list of input images
    """
    w = sess.run(weights)
    num_chosen_filters = len(indices)
            
    # Create sub-plots for each filter.
    fig, axes = plt.subplots(1, num_chosen_filters)

    for i, ax in enumerate(axes.flat):
        if i < num_chosen_filters:
            filter_img = w[:, :, :, indices[i]]
            normalized_filter = normalize_image(filter_img)
            ax.imshow(normalized_filter.astype(np.uint8), interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])  
    plt.show() 
        
    # Build graph for filter respones    
    processed_image = vgg_preprocessing.preprocess_image(image_placeholder, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    conv = tf.nn.conv2d(processed_images, w, strides=[1,1,1,1], padding='SAME') 
    
    # Eval and plot responses for each image in images
    batch_size = len(images)
    for j in range(batch_size):    
        res_sess = tf.Session()
        res_sess.run(tf.global_variables_initializer())
        filterResponse = res_sess.run(conv, feed_dict={image_placeholder: images[j]})
        
        fr_min = np.min(filterResponse)
        fr_max = np.max(filterResponse)
        
        fig, axes = plt.subplots(1, num_chosen_filters)
        for i, ax in enumerate(axes.flat):
            if i < num_chosen_filters:
                filter_res = filterResponse[0, :, :, indices[i]]
                ax.imshow(filter_res, vmin=fr_min, vmax=fr_max, cmap='viridis')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show() 
        
with tf.variable_scope('vgg_16', reuse=True):
    weights = tf.get_variable('conv1/conv1_1/weights')
    indices = [52, 21]
    burrito_transforms = [g_tform_burrito, color_tform_burrito, filtered_burrito]
    plot_conv_weights(weights, indices, burrito_transforms)
    
#%% FC-layer feature extraction
    features = end_points['vgg_16/fc7']
dog_features_mat = []
cat_features_mat = []
path = os.path.join('..', 'Data')
first = 0
for file in os.listdir(os.path.join(path, 'dogs')):
    dog_img = mpimg.imread(os.path.join(path, 'dogs', file))
    dog_features_vec = sess.run(features, feed_dict={image_placeholder: dog_img})
    dog_features_vec = dog_features_vec[0,0,0,:]
    dog_features_vec = np.expand_dims(dog_features_vec, axis=0)
    if first == 0:
        dog_features_mat = dog_features_vec
        first += 1
    else:
        dog_features_mat = np.append(dog_features_mat, dog_features_vec, axis=0)

first = 0
for file in os.listdir(os.path.join(path, 'cats')):
    cat_img = mpimg.imread(os.path.join(path, 'cats', file))
    cat_features_vec = sess.run(features, feed_dict={image_placeholder: cat_img})
    cat_features_vec = cat_features_vec[0,0,0,:]
    cat_features_vec = np.expand_dims(cat_features_vec, axis=0)
    if first == 0:
        cat_features_mat = cat_features_vec
        first += 1
    else:
        cat_features_mat = np.append(cat_features_mat, cat_features_vec, axis=0)
        
from sklearn.decomposition import PCA

fig = plt.figure()
ax = fig.add_subplot(111)

features_mat = np.append(dog_features_mat, cat_features_mat, axis=0)
features_reduced = PCA(n_components=2).fit_transform(features_mat)
ax.scatter(features_reduced[0:10,0], features_reduced[0:10,1], color='b', label='dog')
ax.scatter(features_reduced[10:20,0], features_reduced[10:20,1], color='r', label='cat')
plt.legend(loc='lower left')
plt.show()