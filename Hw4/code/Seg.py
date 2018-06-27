# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------- #
#%%

# =============================================================================
# Computer Vision - Homework 2 Task 3
# 
# Orr Avrech - 302857065
# Opher Bar Nathan - 302188628
# =============================================================================


#%%

# =============================================================================
# import relevant packages
# =============================================================================

import cv2
import numpy as np
import os

import argparse
import tarfile
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image

import tensorflow as tf

import pickle


#%%
# =============================================================================
# working directory
# =============================================================================
clips_dir = os.path.join('..','clips')
Data_dir = os.path.join('..','data')
seg_object = 'person'
video_name = 'bahai_4k.mp4'

doCapture = True # True

#%%

# =============================================================================
# in this section the video is captured to frames:
# =============================================================================
if doCapture:

    video_path = os.path.join(clips_dir,video_name)
    
    cap = cv2.VideoCapture(video_path)
    
    try:
        if not os.path.exists(os.path.join(Data_dir,'frames_' + video_name[0:-4] )):
            os.makedirs(os.path.join(Data_dir,'frames_' + video_name[0:-4]))
        
    except OSError:
            print ('Error: Creating directory of data')
    
    currentFrame = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            break
    
        # Saves image of the current frame in jpg file
        name = 'frame' + str(currentFrame) + '.jpg'
        frame_path = os.path.join(Data_dir,'frames_' + video_name[0:-4],name)
        
        print ('Creating...' + name)
        cv2.imwrite(frame_path, frame)
        
        currentFrame += 1
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

#%% 
# =============================================================================
# This section includes class and functions for DeepLab segmentation netweork:
# =============================================================================

class DeepLabModel(object):
  # Class to load deeplab model and run inference.

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    #"""Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()
    
    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
   # Runs inference on a single image.

   # Inputs: image: PIL.Image object, raw input image.
   # Output:
   #  resized_image: RGB image resized from original input image.
   #  seg_map: Segmentation map of `resized_image`.
    
   width, height = image.size
   resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
   target_size = (int(resize_ratio * width), int(resize_ratio * height))
   resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
   batch_seg_map = self.sess.run(
       self.OUTPUT_TENSOR_NAME,
       feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
   seg_map = batch_seg_map[0]
   return resized_image, seg_map

def create_pascal_label_colormap():
     # Creates a label colormap used in PASCAL VOC segmentation benchmark.
     # Output:
     #  A Colormap for visualizing segmentation results.

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
          colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

    return colormap


def label_to_color_image(label):
  # Adds color defined by the dataset colormap to the label.
  # Input:
  #  label: A 2D array with integer type, storing the segmentation label.

  # Output:
  # result: A 2D array with floating type. The element of the array
  #    is the color indexed by the corresponding element in the input label
  #    to the PASCAL color map.

    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_seg_object(resized_im, seg_mask_object,model_name,section):
    # Visualizes input image, segmentation map and overlay view.
    # only for the object
    fig = plt.figure(figsize=(30, 9))
    grid_spec = gridspec.GridSpec(1, 3)
    
    plt.subplot(grid_spec[0])
    plt.imshow(resized_im)
    plt.axis('off')
    plt.title('input image',fontsize = 22)
    
    plt.subplot(grid_spec[1])
    plt.imshow(seg_mask_object)
    plt.axis('off')
    plt.title('segmentation map',fontsize = 22)
    
    plt.subplot(grid_spec[2])
    plt.imshow(resized_im)
    plt.imshow(seg_mask_object, alpha=0.6)
    plt.axis('off')
    plt.title('segmentation overlay',fontsize = 22)
    plt.grid('off')
    plt.show()
    
    fig.suptitle('Task 3 sec ' + section +' - Frame Object Segmentation ' + model_name + ' model',fontsize = 26)
    
    fig.savefig(os.path.join(Data_dir,'Task3_' + section + '_frame_object_segmentation_' + model_name + '.jpg'),bbox_inches='tight')
    plt.close(fig)


def vis_seg_general(resized_im, seg_mask ,seg_image, model_name,section):
    # Visualizes input image, segmentation map and overlay view.
    # for general segmentation mask (not onlt for the object - chair)
    fig = plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
        
    plt.subplot(grid_spec[0])
    plt.imshow(resized_im)
    plt.axis('off')
    plt.title('input image',fontsize = 22)
    
    plt.subplot(grid_spec[1])
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map',fontsize = 22)
        
    plt.subplot(grid_spec[2])
    plt.imshow(resized_im)
    plt.imshow(seg_image, alpha=0.75)
    plt.axis('off')
    plt.title('segmentation overlay',fontsize = 22)
        
    unique_labels = np.unique(seg_mask)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
    FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()
          
    fig.suptitle('Task 3 sec ' + section + ' - Frame General Segmentation ' + model_name + ' model',fontsize = 26)
    fig.savefig(os.path.join(Data_dir,'Task3_' + section + '_frame_general_segmentation_' + model_name + '.jpg'),bbox_inches='tight')
    plt.close(fig)

     
def segmentation_image(image_path,seg_object):
    #   Inferences DeepLab model and visualizes result.
    #   Input: image path
    #   Output: 
    original_im = Image.open(image_path)
    resized_im, seg_mask_general = MODEL.run(original_im)

    seg_image = label_to_color_image(seg_mask_general).astype(np.uint8)
    
    # deeplab network make labels for the segments - 
    # we are looking only for the chair    
    chair_seg_ind = (np.where(LABEL_NAMES == seg_object))[0][0]
    
    x,y = np.where(seg_mask_general == chair_seg_ind)
    k,s = np.where(seg_mask_general != chair_seg_ind)
    
    seg_mask_object = np.zeros(seg_mask_general.shape)
    
    seg_mask_object[x,y] = 255
    seg_mask_object[k,s] = 0
     
    return resized_im, seg_mask_object, seg_mask_general, seg_image

#%%
# =============================================================================
# Labels for the DeepLab segmentation network:
# =============================================================================
    
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'])
 
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


#%% segmentation for wanted frames
seg_object = 'person'
frames_path = os.path.join(Data_dir,'frames_' + video_name[0:-4])
frames_to_seg = np.arange(400,1280)

Checkpoint_weight_path = 'D:\\DeepLab\\deeplabv3_pascal_trainval_2018_01_04.tar.gz'
#checkpoint_path = 'D:\GoogleDrive\STUDIES\PythonProj\CV_HW2\Data'
#Checkpoint_weight_path = os.path.join(checkpoint_path,'deeplabv3_pascal_trainval_2018_01_04.tar.gz')
MODEL = DeepLabModel(Checkpoint_weight_path)
print('Xception based model loaded successfully!')
model_name = 'Xception'



frame = 'frame0.jpg'
image_path = os.path.join(frames_path,frame)
resized_im, seg_mask_object, seg_mask_general, seg_image = segmentation_image(image_path,seg_object)


ap_xception = argparse.ArgumentParser()
ap_xception.add_argument("-ext", "--extension", required=False, default='jpg', help="extension name. default is 'jpg'.")
ap_xception.add_argument("-o", "--output", required=False, 
                default=os.path.join(Data_dir,'Task3_sec5 - chair_seg_vid_' + model_name + '.mp4'), 
                help="output video file")

args_Xception = vars(ap_xception.parse_args())
ext = args_Xception['extension']
output = args_Xception['output']
   
height, width = seg_mask_object.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output, fourcc, 25.0, (width, height))


frames_num = []
for frame in os.listdir(frames_path):
    if frame.endswith(ext):
        frame_num = frame[:-4]
        frame_num = frame_num[5:]
        frames_num.append(frame_num)
        
frames_num.sort(key=int)
frames_num = frames_num[400:1291]


frames = []
for num in frames_num:
    f = 'frame' + num + '.jpg'
    frames.append(f)

frames_list_resized_im = list()
frames_list_seg_mask_object = list()
frames_list_seg_mask_general = list()
frames_list_seg_image = list()


for frame in frames:
    
    frame_path = os.path.join(frames_path,frame)
    resized_im, seg_mask_object, seg_mask_general, seg_image = segmentation_image(frame_path,seg_object)
    
    frames_list_resized_im.append(seg_mask_object)
    frames_list_seg_mask_object.append(seg_mask_object)
    frames_list_seg_mask_general.append(seg_mask_object)
    frames_list_seg_image.append(seg_mask_object)
    
    print(frame + ' is in process')
    # deepLab returns few segmentations, 
    # we take only the relevant for us (chair)

    k,s = np.where(seg_mask_object == 0)
   
    segmentation_frame = np.array(resized_im)
    
    segmentation_frame[k,s,:] = [0,0,0]
           
    out.write(cv2.cvtColor(segmentation_frame, cv2.COLOR_RGB2BGR)) # Write out frame to video
    
# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))


with open('frames_list_resized_im.pkl', 'wb') as f:
    pickle.dump(frames_list_resized_im, f)
with open('frames_list_seg_mask_object.pkl', 'wb') as f:
    pickle.dump(frames_list_seg_mask_object, f)
with open('frames_list_seg_mask_general.pkl', 'wb') as f:
    pickle.dump(frames_list_seg_mask_general, f)
with open('frames_list_seg_image.pkl', 'wb') as f:
    pickle.dump(frames_list_seg_image, f)


#%%
# =============================================================================
#  Loading model and Xception based model checkpoint 
#  (for deeplab segmentation network):
# =============================================================================

Checkpoint_weight_path = 'D:\\DeepLab\\deeplabv3_pascal_trainval_2018_01_04.tar.gz'
#checkpoint_path = 'D:\GoogleDrive\STUDIES\PythonProj\CV_HW2\Data'
#Checkpoint_weight_path = os.path.join(checkpoint_path,'deeplabv3_pascal_trainval_2018_01_04.tar.gz')
MODEL = DeepLabModel(Checkpoint_weight_path)
print('Xception based model loaded successfully!')

model_name = 'Xception'


#%%
# =============================================================================
# Section 3
# making segmantation for one frame
# and display result
# =============================================================================

frames_path = os.path.join(Data_dir,'frames_' + video_name[0:-4])

#frame = 'frame250.jpg'
frame = 'frame0.jpg'
image_path = os.path.join(frames_path,frame)

resized_im, seg_mask_object, seg_mask_general, seg_image = segmentation_image(image_path,seg_object)

vis_seg_object(resized_im,seg_mask_object,model_name,'3')
vis_seg_general(resized_im,seg_mask_general,seg_image,model_name,'3')


#%%
# =============================================================================
#  For section 4 - using other model
#  Loading model and MobileNet based model checkpoint 
#  (for deeplab segmentation network):
# =============================================================================

Checkpoint_weight_path = 'D:\\DeepLab\\deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz'
#checkpoint_path = 'D:\GoogleDrive\STUDIES\PythonProj\CV_HW2\Data'
#Checkpoint_weight_path = os.path.join(checkpoint_path,'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz')

#Checkpoint_weight_path = os.path.join(Data_dir,'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz')
MODEL = DeepLabModel(Checkpoint_weight_path)
print('MobileNet based model loaded successfully!')

model_name = 'MobileNet'
#image_path = os.path.join(Data_dir,'anna.jpg')
resized_im, seg_mask_object, seg_mask_general, seg_image = segmentation_image(image_path,seg_object)

vis_seg_object(resized_im,seg_mask_object,model_name,'4')
vis_seg_general(resized_im,seg_mask_general,seg_image,model_name,'4')

#resized_im.save(os.path.join(Data_dir,'anna_re.png'))
#seg_mask = Image.fromarray(seg_mask_object)
#seg_mask = seg_mask.convert('RGB')
#seg_mask.save(os.path.join(Data_dir,'anna_mask.png'))
#
#size = 1000, 1320
#resized_im_new = resized_im.resize((1000,1320), Image.ANTIALIAS)
#resized_im_new.save(os.path.join(Data_dir,'anna_re.png'))
#
#seg_mask_new = seg_mask.resize((1000, 1320), Image.ANTIALIAS)
#seg_mask_new.save(os.path.join(Data_dir,'anna_mask.png'))

#%%
# =============================================================================
#  Section 5
#  Construct the argument parser and parse the arguments
#  using the MobileNet based model for faster results
# =============================================================================
ap_MobileNet = argparse.ArgumentParser()
ap_MobileNet.add_argument("-ext", "--extension", required=False, default='jpg', help="extension name. default is 'jpg'.")
ap_MobileNet.add_argument("-o", "--output", required=False, 
                default=os.path.join(Data_dir,'Task3_sec5 - chair_seg_vid_' + model_name + '.mp4'), 
                help="output video file")
args_ap_MobileNet = vars(ap_MobileNet.parse_args())
ext = args_ap_MobileNet['extension']
output = args_ap_MobileNet['output']
   

#%%
# =============================================================================
# making list of frames
# =============================================================================
frames_num = []
for frame in os.listdir(frames_path):
    if frame.endswith(ext):
        frame_num = frame[:-4]
        frame_num = frame_num[5:]
        frames_num.append(frame_num)
        
frames_num.sort(key=int)

frames = []
for num in frames_num:
    f = 'frame' + num + '.jpg'
    frames.append(f)

    
#%%
# =============================================================================
#  Define the codec and create VideoWriter object
# =============================================================================
height, width = seg_mask_object.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output, fourcc, 30.0, (width, height))


#%%
# =============================================================================
# making segmentation for each frame and create a video
# for MobileNet model:
# it is taking a while
# =============================================================================

for frame in frames:
    
    frame_path = os.path.join(frames_path,frame)
    resized_im, seg_mask_object, seg_mask_general, seg_image = segmentation_image(frame_path,seg_object)
    print(frame + ' is in process')
    # deepLab returns few segmentations, 
    # we take only the relevant for us (chair)

    k,s = np.where(seg_mask_object == 0)
   
    segmentation_frame = np.array(resized_im)
    
    segmentation_frame[k,s,:] = [0,0,0]
           
    out.write(cv2.cvtColor(segmentation_frame, cv2.COLOR_RGB2BGR)) # Write out frame to video
    
# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))


#%%
# =============================================================================
#  Loading model and Xception based model checkpoint 
#  and making a video based Xception model:
# =============================================================================

# Checkpoint_weight_path = 'D:\\DeepLab\\deeplabv3_pascal_trainval_2018_01_04.tar.gz'

#Checkpoint_weight_path = os.path.join(Data_dir,'deeplabv3_pascal_trainval_2018_01_04.tar.gz')
#MODEL = DeepLabModel(Checkpoint_weight_path)


Checkpoint_weight_path = 'D:\\DeepLab\\deeplabv3_pascal_trainval_2018_01_04.tar.gz'
MODEL = DeepLabModel(Checkpoint_weight_path)

print('Xception based model loaded successfully!')
model_name = 'Xception'

ap_xception = argparse.ArgumentParser()
ap_xception.add_argument("-ext", "--extension", required=False, default='jpg', help="extension name. default is 'jpg'.")
ap_xception.add_argument("-o", "--output", required=False, 
                default=os.path.join(Data_dir,'Task3_sec5 - chair_seg_vid_' + model_name + '.mp4'), 
                help="output video file")

args_Xception = vars(ap_xception.parse_args())
ext = args_Xception['extension']
output = args_Xception['output']
   
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output, fourcc, 30.0, (width, height))

for frame in frames:
    
    frame_path = os.path.join(frames_path,frame)
    resized_im, seg_mask_object, seg_mask_general, seg_image = segmentation_image(frame_path,seg_object)
    print(frame + ' is in process')
    # deepLab returns few segmentations, 
    # we take only the relevant for us (chair)

    k,s = np.where(seg_mask_object == 0)
   
    segmentation_frame = np.array(resized_im)
    
    segmentation_frame[k,s,:] = [0,0,0]
           
    out.write(cv2.cvtColor(segmentation_frame, cv2.COLOR_RGB2BGR)) # Write out frame to video
    
# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))



