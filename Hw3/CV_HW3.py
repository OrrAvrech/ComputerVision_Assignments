# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:43:18 2018

@author: opher
"""

#%%

# =============================================================================
# Computer Vision - Homework 3
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

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pylab as pl

#%%

Data_dir = os.path.join('.','data')
seg_object = 'chair'
doCapture = False # True

#%%
frames_path = os.path.join(Data_dir,'frames')

frames_num = []
for frame in os.listdir(frames_path):
    if frame.endswith('.jpg'):
        frame_num = frame[:-4]
        frame_num = frame_num[5:]
        frames_num.append(frame_num)
        
frames_num.sort(key=int)

frames = []
for num in frames_num:
    f = 'frame' + num + '.jpg'
    frames.append(f)

frames_jump = frames[0:-1:30]
frames_jump = frames_jump [0:6]

#%%
# =============================================================================
# 1. Viewing. 
# For all the frames
# =============================================================================

# =============================================================================
# Plot Sequence Function
# =============================================================================

def plot_read_sequence(frame_list, pause_time):
    plt.figure()
    for f in frame_list:
        plt.clf()
        im = cv2.imread(os.path.join(frames_path,f))
        imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(imRGB)
        pl.pause(pause_time)
        
def plot_frame_list(frame_list, pause_time):
    plt.figure()
    for f in frame_list:
        plt.clf()
        imRGB = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        plt.imshow(imRGB)
        pl.pause(pause_time)

plot_read_sequence(frames_jump, 0.05)

#%%
# ============================================================================
# 2. Detection.
# Harris corner detection
# =============================================================================
 
# =============================================================================
#     NON-Maximum Suprassion
# =============================================================================
	
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

# =============================================================================
# Box to Dot
# =============================================================================

def from_box_to_dot(a):
    return round((a[0]+a[2])/2), round((a[1]+a[3])/2)
#%%
# =============================================================================
# Detect Corners 
# =============================================================================

Nimages = len(frames_jump)
TH_par = 0.005

boxes_nms = list()
dots_nms = list()

im_show_proportion = 1
# length of a side of the box (square) - local area for the NMS:
box_dim = 10 

plt.ion()

for frameItr in range(Nimages):

    # choose a frame
    frame_path = os.path.join(frames_path,frames_jump[frameItr])

    img = cv2.imread(frame_path)
    
# =============================================================================
#     # Padding the image for getting boxes with wanted size - not cropped
#     BLACK = [0, 0, 0]
#     img_pad = cv2.copyMakeBorder(img,box_dim,box_dim,box_dim,box_dim,cv2.BORDER_CONSTANT,value=BLACK)
# =============================================================================
    
    img_harris_dots = img.copy()
    img_harris_boxes = img.copy()
    img_harris_dots_nms = img.copy()
    img_harris_boxes_nms = img.copy()
    
    hight, width, chanel = img.shape

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # apply Harris corner detection
    dst = cv2.cornerHarris(gray,5,5,0.04)
    
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    # make dots on the corners by Harris
    img_harris_dots[dst>TH_par*dst.max()]=[0,0,255]
    
    # apply Threshold on Harris
    Harris_points = np.where(dst>TH_par*dst.max())
    
    xx_harris = Harris_points[0]
    yy_harris = Harris_points[1]
    
    # prepering the points to non max suppression function
    boxes_size = [len(xx_harris),4]
    boxes = np.zeros(boxes_size)
    
# =============================================================================
#     BEFORE PADDING WITH 0
#     # updating boxes x,y start and end
#     boxes[:,0] = [(ii-box_dim if ii>=box_dim else 0) for ii in xx_harris] # x_start
#     boxes[:,1] = [(ii-box_dim if ii>=box_dim else 0) for ii in yy_harris] # y_start
#     boxes[:,2] = [(ii+box_dim if ii<hight-box_dim else hight) for ii in xx_harris] # x_end
#     boxes[:,3] = [(ii+box_dim if ii<width-box_dim else width) for ii in yy_harris] # y_end
# =============================================================================
     
    # AFTER PADDING WITH 0
    # updating boxes x,y start and end
    boxes[:,0] = [ii-box_dim for ii in xx_harris] # x_start
    boxes[:,1] = [ii-box_dim for ii in yy_harris] # y_start
    boxes[:,2] = [ii+box_dim for ii in xx_harris] # x_end
    boxes[:,3] = [ii+box_dim for ii in yy_harris] # y_end
    
    # run non max suppression:
    boxes_nms_iter = non_max_suppression_fast(boxes,0.3)
    

    # make dots from boxes_nms
    dots_nms_iter = np.apply_along_axis(from_box_to_dot,1,boxes_nms_iter)
    dots_nms_iter = dots_nms_iter.astype('int')
    
    boxes_nms.append(boxes_nms_iter)
    dots_nms.append(dots_nms_iter)
    
    for (startX, startY, endX, endY) in boxes_nms_iter:
        cv2.rectangle(img_harris_boxes_nms, (startY, startX), (endY, endX), (0, 255, 0), 2)
        
    for (startX, startY, endX, endY) in boxes.astype('int'):
        cv2.rectangle(img_harris_boxes, (startY, startX), (endY, endX), (0, 255, 0), 2)
    
    if frameItr == 0:
        img_harris_boxes_nms_ref = img_harris_boxes_nms.copy()
        
    im_resized = cv2.resize(img_harris_boxes_nms, (0, 0), fx=im_show_proportion, fy=im_show_proportion)
    im_to_show = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(im_to_show)
    plt.show()

#%%
# =============================================================================
# 3. Manual Matching
# we chose features on the filmed object, not in the background 
# maybe need to chose the background
# =============================================================================

selected_fetures_box = list()
selected_fetures_dot = list()   

frame_path = os.path.join(frames_path,frames_jump[0])
img = cv2.imread(frame_path)

img_manual_match = img.copy()
# =============================================================================
# selected_fetures_box.append(boxes_nms[0][[82,63,60,61,62,66,73,68]]) #
# selected_fetures_box.append(boxes_nms[1][[142,98,93,95,94,109,113,103]])
# selected_fetures_box.append(boxes_nms[2][[138,105,92,93,89,118,113,103]])#
# selected_fetures_box.append(boxes_nms[3][[104,78,62,63,61,93,66,45]])
# selected_fetures_box.append(boxes_nms[4][[142,102,84,82,77,118,67,52]])#
# selected_fetures_box.append(boxes_nms[5][[121,84,75,72,68,95,59,41]])
# 
# selected_fetures_dot.append(dots_nms[0][[82,63,60,61,62,66,73,68]])
# selected_fetures_dot.append(dots_nms[1][[142,98,93,95,94,109,113,103]])
# selected_fetures_dot.append(dots_nms[2][[138,105,92,93,89,118,113,103]])
# selected_fetures_dot.append(dots_nms[3][[104,78,62,63,61,93,66,45]])
# selected_fetures_dot.append(dots_nms[4][[142,102,84,82,77,118,67,52]])
# selected_fetures_dot.append(dots_nms[5][[121,84,75,72,68,95,59,41]])
# =============================================================================

selected_fetures_box.append(boxes_nms[0][[81,62,59,60,61,65,72,67]]) 
selected_fetures_box.append(boxes_nms[1][[142,98,93,95,94,109,113,103]])
selected_fetures_box.append(boxes_nms[2][[138,104,91,92,88,117,112,102]])
selected_fetures_box.append(boxes_nms[3][[104,78,62,63,61,93,66,45]])
selected_fetures_box.append(boxes_nms[4][[143,103,85,83,78,119,67,52]])
selected_fetures_box.append(boxes_nms[5][[121,84,75,72,68,95,59,41]])

selected_fetures_dot.append(dots_nms[0][[81,62,59,60,61,65,72,67]])
selected_fetures_dot.append(dots_nms[1][[142,98,93,95,94,109,113,103]])
selected_fetures_dot.append(dots_nms[2][[138,104,91,92,88,117,112,102]])
selected_fetures_dot.append(dots_nms[3][[104,78,62,63,61,93,66,45]])
selected_fetures_dot.append(dots_nms[4][[143,103,85,83,78,119,67,52]])
selected_fetures_dot.append(dots_nms[5][[121,84,75,72,68,95,59,41]])


for frameItr in range(Nimages):
    
    frame_path = os.path.join(frames_path,frames_jump[frameItr])
    img_manual_match = cv2.imread(frame_path)
    
    startX, startY, endX, endY = selected_fetures_box[frameItr][0]
    cv2.rectangle(img_manual_match, (startY, startX), (endY, endX), (0, 255, 255), 3)
    
    X, Y = selected_fetures_dot[frameItr][1]
    cv2.circle(img_manual_match, (Y, X), 10, (0, 0, 255), 3)
    
    X, Y = selected_fetures_dot[frameItr][2]
    cv2.circle(img_manual_match, (Y, X), 10, (255, 0, 0), 3)
    
    startX, startY, endX, endY = selected_fetures_box[frameItr][3]
    cv2.rectangle(img_manual_match, (startY, startX), (endY, endX), (0, 255, 0), 3)    

    startX, startY, endX, endY = selected_fetures_box[frameItr][4]
    cv2.rectangle(img_manual_match, (startY, startX), (endY, endX), (0, 0, 255), 3)
    
    startX, startY, endX, endY = selected_fetures_box[frameItr][5]
    vrx = np.array([[startY,startX-5], [startY-5,(startX+endX)/2],[startY,endX+5],[endY,endX],[endY,startX]],np.int32)
    vrx = vrx.reshape((-1,1,2))
    cv2.polylines(img_manual_match, [vrx], True, (255,255,255),3)
    
    startX, startY, endX, endY = selected_fetures_box[frameItr][6]
    vrx = np.array([[startY,startX-5], [startY-5,(startX+endX)/2],[startY,endX+5],[endY,endX],[endY,startX]],np.int32)
    vrx = vrx.reshape((-1,1,2))
    cv2.polylines(img_manual_match, [vrx], True, (255,0,255),3)
    
    X, Y = selected_fetures_dot[frameItr][7]
    cv2.circle(img_manual_match, (Y, X), 10, (0, 255, 255), 3)
    
    im_to_show = cv2.cvtColor(img_manual_match, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(im_to_show)
    plt.show()

#%%
# =============================================================================
# 4. Transformation 
#    find affine transformation parametrs with Least squre method
# =============================================================================
def find_affine_trans_parameters(img_pts_ref,img_pts_trg):
    
    ones = np.asarray([np.ones(np.shape(img_pts_ref)[0]).astype('int')])
    
    b = np.concatenate((img_pts_ref, ones.T), axis=1)
    a = np.concatenate((img_pts_trg, ones.T), axis=1)
    
    M ,residuals ,rank ,s = np.linalg.lstsq(a, b, rcond=None)
    M[M < 1e-10] = 0
        
    return M.T

#%%
affine_refMatrix = list()
affine_adjacentMatrix = list()
img_pts_ref = selected_fetures_dot[0]

for frameItr in np.arange(Nimages):
    refMatrix_tmp = find_affine_trans_parameters(img_pts_ref, selected_fetures_dot[frameItr])
    
    affine_refMatrix.append(refMatrix_tmp)
    
    if frameItr == 0:
        adjacentMatrix_tmp = find_affine_trans_parameters(selected_fetures_dot[frameItr], selected_fetures_dot[frameItr])
    else:
        adjacentMatrix_tmp = find_affine_trans_parameters(selected_fetures_dot[frameItr-1], selected_fetures_dot[frameItr])
    
    affine_adjacentMatrix.append(adjacentMatrix_tmp)
    
#%%
# =============================================================================
# 5. Stabilization
# =============================================================================

def MatrixCumMul(matrixList, iteration):
    res = np.zeros(np.shape(matrixList[0]))
    if iteration == 0:
        res = matrixList[0]
    for ii in range(iteration):
        if ii == 0:
            res = matrixList[ii]
        else:
            res = np.matmul(res, matrixList[ii])
    
    return res

#%%

height, width, channel = np.shape(img)
frameListRef = list()
frameListAdjc = list()
for frameItr in np.arange(Nimages):
    frame_path = os.path.join(frames_path,frames_jump[frameItr])
    img = cv2.imread(frame_path)
    
    Mref = affine_refMatrix[frameItr]
    dstRef = cv2.warpAffine(img, Mref[0:2, :], (width, height), flags=cv2.INTER_LINEAR)
    frameListRef.append(dstRef)
    
    Madjc = MatrixCumMul(affine_adjacentMatrix, frameItr)
    dstAdjc = cv2.warpAffine(img, Madjc[0:2, :], (width, height), flags=cv2.INTER_LINEAR)
    frameListAdjc.append(dstAdjc)

plot_frame_list(frameListRef, 0.8)
plot_frame_list(frameListAdjc, 0.8)

#%%
# =============================================================================
# 6. Automatic Matching
# =============================================================================

L = 400
W = 20

frame_path_ref = os.path.join(frames_path,frames_jump[0])
img_ref = cv2.imread(frame_path)

# padding with white for highr SSD
COLOR = [0, 0, 0]
img_ref_pad = cv2.copyMakeBorder(img,int(W),int(W),int(W),int(W),cv2.BORDER_CONSTANT,value=COLOR)

features_pts_ref = dots_nms[0]
feature_L_window_ref = np.zeros(np.multiply([1, 2] ,np.shape(features_pts_ref)))

feature_L_window_ref[:,0] = [(ii-int(L/2) if ii>=int(L/2) else 0) for ii in features_pts_ref[:,0]] # x_start
feature_L_window_ref[:,1] = [(ii-int(L/2) if ii>=int(L/2) else 0) for ii in features_pts_ref[:,1]] # y_start
feature_L_window_ref[:,2] = [(ii+int(L/2) if ii<hight-int(L/2) else hight) for ii in features_pts_ref[:,0]] # x_end
feature_L_window_ref[:,3] = [(ii+int(L/2) if ii<width-int(L/2) else width) for ii in features_pts_ref[:,1]] # y_end

# =============================================================================
# feature_W_window_ref = np.zeros(np.multiply([1, 2] ,np.shape(features_pts_ref)))
# 
# feature_W_window_ref[:,0] = [(ii-int((W/2)-1) if ii>=int((W/2)-1) else 0) for ii in features_pts_ref[:,0]] # x_start
# feature_W_window_ref[:,1] = [(ii-int((W/2)-1) if ii>=int((W/2)-1) else 0) for ii in features_pts_ref[:,1]] # y_start
# feature_W_window_ref[:,2] = [(ii+int((W/2)+1) if ii<hight-int((W/2)+1) else hight) for ii in features_pts_ref[:,0]] # x_end
# feature_W_window_ref[:,3] = [(ii+int((W/2)+1) if ii<width-int((W/2)+1) else width) for ii in features_pts_ref[:,1]] # y_end
# =============================================================================


features_compatable_table = np.empty((np.shape(features_pts_ref)[0],Nimages))
features_compatable_table[:,0] = (np.arange(np.shape(features_pts_ref)[0])).T
features_compatable_table[:,1:Nimages] = np.nan

Nfeatures_Ref = np.shape(features_pts_ref)[0]

for frameIter in 1+np.arange(Nimages-1):
    
    frame_path = os.path.join(frames_path,frames_jump[frameIter])
    img_curr_frame = cv2.imread(frame_path)
    img_curr_frame_pad = cv2.copyMakeBorder(img,int(W),int(W),int(W),int(W),cv2.BORDER_CONSTANT,value=COLOR)
    
    curr_frame_features = dots_nms[frameIter]
    Nfeatures_curr = np.shape(curr_frame_features)[0]
    
    for ref_feature_Iter in np.arange(Nfeatures_Ref):
        
        startX, startY, endX, endY = feature_L_window_ref[ref_feature_Iter]
#        start_W_X, start_W_Y, end_W_X, end_W_Y = feature_W_window_ref[ref_feature_Iter]
        start_W_X, start_W_Y, end_W_X, end_W_Y = boxes_nms[0][ref_feature_Iter,:]
        
        curr_featurs_indices = list()
        
        # in the current frame - find the features which inside LxL:
        
        for cur_feature_Iter in np.arange(Nfeatures_curr):
            
            if (
                    curr_frame_features[cur_feature_Iter,0] >= startX and 
                    curr_frame_features[cur_feature_Iter,1] >= startY and
                    curr_frame_features[cur_feature_Iter,0] <= endX and
                    curr_frame_features[cur_feature_Iter,1] <= endY
             ):
                                    
                curr_featurs_indices.append(cur_feature_Iter)
        
        Ncompfeatures = len(curr_featurs_indices)
        
        if Ncompfeatures == 0:
            
            continue
        
        ref_window = img_ref_pad[ start_W_X+W : end_W_X+W , start_W_Y+W : end_W_Y+W , :]
        
        most_compatable_feature_ind = -1
        min_ssd = -1
        
        for compareIter in curr_featurs_indices:
            
            f_start_X, f_start_Y, f_end_X, f_end_Y = boxes_nms[frameIter][compareIter,:]
            curr_frame_window = img_curr_frame_pad[f_start_X+W:f_end_X+W, f_start_Y+W:f_end_Y+W, :]
            
            ssd = np.sum((ref_window[:,:,0:3] - curr_frame_window[:,:,0:3])**2)
            
            if min_ssd==-1:
                min_ssd = ssd
                most_compatable_feature_ind = compareIter
            else:
                if ssd < min_ssd: 
                    min_ssd = ssd
                    most_compatable_feature_ind = compareIter
                    
        # 1st column contains ref features indices 0.....107
        # 2nd column contains compatable features indices of 2nd frame
        # 3rd column contains compatable features indices of 3rd frame
        # so on....
        features_compatable_table[ref_feature_Iter,frameIter] = most_compatable_feature_ind

#%%  checking for features in all frames
        
        
frame0 = cv2.imread(os.path.join(frames_path,frames_jump[0]))
frame1 = cv2.imread(os.path.join(frames_path,frames_jump[1])) 
frame2 = cv2.imread(os.path.join(frames_path,frames_jump[2]))
frame3 = cv2.imread(os.path.join(frames_path,frames_jump[3]))
frame4 = cv2.imread(os.path.join(frames_path,frames_jump[4]))
frame5 = cv2.imread(os.path.join(frames_path,frames_jump[5]))


#%%

Iter_vec = [81,62,59,60,61,65,72,67]

for Iter in Iter_vec:     



    #Iter=Iter+1    
    frame0_f = features_compatable_table[Iter,0]
    frame1_f = features_compatable_table[Iter,1]
    frame2_f = features_compatable_table[Iter,2]
    frame3_f = features_compatable_table[Iter,3]
    frame4_f = features_compatable_table[Iter,4]
    frame5_f = features_compatable_table[Iter,5]
    
    # =============================================================================
    #     if (
    #         np.isnan(frame0_f) or np.isnan(frame1_f) or np.isnan(frame2_f) or
    #         np.isnan(frame3_f) or np.isnan(frame4_f) or np.isnan(frame5_f)
    #         ):
    #         
    #         continue
    # =============================================================================
    
    fig, ( (ax1, ax2, ax3),( ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(frame0)
    startX, startY, endX, endY = boxes_nms[0][int(frame0_f),:]
    rect = patches.Rectangle((startY,startX),20,20,linewidth=1.5,edgecolor='w',facecolor='none')
    ax1.add_patch(rect)
    
    ax2.imshow(frame1)
    startX, startY, endX, endY = boxes_nms[1][int(frame1_f),:]
    rect = patches.Rectangle((startY,startX),20,20,linewidth=1.5,edgecolor='w',facecolor='none')
    ax2.add_patch(rect)
    
    ax3.imshow(frame2)
    startX, startY, endX, endY = boxes_nms[2][int(frame2_f),:]
    rect = patches.Rectangle((startY,startX),20,20,linewidth=1.5,edgecolor='w',facecolor='none')
    ax3.add_patch(rect)
    
    ax4.imshow(frame3)
    startX, startY, endX, endY = boxes_nms[3][int(frame3_f),:]
    rect = patches.Rectangle((startY,startX),20,20,linewidth=1.5,edgecolor='w',facecolor='none')
    ax4.add_patch(rect)
    
    ax5.imshow(frame4)
    startX, startY, endX, endY = boxes_nms[4][int(frame4_f),:]
    rect = patches.Rectangle((startY,startX),20,20,linewidth=1.5,edgecolor='w',facecolor='none')
    ax5.add_patch(rect)
    
    ax6.imshow(frame5)
    startX, startY, endX, endY = boxes_nms[5][int(frame5_f),:]
    rect = patches.Rectangle((startY,startX),20,20,linewidth=1.5,edgecolor='w',facecolor='none')
    ax6.add_patch(rect)
    
    fig.suptitle('feature #' + str(int(frame0_f)))

    
#    fig.show()

# =============================================================================
# startX, startY, endX, endY = boxes_nms[0][50,:]
# cv2.rectangle(img_ref, (startY, startX), (endY, endX), (0, 255, 0), 3) 
# 
# startX, startY, endX, endY = boxes_nms[0][38,:]
# cv2.rectangle(img_curr_frame, (startY, startX), (endY, endX), (0, 255, 0), 3) 
# 
# plt.figure()
# plt.imshow(img_ref)
# plt.figure()
# plt.imshow(img_curr_frame)
# =============================================================================

#%%












