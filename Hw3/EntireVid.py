#%%
# =============================================================================
# import relevant packages
# =============================================================================
import cv2
import numpy as np
import os

from matplotlib import pyplot as plt

import pylab as pl

#%%

Data_dir = os.path.join('.','data')
seg_object = 'santa' #'santa'
doCapture = False # False, True

#%%

# =============================================================================
# in this section the video is captured to frames:
# =============================================================================
if doCapture:

    video_name = seg_object + '_video.mp4'
    video_path = os.path.join(Data_dir,video_name)
    
    cap = cv2.VideoCapture(video_path)
    
    try:
        if not os.path.exists(os.path.join(Data_dir,'frames'+'_'+seg_object)):
            os.makedirs(os.path.join(Data_dir,'frames'+'_'+seg_object))
        
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
        frame_path = os.path.join(Data_dir,'frames'+'_'+seg_object,name)
        
        print ('Creating...' + name)
        cv2.imwrite(frame_path, frame)
        
        currentFrame += 1
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

#%%
frames_path = os.path.join(Data_dir,'frames'+'_'+seg_object)

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


#%%
# =============================================================================
# 1. Viewing. 
# For all the frames
# =============================================================================

# =============================================================================
# Plot Sequence Function
# =============================================================================

def saveVideo(frameList, vidName, fps):
    video_name = vidName 
    height, width, layers = np.shape(frameList[0])
    
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
    
    for frame in frameList:
        video.write(frame)
    
    cv2.destroyAllWindows()
    video.release()

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
# Harris Corners Detection 
# =============================================================================

Nimages = len(frames)
TH_par = 0.02 

boxes_nms = list()
dots_nms = list()

im_show_proportion = 1
# length of a side of the box (square) - local area for the NMS:
box_dim = 10 

plt.ion()

for frameItr in range(Nimages):

    # choose a frame
    frame_path = os.path.join(frames_path,frames[frameItr])

    img = cv2.imread(frame_path)
       
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
        

#%%
# =============================================================================
# 4. Transformation 
#    find affine transformation parametrs with Least squre method
# =============================================================================
def find_affine_trans_parameters(img_pts_ref,img_pts_trg):
    
    ones = np.asarray([np.ones(np.shape(img_pts_ref)[0]).astype('int')])
    
    b = np.concatenate((img_pts_ref, ones.T), axis=1)
    a = np.concatenate((img_pts_trg, ones.T), axis=1)
    
    M ,residuals ,rank ,s = np.linalg.lstsq(a, b)
    M[M < 1e-10] = 0
        
    return M.T

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
# =============================================================================
# 6. Automatic Matching
# =============================================================================

height, width, channel = np.shape(img)

L = 1800
W = 46

frame_path_ref = os.path.join(frames_path,frames[0])
img_ref = cv2.imread(frame_path)

# padding for image edges features boxes
COLOR = [0, 0, 0]
img_ref_pad = cv2.copyMakeBorder(img_ref,int(W),int(W),int(W),int(W),cv2.BORDER_CONSTANT,value=COLOR)

features_pts_ref = dots_nms[0]

#create WxW area around feature point
features_boxes_ref = np.empty((np.shape(features_pts_ref)[0],4))

features_boxes_ref[:,0] = [ii-int(W/2) for ii in features_pts_ref[:,0]] # x_start
features_boxes_ref[:,1] = [ii-int(W/2) for ii in features_pts_ref[:,1]] # y_start
features_boxes_ref[:,2] = [ii+int(W/2) for ii in features_pts_ref[:,0]] # x_end
features_boxes_ref[:,3] = [ii+int(W/2) for ii in features_pts_ref[:,1]] # y_end

#create LxL area around feature point
feature_L_window_ref = np.zeros(np.multiply([1, 2] ,np.shape(features_pts_ref)))

feature_L_window_ref[:,0] = [(ii-int(L/2) if ii>=int(L/2) else 0) for ii in features_pts_ref[:,0]] # x_start
feature_L_window_ref[:,1] = [(ii-int(L/2) if ii>=int(L/2) else 0) for ii in features_pts_ref[:,1]] # y_start
feature_L_window_ref[:,2] = [(ii+int(L/2) if ii<hight-int(L/2) else hight) for ii in features_pts_ref[:,0]] # x_end
feature_L_window_ref[:,3] = [(ii+int(L/2) if ii<width-int(L/2) else width) for ii in features_pts_ref[:,1]] # y_end


features_compatable_table = np.empty((np.shape(features_pts_ref)[0],Nimages))
features_compatable_table[:,0] = (np.arange(np.shape(features_pts_ref)[0])).T
features_compatable_table[:,1:Nimages] = np.nan

Nfeatures_Ref = np.shape(features_pts_ref)[0]

for frameIter in 1+np.arange(Nimages-1):
    
    frame_path = os.path.join(frames_path,frames[frameIter])
    img_curr_frame = cv2.imread(frame_path)
    
    img_curr_frame_pad = cv2.copyMakeBorder(img_curr_frame,int(W),int(W),int(W),int(W),cv2.BORDER_CONSTANT,value=COLOR)
    
    curr_frame_features = dots_nms[frameIter]
    
    curr_frame_features_box = np.zeros((np.shape(curr_frame_features)[0],4))

    curr_frame_features_box[:,0] = [ii-int(W/2) for ii in curr_frame_features[:,0]] # x_start
    curr_frame_features_box[:,1] = [ii-int(W/2) for ii in curr_frame_features[:,1]] # y_start
    curr_frame_features_box[:,2] = [ii+int(W/2) for ii in curr_frame_features[:,0]] # x_end
    curr_frame_features_box[:,3] = [ii+int(W/2) for ii in curr_frame_features[:,1]] # y_end
    
    Nfeatures_curr = np.shape(curr_frame_features)[0]
    
    for ref_feature_Iter in np.arange(Nfeatures_Ref):
        
        startX, startY, endX, endY = feature_L_window_ref[ref_feature_Iter].astype('int')
        start_W_X, start_W_Y, end_W_X, end_W_Y = features_boxes_ref[ref_feature_Iter,:].astype('int')
        
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
            
            f_start_X, f_start_Y, f_end_X, f_end_Y = curr_frame_features_box[compareIter,:].astype('int')
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
        
#%%

# =============================================================================
# 7. RANSAC
# =============================================================================
    
def featureCoordinates(featureTableIndices, dots_nms):
    Nimages   = len(dots_nms)
    Nfeatures = np.shape(featureTableIndices)[0]
    featureTableCoors = np.zeros((Nimages, Nfeatures, 2))
    for i in range(Nimages):
        idxs = featureTableIndices[:, i].astype('int')
        dots = dots_nms[i][idxs]
        featureTableCoors[i, :, :] = dots
        
    return featureTableCoors
    
def ransac(matched_feature_pairs):
    numIter = 100
    numPairs = np.shape(matched_feature_pairs)[1]
    inliers = list()
    maxGroupSize = 0
    maxIdx = 0
    for i in range(numIter):
        # Random Selection of Features
        idxs = np.random.randint(low=0, high=numPairs, size=3)
        # Geometric Transform Between Feature Pairs
        tform = find_affine_trans_parameters(matched_feature_pairs[0, idxs, :].astype(np.float32), matched_feature_pairs[1, idxs, :].astype(np.float32))
        tform = tform[0:2, :]
        # Obtain Features Using the 3-Points-Based-Transformation
        current_features    = matched_feature_pairs[1, :, :]
        calculated_features = np.matmul(current_features, tform[:,0:2]) + tform[:,2]
        # Euclidean Distance Between Calculated and Actual Coordinates
        dist = np.linalg.norm(matched_feature_pairs[0, :, :] - calculated_features, axis=1)
        # Create Inlier Group
        inliers.append(np.argwhere(dist < 100))
        currGroupSize = np.shape(inliers[int(i)])[0]
        if currGroupSize > maxGroupSize:
            maxGroupSize = currGroupSize
            maxIdx = i
    
    # Largest Inlier Group
    maxInlier = inliers[maxIdx]
    maxInlier = np.reshape(maxInlier, (-1,))
    # Calculate Transformation Based on Largest Inlier Group
    final_tform = find_affine_trans_parameters(matched_feature_pairs[0, maxInlier, :].astype(np.float32), matched_feature_pairs[1, maxInlier, :].astype(np.float32))

    return final_tform

#%%
# =============================================================================
# 8. Stabilization II
# =============================================================================

affine_adjacentMatrix = list()
autoFrameListAdjc   = list()
featureTableCoors = featureCoordinates(features_compatable_table, dots_nms)
for i in range(Nimages):
    if frameItr == 0:
        adjacentMatrix_tmp, _ = cv2.findHomography(featureTableCoors[0, :, :], featureTableCoors[0, :, :], cv2.RANSAC,5.0)
    else:
        adjacentMatrix_tmp, _ = cv2.findHomography(featureTableCoors[i-1, :, :], featureTableCoors[i, :, :], cv2.RANSAC,5.0)
        
    affine_adjacentMatrix.append(adjacentMatrix_tmp)
    
for frameItr in np.arange(Nimages):
    frame_path = os.path.join(frames_path,frames[frameItr])
    img = cv2.imread(frame_path)
    
    Madjc = MatrixCumMul(affine_adjacentMatrix, frameItr)
    dstAdjc = cv2.warpAffine(img, Madjc[0:2, :], (width, height), flags=cv2.INTER_LINEAR)
    autoFrameListAdjc.append(dstAdjc)
    
#%%
#==============================================================================
# Save Entire Video (Stabilization II)
#==============================================================================
saveVideo(autoFrameListAdjc, os.path.join(Data_dir,'stab2.avi'), 25)