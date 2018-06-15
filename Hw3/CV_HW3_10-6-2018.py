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
    boxes[:,0] = [(ii-box_dim if ii>=box_dim else 0) for ii in xx_harris] # x_start
    boxes[:,1] = [(ii-box_dim if ii>=box_dim else 0) for ii in yy_harris] # y_start
    boxes[:,2] = [(ii+box_dim if ii<hight-box_dim else hight) for ii in xx_harris] # x_end
    boxes[:,3] = [(ii+box_dim if ii<width-box_dim else width) for ii in yy_harris] # y_end
    
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
selected_fetures_box.append(boxes_nms[0][[82,63,60,61,62,66,73,68]])
selected_fetures_box.append(boxes_nms[1][[142,98,93,95,94,109,113,103]])
selected_fetures_box.append(boxes_nms[2][[138,105,92,93,89,118,113,103]])
selected_fetures_box.append(boxes_nms[3][[104,78,62,63,61,93,66,45]])
selected_fetures_box.append(boxes_nms[4][[142,102,84,82,77,118,67,52]])
selected_fetures_box.append(boxes_nms[5][[121,84,75,72,68,95,59,41]])

selected_fetures_dot.append(dots_nms[0][[82,63,60,61,62,66,73,68]])
selected_fetures_dot.append(dots_nms[1][[142,98,93,95,94,109,113,103]])
selected_fetures_dot.append(dots_nms[2][[138,105,92,93,89,118,113,103]])
selected_fetures_dot.append(dots_nms[3][[104,78,62,63,61,93,66,45]])
selected_fetures_dot.append(dots_nms[4][[142,102,84,82,77,118,67,52]])
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
    cv2.polylines(img_manual_match, [vrx], True, (0,0,0),3)
    
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
    
    a = np.concatenate((img_pts_ref, ones.T), axis=1)
    b = np.concatenate((img_pts_trg, ones.T), axis=1)
    
    M ,residuals ,rank ,s = np.linalg.lstsq(a, b, rcond=None)
    M[M < 1e-10] = 0
        
    return M.T

affine_refMatrix = list()
affine_adjacentMatrix = list()
img_pts_ref = selected_fetures_dot[0]

for frameItr in np.arange(Nimages):
    refMatrix_tmp = find_affine_trans_parameters(img_pts_ref,selected_fetures_dot[frameItr])
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
    for ii in range(iteration):
        if ii == 0:
            res = matrixList[ii]
        else:
            res = np.matmul(res, matrixList[ii])
    
    return res

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

plot_frame_list(frameListRef, 0.1)
#plot_frame_list(frameListAdjc, 0.1)

#bigMat_inv = np.linalg.inv(affine_BigMatrix[frameNum])
#xmesh , ymesh = np.meshgrid(np.arange(height), np.arange(width))
#xmesh_flat = xmesh.flatten()
#ymesh_flat = ymesh.flatten()
#ones_vec = np.ones(np.shape(xmesh_flat)[0])
#source_coordinates = np.stack((xmesh_flat, ymesh_flat, ones_vec), axis=0)
#
#stablized_coordinates = np.matmul(bigMat_inv, source_coordinates)
#
#r = np.zeros((width,))
#g = np.zeros((width,))
#b = np.zeros((width,))
## interp2d
#for i in range(width):
#    r[i] = interpolate.interp2d(stablized_coordinates[0, height*i : height*(i+1)], stablized_coordinates[1, height*i : height*(i+1)], img[:,i,0], kind='linear')
#    g[i] = interpolate.interp2d(stablized_coordinates[0, height*i : height*(i+1)], stablized_coordinates[1, :height*i : height*(i+1)], img[:,i,1], kind='linear')
#    b[i] = interpolate.interp2d(stablized_coordinates[0, height*i : height*(i+1)], stablized_coordinates[1, height*i : height*(i+1)], img[:,i,2], kind='linear')

#%%
# =============================================================================
# 6. Automatic Matching
# =============================================================================
































#%%
# =============================================================================
# def Affine_Fit( from_pts, to_pts ):
#     q = from_pts
#     p = to_pts
#     
#     if len(q) != len(p) or len(q)<1:
#         print ("from_pts and to_pts must be of same size.")
#         return False
#     
#     dim = len(q[0]) # num of dimensions
#     if len(q) < dim:
#         print ("Too few points => under-determined system.")
#         return False
#     
#     # Make an empty (dim) x (dim+1) matrix and fill it
#     c = [[0.0 for a in range(dim)] for i in range(dim+1)]
#     for j in range(dim):
#         for k in range(dim+1):
#             for i in range(len(q)):
#                 qt = list(q[i]) + [1]
#                 c[k][j] += qt[k] * p[i][j]
#     
#     # Make an empty (dim+1) x (dim+1) matrix and fill it
#     Q = [[0.0 for a in range(dim)] + [0] for i in range(dim+1)]
#     for qi in q:
#         qt = list(qi) + [1]
#         for i in range(dim+1):
#             for j in range(dim+1):
#                 Q[i][j] += qt[i] * qt[j]
#     
#     # Ultra simple linear system solver. Replace this if you need speed.
#     def gauss_jordan(m, eps = 1.0/(10**10)):
#       """Puts given matrix (2D array) into the Reduced Row Echelon Form.
#      Returns True if successful, False if 'm' is singular.
#      NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
#      Written by Jarno Elonen in April 2005, released into Public Domain"""
#       
#       (h, w) = (len(m), len(m[0]))
#       for y in range(0,h):
#           maxrow = y
#           for y2 in range(y+1, h):    # Find max pivot
#               if abs(m[y2][y]) > abs(m[maxrow][y]):
#                   maxrow = y2
#               (m[y], m[maxrow]) = (m[maxrow], m[y])
#               if abs(m[y][y]) <= eps:     # Singular?
#                   return False
#               for y2 in range(y+1, h):    # Eliminate column y
#                   c = m[y2][y] / m[y][y]
#                   for x in range(y, w):
#                       m[y2][x] -= m[y][x] * c
#           for y in range(h-1, 0-1, -1): # Backsubstitute
#               c  = m[y][y]
#               for y2 in range(0,y):
#                   for x in range(w-1, y-1, -1):
#                       m[y2][x] -=  m[y][x] * m[y2][y] / c
#               m[y][y] /= c
#               for x in range(h, w):       # Normalize row y
#                   m[y][x] /= c
#           return True
#     
#     # Augement Q with c and solve Q * a' = c by Gauss-Jordan
#     M = [ Q[i] + c[i] for i in range(dim+1)]
#     if not gauss_jordan(M):
#         print ("Error: singular matrix. Points are probably coplanar.")
#         return False
#     
#     # Make a result object
#     class Transformation:
#     #"""Result object that represents the transformation
#     #   from affine fitter."""
#     
#         def To_Str(self):
#             res = ""
#             for j in range(dim):
#                 str = "x%d' = " % j
#                 for i in range(dim):
#                     str +="x%d * %f + " % (i, M[i][j+dim+1])
#                 str += "%f" % M[dim][j+dim+1]
#                 res += str + "\n"
#             return res
#     
#         def Transform(self, pt):
#             res = [0.0 for a in range(dim)]
#             for j in range(dim):
#                 for i in range(dim):
#                     res[j] += pt[i] * M[i][j+dim+1]
#                 res[j] += M[dim][j+dim+1]
#             return res
# 
#     return Transformation()
# 
# #%%
# 
# from_pt = selected_fetures_dot[0]
# to_pt = selected_fetures_dot[2]    
# trn = Affine_Fit(from_pt, to_pt)
# 
# print ("Transformation is:")
# print (trn.To_Str())
# 
# err = 0.0
# for i in range(len(from_pt)):
#     fp = from_pt[i]
#     tp = to_pt[i]
#     t = trn.Transform(fp)
#     print ("%s => %s ~= %s" % (fp, tuple(t), tp))
#     err += ((tp[0] - t[0])**2 + (tp[1] - t[1])**2)**0.5
# 
# print ("Fitting error = %f" % err)
# =============================================================================

    
#%% 
    
# =============================================================================
# # make dots on corners in image - AFTER non max suppresion      
# img_harris_dots_nms[nms_dots[:,0],nms_dots[:,1],:] = [0, 255, 0]
# 
# # make boxes on image - BEFORE non max suppresion
# boxes = boxes.astype(int)
# for (startX, startY, endX, endY) in boxes:
# 	cv2.rectangle(img_harris_boxes, (startY, startX), (endY, endX), (0, 0, 255), 1)
# 
# # make boxes on image - AFTER non max suppresion      
# for (startX, startY, endX, endY) in boxes_nms:
#     cv2.rectangle(img_harris_boxes_nms, (startY, startX), (endY, endX), (0, 255, 0), 2)
# =============================================================================
 
#%% 
# =============================================================================
# display the images in CV2
# =============================================================================
# =============================================================================
# cv2.imshow("Harris - Points", img_harris_dots)
# cv2.imshow("NMS Harris - Points", img_harris_dots_nms)        
# cv2.imshow("Harris - Rectangle", img_harris_boxes)
# =============================================================================


# =============================================================================
# cv2.imshow("NMS Harris - Rectangle", img_harris_boxes_nms)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
# =============================================================================
#%% 
# =============================================================================
# display the images in PyLab
# =============================================================================
# =============================================================================
# pl.subplot(311),pl.imshow(cv2.cvtColor(img_harris_dots, cv2.COLOR_RGB2BGR),'gray'),pl.title('Harris - Points')
# pl.axis('off')
# 
# pl.subplot(312),pl.imshow(cv2.cvtColor(img_harris_boxes, cv2.COLOR_RGB2BGR),'gray'),pl.title('Harris - Rectangle')
# pl.axis('off')
# 
# pl.subplot(313),pl.imshow(cv2.cvtColor(img_harris_boxes_nms, cv2.COLOR_RGB2BGR),'gray'),pl.title('NMS Harris - Rectangle')
# pl.axis('off')
# 
# pl.show()
# =============================================================================


#%% To Do

# =============================================================================
# Do the Detection in loop for the 6 images
# =============================================================================

















