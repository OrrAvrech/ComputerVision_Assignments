import os
import cv2
import pickle
from skimage import color
import numpy as np
from matplotlib import pyplot as plt
from utils import save_img

#%% Load Masks List

Mask_dir = os.path.join('..','data','list_dream')
maskList = pickle.load(open(os.path.join(Mask_dir, 'frames_list_seg_mask_object.pkl'), "rb" ))

#%% Edge Detection on 1 Frame

# Choose Frame
frames_dir = os.path.join('..','data','frames_Dream','frame')
currentFrame = 600
name = frames_dir + str(currentFrame) + '.jpg'

img = cv2.imread(name)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_height = img.shape[0]
img_width = img.shape[1]

# Corresponding Mask
mask = maskList[200]
resized_mask = np.floor(cv2.resize(mask, (img_width, img_height)) / 255)
resized_mask = color.gray2rgb(resized_mask)
#plt.imshow(resized_mask, cmap='gray')
maskedImg = np.multiply(img, resized_mask).astype('uint8')
#plt.imshow(maskedImg)

# =============================================================================
# Canny
# =============================================================================

maskedImg_blur = cv2.blur(maskedImg, ksize=(5,5))
cannyImg = cv2.Canny(maskedImg_blur.astype('uint8'), 5, 30)
#plt.imshow(cannyImg, cmap='gray')

# =============================================================================
# Sobel (Only for Visualization)
# =============================================================================
sobelx = cv2.Sobel(maskedImg_blur,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(maskedImg_blur,cv2.CV_64F,0,1,ksize=5)
sobel = np.sqrt(sobelx**2 + sobely**2)

# =============================================================================
# Paste Background
# =============================================================================

# Resize and Show Background Image
bg_file = 'frame500_frame500.jpg'
bg = cv2.imread(os.path.join('..','data','out_images',bg_file))
resized_bg = cv2.resize(bg, (img_width, img_height))
bg_rgb = cv2.cvtColor(resized_bg, cv2.COLOR_BGR2RGB)
#plt.imshow(bg_rgb)

# Unmask Background
rgbmask=resized_mask
unmaskedBgg = np.multiply(bg_rgb, 1 - rgbmask).astype('uint8')
unmaskedBgg = cv2.cvtColor(unmaskedBgg, cv2.COLOR_BGR2RGB)
plt.imshow(unmaskedBgg)

# Merge Segmented Image (after Canny) with Background
totalFrame      = cv2.cvtColor(maskedImg    + unmaskedBgg, cv2.COLOR_BGR2RGB)
totalFrameCanny = cv2.cvtColor(color.gray2rgb(255-cannyImg) + unmaskedBgg, cv2.COLOR_BGR2RGB)


plt.figure()
plt.subplot(131),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(totalFrame)
plt.title('Transformed Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(totalFrameCanny)
plt.title('Transformed Image (Canny)'), plt.xticks([]), plt.yticks([])
plt.show()

#%%
def SegmentOnStyledBg(frame_dir, background_dir, res_out_dir, withCanny, idx_range, frame_sidx, bg_sidx):
    
    sidx = idx_range[0]
    eidx = idx_range[1]
    for ii in range(sidx, eidx):
        # Choose Frame
        currentFrame    = ii + (frame_sidx - sidx) # start list_Dream from frame500
        frameName       = frames_dir + str(currentFrame) + '.jpg'
        img             = cv2.imread(frameName)
        
        # Corresponding Mask
        mask            = maskList[ii-sidx+(frame_sidx - sidx)]
        resized_mask    = np.floor(cv2.resize(mask, (img_width, img_height)) / 255)
        resized_mask    = color.gray2rgb(resized_mask)
        maskedImg       = np.multiply(img, resized_mask).astype('uint8')
        
        # Resize Background Image
        currentBg   = ii + (bg_sidx - sidx)
        bgName      = background_dir + str(currentBg) + '_frame' + str(currentBg) + '.jpg'
        bg          = cv2.imread(bgName)
        resized_bg  = cv2.resize(bg, (img_width, img_height))
        bg_rgb      = cv2.cvtColor(resized_bg, cv2.COLOR_BGR2RGB)
        
        # Unmask Background
        unmaskedBgg = np.multiply(bg_rgb, 1 - resized_mask).astype('uint8')
        unmaskedBgg = cv2.cvtColor(unmaskedBgg, cv2.COLOR_BGR2RGB)
    
        if withCanny == True:
            maskedImg_blur = cv2.blur(maskedImg, ksize=(5,5))
            cannyImg = cv2.Canny(maskedImg_blur.astype('uint8'), 5, 30)
            # Merge Segmented Image (after Canny) with Background
            totalFrame  = cv2.cvtColor(color.gray2rgb(255-cannyImg) + unmaskedBgg, cv2.COLOR_BGR2RGB)
        else:
            # Merge Segmented Image (after Canny) with Background
            totalFrame  = cv2.cvtColor(maskedImg    + unmaskedBgg, cv2.COLOR_BGR2RGB)
        
        # =========================================================================
        # Save Result
        # =========================================================================
        # Format for out filename: {out_path}/{frame_prefix}_{bg_prefix}.jpg
        frame_prefix = 'frame' + str(currentFrame)
        bg_prefix    = 'bg' + str(currentBg)
        os.makedirs(res_out_dir, exist_ok=True)
        out_f = os.path.join(res_out_dir, '{}_{}.jpg'.format(frame_prefix, bg_prefix))    
        save_img(out_f, totalFrame)

#%%
background_dir = os.path.join('..','data','bahai_TameImpala','frame')
res_out_dir    = os.path.join('..','data','Results_Canny')
withCanny = True
idx_range = (400,600+1)

SegmentOnStyledBg(frames_dir, background_dir, res_out_dir, withCanny, idx_range, 500, 400)
