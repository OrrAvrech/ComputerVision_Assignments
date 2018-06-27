from __future__ import division, print_function

import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from utils import preserve_colors_np
from utils import get_files, get_img, get_img_crop, save_img, resize_to, center_crop
import scipy
import time
from wct import WCT

#%%

def styleTransfer(content_path, style_path, out_path, alpha, concat):
    
    #==========================================================================
    #     args
    #==========================================================================
    args              = type('', (), {})
    args.checkpoints  = ['models/relu5_1', 'models/relu4_1', 'models/relu3_1', 'models/relu2_1', 'models/relu1_1']
    args.relu_targets = ['relu5_1', 'relu4_1', 'relu3_1', 'relu2_1', 'relu1_1']
    args.vgg_path     = 'models/vgg_normalised.t7'
    args.content_path = content_path
    args.style_path   = style_path
    args.out_path     = out_path
    args.keep_colors  = False
    args.device       = '/cpu:0'
    args.style_size   = 512
    args.crop_size    = 0
    args.content_size = 0
    args.passes       = 1
    args.random       = 0
    args.alpha        = alpha
    args.concat       = concat
    args.adain        = False
    args.swap5        = False
    args.ss_alpha     = 0.6
    args.ss_patch_size= 3
    args.ss_stride    = 1
    
    
    start = time.time()

    # Load the WCT model
    wct_model = WCT(checkpoints=args.checkpoints, 
                                relu_targets=args.relu_targets,
                                vgg_path=args.vgg_path, 
                                device=args.device,
                                ss_patch_size=args.ss_patch_size, 
                                ss_stride=args.ss_stride)

    # Get content & style full paths
    if os.path.isdir(args.content_path):
        content_files = get_files(args.content_path)
    else: # Single image file
        content_files = [args.content_path]
    if os.path.isdir(args.style_path):
        style_files = get_files(args.style_path)
        if args.random > 0:
            style_files = np.random.choice(style_files, args.random)
    else: # Single image file
        style_files = [args.style_path]

    os.makedirs(args.out_path, exist_ok=True)

    count = 0

    ### Apply each style to each content image
    for content_fullpath in content_files:
        content_prefix, content_ext = os.path.splitext(content_fullpath)
        content_prefix = os.path.basename(content_prefix)  # Extract filename prefix without ext

        content_img = get_img(content_fullpath)
        if args.content_size > 0:
            content_img = resize_to(content_img, args.content_size)
        
        for style_fullpath in style_files: 
            style_prefix, _ = os.path.splitext(style_fullpath)
            style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

            # style_img = get_img_crop(style_fullpath, resize=args.style_size, crop=args.crop_size)
            # style_img = resize_to(get_img(style_fullpath), content_img.shape[0])

            style_img = get_img(style_fullpath)

            if args.style_size > 0:
                style_img = resize_to(style_img, args.style_size)
            if args.crop_size > 0:
                style_img = center_crop(style_img, args.crop_size)

            if args.keep_colors:
                style_img = preserve_colors_np(style_img, content_img)

            # if args.noise:  # Generate textures from noise instead of images
            #     frame_resize = np.random.randint(0, 256, frame_resize.shape, np.uint8)
            #     frame_resize = gaussian_filter(frame_resize, sigma=0.5)

            # Run the frame through the style network
            stylized_rgb = wct_model.predict(content_img, style_img, args.alpha, args.swap5, args.ss_alpha, args.adain)

            if args.passes > 1:
                for _ in range(args.passes-1):
                    stylized_rgb = wct_model.predict(stylized_rgb, style_img, args.alpha, args.swap5, args.ss_alpha, args.adain)

            # Stitch the style + stylized output together, but only if there's one style image
            if args.concat:
                # Resize style img to same height as frame
                style_img_resized = scipy.misc.imresize(style_img, (stylized_rgb.shape[0], stylized_rgb.shape[0]))
                # margin = np.ones((style_img_resized.shape[0], 10, 3)) * 255
                stylized_rgb = np.hstack([style_img_resized, stylized_rgb])

            # Format for out filename: {out_path}/{content_prefix}_{style_prefix}.{content_ext}
            out_f = os.path.join(args.out_path, '{}_{}{}'.format(content_prefix, style_prefix, content_ext))
            # out_f = f'{content_prefix}_{style_prefix}.{content_ext}'
            
            save_img(out_f, stylized_rgb)

            count += 1
            print("{}: Wrote stylized output image to {}".format(count, out_f))

    print("Finished stylizing {} outputs in {}s".format(count, time.time() - start))
    
#%% Use styleTransfer (One Iteration)

content_path = os.path.join('..','data','content','frame800.jpg')
style_path   = os.path.join('..','data','styles','baa.jpg')
out_path     = os.path.join('..','data','out_images')   
alpha        = 0.1
concat       = False
styleTransfer(content_path, style_path, out_path, alpha, concat)

#%% Iterate Over Background Frames and Style Frames
styleFrames_dir = os.path.join('..','data','frames_TameImpala')
bgFrames_dir    = os.path.join('..','data','frames_bahai_4k')
out_path         = os.path.join('..','data','bahai_TameImpala')

alpha = 0.1
concat=False

for ii in range(400,1400):
    content_path = os.path.join(bgFrames_dir, 'frame' + str(ii) + '.jpg')
    style_path = os.path.join(styleFrames_dir, 'frame' + str(ii) + '.jpg')
    styleTransfer(content_path, style_path, out_path, alpha, concat)
    K.clear_session()
    tf.reset_default_graph()
