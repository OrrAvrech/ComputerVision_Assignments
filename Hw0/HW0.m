% HW0
% Orr Avrech 302857065
% Opher Bar Nathan 302188628


%% Question 1 – Filters and convolution:

clc; close all; clear;

%% 1

clc; close all; clear;

%% Convolution
% %{
avg = fspecial('average',8);
disk = fspecial('disk',3);
gauss = fspecial('gaussian',5,10);
lap = fspecial('laplacian',0.5);
prewitt = fspecial('prewitt');
sobel = fspecial('sobel');

cameraman = im2double(rgb2gray(imread('cameraman.jpg')));


cameraman_avg = conv2(cameraman,avg,'same');
cameraman_disk = conv2(cameraman,disk,'same');
cameraman_gauss = conv2(cameraman,gauss,'same');
cameraman_lap = conv2(cameraman,lap,'same');
cameraman_prewitt = conv2(cameraman,prewitt,'same');
cameraman_sobel = conv2(cameraman,sobel,'same');

cameraman_fig = figure('Name','Cameraman');
imshow(cameraman)
title('Original Image')
print(cameraman_fig,'cameraman_fig','-djpeg')

cameraman_filters_fig = figure('Name','filters');
subplot(2,3,1)
imshow (cameraman_avg)
title('Average Filter')
subplot(2,3,2)
imshow (cameraman_disk)
title('Disk Filter')
subplot(2,3,3)
imshow (cameraman_gauss)
title('Gaussian Filter')
subplot(2,3,4)
imshow (cameraman_lap)
title('Laplacian Filter')
subplot(2,3,5)
imshow (cameraman_prewitt)
title('Prewitt Filter')
subplot(2,3,6)
imshow (cameraman_sobel)
title('Sobel Filter')

print(cameraman_fig,'cameraman filters','-djpeg')
print(cameraman_filters_fig,'cameraman filters','-djpeg')

%% filters parameters

tiger = im2double(rgb2gray(imread('tiger.jpg')));
peppers = im2double(rgb2gray(imread('peppers.jpg')));

tiger_avg_fig = figure('Name','avg parameter');
subplot(2,2,1)
imshow(tiger)
title('Original')
subplot(2,2,2)
avg = fspecial('average',3);
imshow(conv2(tiger,avg))
title('Avg parameter = 3')
subplot(2,2,3)
avg = fspecial('average',30);
imshow(conv2(tiger,avg))
title('Avg parameter = 30')
subplot(2,2,4)
avg = fspecial('average',70);
imshow(conv2(tiger,avg))
title('Avg parameter = 70')
print(tiger_avg_fig,'tiger_avg_fig','-djpeg');

tiger_disk_fig = figure('Name','disk parameter');
subplot(2,2,1)
imshow(tiger)
title('Original')
subplot(2,2,2)
disk = fspecial('disk',3);
imshow(conv2(tiger,disk))
title('Disk parameter = 3')
subplot(2,2,3)
disk = fspecial('disk',30);
imshow(conv2(tiger,disk))
title('Disk parameter = 30')
subplot(2,2,4)
disk = fspecial('disk',70);
imshow(conv2(tiger,disk))
title('Disk parameter = 70')
print(tiger_disk_fig,'tiger_disk_fig','-djpeg');

peppers_gauss_fig = figure('Name','gauss parameter');
subplot(2,2,1)
imshow(peppers)
title('Original')
subplot(2,2,2)
gauss = fspecial('gaussian',2,5);
imshow(conv2(peppers,gauss))
title('Gaussian Size = 3, sigma = 5')
subplot(2,2,3)
gauss = fspecial('gaussian',7,10);
imshow(conv2(peppers,gauss))
title('Gaussian Size = 7, sigma = 10')
subplot(2,2,4)
gauss = fspecial('gaussian',10,50);
imshow(conv2(peppers,gauss))
title('Gaussian Size = 10, sigma = 50')
print(peppers_gauss_fig,'peppers_gauss_fig','-djpeg');

peppers_lap_fig = figure('Name','laplace parameter');
subplot(2,2,1)
imshow(peppers)
title('Original')
subplot(2,2,2)
lap = fspecial('laplacian',0.1);
imshow(conv2(peppers,lap))
title('Laplacian parameter = 0.1')
subplot(2,2,3)
lap = fspecial('laplacian',0.5);
imshow(conv2(peppers,lap))
title('Laplacian  parameter = 0.5')
subplot(2,2,4)
lap = fspecial('laplacian',1.0);
imshow(conv2(peppers,lap))
title('Laplacian  parameter = 1.0')
print(peppers_lap_fig,'peppers_lap_fig','-djpeg');


%% 2

% peppers = im2double(rgb2gray(imread('peppers.jpg')));
avg = fspecial('average',15);
peppers_valid = conv2(peppers,avg,'valid');
peppers_same = conv2(peppers,avg,'same');
peppers_full = conv2(peppers,avg,'full');

Peppers_conv2 = figure('Name','peppers conv2');
subplot(1,3,1)
imshow(peppers_valid)
title('Peppers Valid')
subplot(1,3,2)
imshow(peppers_same)
title('Peppers Same')
subplot(1,3,3)
imshow(peppers_full)
title('Peppers Full')

print(Peppers_conv2,'peppers conv2','-djpeg');
%}
%% Question 2 – Image histograms:

clc; close all; clear; 

%% 1 - gamma correction
% %{
gamma = [ 0.5 1 1.5 ];

LUT = arrayfun(@(x) (0:1/255:1).^x, gamma, 'UniformOutput',false );

tiger = (rgb2gray(imread('tiger.jpg')));
cameraman = (rgb2gray(imread('cameraman.jpg')));
moon = (imread('moon.GIF'));
peppers = (rgb2gray(imread('peppers.jpg')));
valley = (imread('Valley.jpg'));

tiger_gamma = cellfun(@(x) uint8(round(255*x(double(tiger)+1))), LUT,...
    'UniformOutput',false);
cameraman_gamma = cellfun(@(x) uint8(round(255*x(double(cameraman)+1))), LUT,...
    'UniformOutput',false);
moon_gamma = cellfun(@(x) uint8(round(255*x(double(moon)+1))), LUT,...
    'UniformOutput',false);
peppers_gamma = cellfun(@(x) uint8(round(255*x(double(peppers)+1))), LUT,...
    'UniformOutput',false);
valley_gamma = cellfun(@(x) uint8(round(255*x(double(valley)+1))), LUT,...
    'UniformOutput',false);

Tiger_Gamma_Correction = figure('Name','Tiger Gamma Correction');
subplot(2,2,1)
imshow(tiger)
title('Original')
subplot(2,2,2)
imshow(tiger_gamma{1})
title('\gamma = 0.5')
subplot(2,2,3)
imshow(tiger_gamma{2})
title('\gamma = 1')
subplot(2,2,4)
imshow(tiger_gamma{3})
title('\gamma = 1.5')
print(Tiger_Gamma_Correction,'Tiger_Gamma_Correction', '-djpeg')

Cameraman_Gamma_Correction = figure('Name','Cameraman Gamma Correction');
subplot(2,2,1)
imshow(cameraman)
title('Original')
subplot(2,2,2)
imshow(cameraman_gamma{1})
title('\gamma = 0.5')
subplot(2,2,3)
imshow(cameraman_gamma{2})
title('\gamma = 1')
subplot(2,2,4)
imshow(cameraman_gamma{3})
title('\gamma = 1.5')
print(Cameraman_Gamma_Correction,'Cameraman_Gamma_Correction', '-djpeg')


Moon_Gamma_Correction = figure('Name','Moon Gamma Correction');
subplot(2,2,1)
imshow(moon)
title('Original')
subplot(2,2,2)
imshow(moon_gamma{1})
title('\gamma = 0.5')
subplot(2,2,3)
imshow(moon_gamma{2})
title('\gamma = 1')
subplot(2,2,4)
imshow(moon_gamma{3})
title('\gamma = 1.5')
print(Moon_Gamma_Correction,'Moon_Gamma_Correction', '-djpeg')


Peppers_Gamma_Correction = figure('Name','Peppers Gamma Correction');
subplot(2,2,1)
imshow(peppers)
title('Original')
subplot(2,2,2)
imshow(peppers_gamma{1})
title('\gamma = 0.5')
subplot(2,2,3)
imshow(peppers_gamma{2})
title('\gamma = 1')
subplot(2,2,4)
imshow(peppers_gamma{3})
title('\gamma = 1.5')
print(Peppers_Gamma_Correction,'Peppers_Gamma_Correction', '-djpeg')


Valley_Gamma_Correction = figure('Name','Valley Gamma Correction');
subplot(2,2,1)
imshow(valley)
title('Original')
subplot(2,2,2)
imshow(valley_gamma{1})
title('\gamma = 0.5')
subplot(2,2,3)
imshow(valley_gamma{2})
title('\gamma = 1')
subplot(2,2,4)
imshow(valley_gamma{3})
title('\gamma = 1.5')
print(Valley_Gamma_Correction,'Valley_Gamma_Correction', '-djpeg')

%}


%% 2 Histogram Equalization

clc; close all; clear;

moon = imread('moon.GIF');
valley = (imread('Valley.jpg'));
uint8_vec = 0:1:255;

moon_histeq = histeq(moon,uint8_vec);
valley_histeq = histeq(valley,uint8_vec);

Moon_Histogram_Equalization = figure('Name', 'Moon Histogram Equalization');
subplot(2,2,1)
imshow(moon)
title('Original')
subplot(2,2,2)
imshow(moon_histeq)
title('Histogram Equalization')
subplot(2,2,3)
histogram(moon)
title('Original Histogram')
subplot(2,2,4)
histogram(moon_histeq)
title('Histogram Equalization')
print(Moon_Histogram_Equalization,'Moon_Histogram_Equalization', '-djpeg')


Valley_Histogram_Equalization = figure('Name', 'Valley Histogram Equalization');
subplot(2,2,1)
imshow(valley)
title('Original')
subplot(2,2,2)
imshow(valley_histeq)
title('Histogram Equalization')
subplot(2,2,3)
histogram(valley)
title('Original Histogram')
subplot(2,2,4)
histogram(valley_histeq)
title('Histogram Equalization')
print(Valley_Histogram_Equalization,'Valley_Histogram_Equalization', '-djpeg')


%% 3 RGB to HSV

clc; close all; clear;

tiger = imread('tiger.jpg');
peppers = imread('peppers.jpg');

tiger_hsv = rgb2hsv(tiger);
peppers_hsv = rgb2hsv(peppers);

RGB_HSV = figure('Name','RGB_HSV');
subplot(2,2,1)
imshow(tiger)
title('Tiger RGB')
subplot(2,2,2)
imshow(tiger_hsv)
title('Tiger HSV')
subplot(2,2,3)
imshow(peppers)
title('Peppers RGB')
subplot(2,2,4)
imshow(peppers_hsv)
title('Peppers HSV')

print(RGB_HSV,'RGB_HSV','-djpeg');

