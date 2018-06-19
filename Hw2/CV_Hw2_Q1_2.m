%% Part 1
%% 1.1
gimg = rgb2gray(imread('../../data/Inputs/imgs/0004_6.png'));
n = 6;
Lpyr = ConstructLaplacianPyramid(gimg, n);
figure;
for i = 1 : n
    subplot(1,n,i)
    imshow(Lpyr(:,:,i), []);
    if i == n
        title('Residual');
    else
        istr = num2str(i-1);
        title(['L' istr])
    end
end

%% 1.3
reconImg = ReconstructImage(Lpyr);
figure;
imshow(reconImg, []);

%% Part 2
%% 2.1
input = im2double(imread('../../data/Inputs/imgs/0004_6.png'));
mask  = im2double(imread('../../data/Inputs/masks/0004_6.png'));
eBg   = im2double(imread('../../data/Examples/bgs/6.jpg'));
input_eBg = ReplaceBg(input, mask, eBg);
figure;
subplot 121
imshow(input, []);
title('Original');
subplot 122
imshow(input_eBg, []);
title('Example Background');

%% 2.2
example = im2double(imread('../../data/Examples/imgs/6.png'));
n = 6;
eps = 1e-4;
% For each channel (R,G,B)
L_input   = zeros(size(input_eBg, 1), size(input_eBg, 2), n, 3);
L_example = zeros(size(example, 1), size(example, 2), n, 3);
S_input   = zeros(size(L_input));
S_example = zeros(size(L_example));
Gain      = zeros(size(S_input));
for i = 1 : 3
    % Input image with example background
    L_input(:,:,:,i) = ConstructLaplacianPyramid(input_eBg(:,:,i), n);
    % Example image
    L_example(:,:,:,i) = ConstructLaplacianPyramid(example(:,:,i), n);
    % Energy
    for j = 1 : n
        sigma = 2;
        S_input(:,:,j,i)   = imgaussfilt(L_input(:,:,j,i).^2, sigma^j);
        S_example(:,:,j,i) = imgaussfilt(L_example(:,:,j,i).^2, sigma^j);
    end
    % Gain
    Gain(:,:,:,i) = sqrt(S_example(:,:,:,i) ./ (S_input(:,:,:,i) + eps));
end
% Clipping
Gain(Gain > 2.8) = 2.8;
Gain(Gain < 0.9) = 0.9;

%% 2.3
L_out = zeros(size(Gain));
for i = 1 : 3
    L_out(:,:,:,i) = Gain(:,:,:,i) .* L_input(:,:,:,i);
    L_out(:,:,n,i) = L_example(:,:,n,i);
end

%% 2.4
rImg = zeros(size(input));
for i = 1 : 3
    rImg(:,:,i) = ReconstructImage(L_out(:,:,:,i));
end
rImg_eBg = ReplaceBg(rImg, mask, eBg);
figure;
subplot 131
imshow(input, []);
subplot 132
imshow(example, []);
subplot 133
imshow(rImg_eBg, []);

%% 2.5
%% input image: girl; style: 16
input = im2double(imread('../../data/Inputs/imgs/0004_6.png'));
mask  = im2double(imread('../../data/Inputs/masks/0004_6.png'));
eBg   = im2double(imread('../../data/Examples/bgs/16.jpg'));
example = im2double(imread('../../data/Examples/imgs/16.png'));
im_b_s0 = styleTransfer(input, mask, eBg, example);
figure;
subplot 131
imshow(input, []);
subplot 132
imshow(example, []);
subplot 133
imshow(im_b_s0, []);

%% input image: girl; style: 21
input = im2double(imread('../../data/Inputs/imgs/0004_6.png'));
mask  = im2double(imread('../../data/Inputs/masks/0004_6.png'));
eBg   = im2double(imread('../../data/Examples/bgs/21.jpg'));
example = im2double(imread('../../data/Examples/imgs/21.png'));
im_g_s21 = styleTransfer(input, mask, eBg, example);
figure;
subplot 131
imshow(input, []);
subplot 132
imshow(example, []);
subplot 133
imshow(im_g_s21, []);

%% input image: boy; style: 0
input = im2double(imread('../../data/Inputs/imgs/0006_001.png'));
mask  = im2double(imread('../../data/Inputs/masks/0006_001.png'));
eBg   = im2double(imread('../../data/Examples/bgs/0.jpg'));
example = im2double(imread('../../data/Examples/imgs/0.png'));
im_b_s0 = styleTransfer(input, mask, eBg, example);
figure;
subplot 131
imshow(input, []);
subplot 132
imshow(example, []);
subplot 133
imshow(im_b_s0, []);

%% input image: boy; style: 9
input = im2double(imread('../../data/Inputs/imgs/0006_001.png'));
mask  = im2double(imread('../../data/Inputs/masks/0006_001.png'));
eBg   = im2double(imread('../../data/Examples/bgs/9.jpg'));
example = im2double(imread('../../data/Examples/imgs/9.png'));
im_b_s9 = styleTransfer(input, mask, eBg, example);
figure;
subplot 131
imshow(input, []);
subplot 132
imshow(example, []);
subplot 133
imshow(im_b_s9, []);

%% input image: boy; style: 10
input = im2double(imread('../../data/Inputs/imgs/0006_001.png'));
mask  = im2double(imread('../../data/Inputs/masks/0006_001.png'));
eBg   = im2double(imread('../../data/Examples/bgs/10.jpg'));
example = im2double(imread('../../data/Examples/imgs/10.png'));
im_b_s10 = styleTransfer(input, mask, eBg, example);
figure;
subplot 131
imshow(input, []);
subplot 132
imshow(example, []);
subplot 133
imshow(im_b_s10, []);

%% 2.6
input = im2double(imread('../../data/Inputs/imgs/0006_001.png'));
mask  = im2double(imread('../../data/Inputs/masks/0006_001.png'));
eBg   = im2double(imread('../../data/Examples/bgs/9.jpg'));
example = im2double(imread('../Data/Frankenstein.jpg'));
[m_i,n_i,~] = size(input);
[m_e,n_e,~] = size(example);
diff = (n_e - n_i)/2;
example_cr = example(1 : m_i, diff+1 : n_i + diff, :);
im_b_frank = styleTransfer(input, mask, eBg, example_pad);
figure;
subplot 131
imshow(input, []);
subplot 132
imshow(example_cr, []);
subplot 133
imshow(im_b_frank, []);