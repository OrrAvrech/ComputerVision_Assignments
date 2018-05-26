%% Laplacian Pyramid
function [Lpyr] = ConstructLaplacianPyramid(I, n)

sigma_1 = 2;
Lpyr = zeros(size(I, 1), size(I, 2), n);
Gpyr = ConstructGaussianPyramid(I, n, sigma_1);
for i = 1 : n-1
    Lpyr(:,:,i) = Gpyr(:,:,i) - Gpyr(:,:,i+1);
end

Lpyr(:,:,n) = Gpyr(:,:,n);
    
end

%% Gaussian Pyramid
function [Gpyr] = ConstructGaussianPyramid(I, n, sigma)

Gpyr = zeros(size(I, 1), size(I, 2), n);
Gpyr(:,:,1) = I;
for i = 2 : n
    Gpyr(:,:,i) = imgaussfilt(Gpyr(:,:,i-1), sigma^i);
end
    
end