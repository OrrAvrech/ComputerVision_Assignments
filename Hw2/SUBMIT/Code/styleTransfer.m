function [ outputImg ] = styleTransfer( input, mask, eBg, example )

n = 6;
eps = 1e-4;

input_eBg = ReplaceBg(input, mask, eBg);
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

%%
L_out = zeros(size(Gain));
for i = 1 : 3
    L_out(:,:,:,i) = Gain(:,:,:,i) .* L_input(:,:,:,i);
    L_out(:,:,n,i) = L_example(:,:,n,i);
end

%%
rImg = zeros(size(input));
for i = 1 : 3
    rImg(:,:,i) = ReconstructImage(L_out(:,:,:,i));
end
rImg_eBg = ReplaceBg(rImg, mask, eBg);

% Return
outputImg = rImg_eBg;

end

