%%
% CV HW1
% Orr Avrech 302857065
% Opher Bar Nathan 302188628


%%

clear ; close all ; clc ;

%% Q2 Edge detection 

%% A  Sobel, Gaussian-Laplace and Canny

%% edge detectors parameters
%  tune parameters for each image
%  [ Pandas ; Faces ; Man Graffiti ]
% NOTE: All the images are in the folder "images".

sobel_TH = [ 0.17 ; 0.21 ; 0.15 ];

LoG_TH =[ 0.04 ; 0.03 ; 0.01 ];
LoG_sigma = [ 1.5 ; 1.65 ; 2 ];

canny_TH_low = [ 0.16 ; 0.16 ; 0.15 ];
canny_TH_hi = [ 0.36 ; 0.36 ; 0.2 ];
canny_sigma = [ 2 ; 2 ; 2 ];

%% Showing the results for each image, with the global parameters above

selected_images = { 'pandas' ; 'faces' ; 'man graffiti' };
Nimage = length(selected_images);

for imIter = 1:Nimage
    
    % NOTE: All the images are in the folder "images".
    im =  im2double(rgb2gray(imread([ 'images\' selected_images{imIter} '.jpg' ])));    
 
    sobel_im = edge(im,'Sobel',...                  % BW = edge(I,method,TH,direction)
        sobel_TH(imIter),'both');                            
    LoG_im = edge(im,'log',...                      % BW = edge(I,method,TH,sigma)
        LoG_TH(imIter),LoG_sigma(imIter));                               
    canny_im = edge(im,'Canny',...                  % BW = edge(I,method,[TH_low TH_hi],sigma)
        [ canny_TH_low(imIter) canny_TH_hi(imIter) ],canny_sigma(imIter));   

    figure()
    imshow(im)
    title('Original image')
%     print(['Q2 A - Original ' selected_images{imIter} ],'-dmeta');

    figure()
    imshow(sobel_im,[])
    title([ 'Sobel, TH = ' num2str(sobel_TH(imIter))])
%     print(['Q2 A - Sobel  ' selected_images{imIter} ],'-dmeta');

    figure()
    imshow(LoG_im,[])
    title([ 'LoG, TH = ' num2str(LoG_TH(imIter))...
        ', \sigma = ' num2str(LoG_sigma(imIter)) ])
%     print(['Q2 A - LoG ' selected_images{imIter} ],'-dmeta');
    
    figure()
    imshow(canny_im,[])
    title(['Canny, TH: Hi = ' num2str(canny_TH_hi(imIter))...
        ', Low = ' num2str(canny_TH_low(imIter))...
        ', \sigma = ' num2str(canny_sigma(imIter))])
%     print(['Q2 A - Canny ' selected_images{imIter} ],'-dmeta');    
    

end


%%

clear ; close all ; clc ;

%% B

%% Data mining

given_images = {'Church' ; 'Golf' ; 'Nuns'};
Nimage = length(given_images);

Nvalues = 100 ; % results in the PDF were used in Nvalues = 1000, 
                % changed to 100 for fester running.
TH_low_canny = 0;
TH = linspace(TH_low_canny+eps,0.99,Nvalues);

Methods = { 'Sobel' , 'LoG' , 'Canny' };
Nmethods = length(Methods);
LoG_sigma = 1.5;
canny_sigma = 1.5;

for imIter = 1:Nimage
    
    im =  im2double(imread([ 'edges_images_GT\' given_images{imIter} '.jpg' ]));
    im_GT =  imread([ 'edges_images_GT\' given_images{imIter} '_GT.bmp' ]);

    im_GT = im_GT>0;
      
    for thIter = 1:Nvalues
              
        for meIter = 1:Nmethods
            
            switch (Methods{meIter})
                case 'Sobel'
                    Edge_im = edge(im,'Sobel',TH(thIter),'both');                      % BW = edge(I,method,TH,direction)
                case 'LoG'
                    Edge_im = edge(im,'log',TH(thIter),LoG_sigma);                       % BW = edge(I,method,TH,sigma)
                case 'Canny'
                    Edge_im = edge(im,'Canny',[ TH_low_canny TH(thIter) ],canny_sigma);   % BW = edge(I,method,[TH_low TH_hi],sigma)
            end 

            intersection = and(Edge_im,im_GT);

            Percision = sum(sum(intersection)) / (eps+sum(sum(im_GT)));
            Recall =  sum(sum(intersection)) / (eps+sum(sum(Edge_im)));
            F = 2*Percision*Recall / (Percision + Recall + eps);

            Res.(Methods{meIter}).Precesion.(given_images{imIter})(thIter) = Percision ;
            Res.(Methods{meIter}).Recall.(given_images{imIter})(thIter) =  Recall ;
            Res.(Methods{meIter}).F.(given_images{imIter})(thIter) = F ;
            
        end
               
    end
        
end 

%% Figure Drawing - Showing Results


F2TH_fig = figure('Name','F vs TH');

for meIter = 1:Nmethods
    
    sumF = zeros(1,Nvalues);
    for imIter = 1:Nimage
       
        sumF = sumF + Res.(Methods{meIter}).F.(given_images{imIter});
    
    end
    Res.(Methods{meIter}).F.mean = sumF/Nimage;
    
    plot(TH,Res.(Methods{meIter}).F.mean,'LineWidth',1.4)
    hold on
    
end
   

hold off

str = '$$ \frac{2PR}{P+R} $$';
mytitleText = ['Q2,B 2 - F = ' str ' vs. Threshold'];

title(mytitleText ,'Fontsize',18,'Fontweight','bold','Interpreter','latex')
xlabel('Threshold','Fontsize',16)
ylabel(['F = ' str] ,'Fontsize',16,'Interpreter','latex')
grid
legend(Methods,'Fontsize',14,'Fontweight','bold')

text = {'Other Parameters:' ; ...
    ['Canny Low Threshold = ' num2str(TH_low_canny)] ;...
    ['Canny \sigma = ' num2str(canny_sigma)] ; ...
    ['LoG \sigma = ' num2str(LoG_sigma)] };
dim = [.7 .5 .3 .3];
annotation('textbox',dim,'String',text,'FitBoxToText','on','Fontsize',14);

%%









