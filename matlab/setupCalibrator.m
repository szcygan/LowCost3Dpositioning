% Clear workspace and close all figure windows
clear
close all

% Read marker positions from an Excel file. The file should contain 3
% column and as many rows as there are markers used. Columns should contain
% the X, Y and Z coordinates of the markers as described in the paper.
markersPositions = readtable('markersFrame.xlsx');

% Select the left and right calibration view files
fileInL = uigetfile('*.jpg','Select Left calibration view file');
fileInR = uigetfile('*.jpg','Select Right calibration view file');

% Read the left and right calibration view images and convert to grayscale
im1 = im2double(rgb2gray(imread(fileInR)));
im2 = im2double(rgb2gray(imread(fileInL)));

% Display instructions for marker pointing order. The order of pointing of
% the markers should match the order in which their coordinates are given
% in the previously loaded Excel file.
instrIm = imread('markingOrder.png');
figure
imshow(instrIm)
title('Marker pointing order')
pause

% Display the left calibration view image and prompt for marker selection
figure
imshow(im1)
[x1,y1] = getpts(); % Get user input for marker points (press Enter when done)

% Display the right calibration view image and prompt for marker selection
figure,imshow(im2)
[x2,y2] = getpts(); % Get user input for marker points (press Enter when done)

%% Get the centers of each marker by performing region growing based segmentation
% ----------- Here the parameterTuning.m script can be used for obtaining
% proper parameter values
% ######################################################### REQUIRED INPUT
% Define parameters for disk structuring element and region growing
r_disc = 6; % Radius of the disk structuring element
reg_maxdist = 0.04; % Maximum distance threshold for region growing
% ######################################################### 

% Create a disk structuring element
SE = strel("disk",r_disc);

% Round marker coordinates to the nearest integer values
x1 = round(x1);
y1 = round(y1);
x2 = round(x2);
y2 = round(y2);

% Initialize binary masks for segmentation output
maskOut1 = zeros(size(im1),'logical');
maskOut2 = maskOut1;

% Loop through each marker
for ii = 1:length(x1)
    % Perform region growing segmentation on the left calibration view image
    J = regiongrowing(im1,y1(ii),x1(ii),reg_maxdist);
    
    % Close the segmented region with a disk structuring element
    J = imclose(J,SE);
    
    % Calculate centroid of the segmented region
    s1(ii) = regionprops(J,'centroid');
    
    % Add segmented region to the binary mask
    maskOut1 = maskOut1 + J;
    
    % Perform region growing segmentation on the right calibration view image
    J = regiongrowing(im2,y2(ii),x2(ii),reg_maxdist);
    
    % Close the segmented region with a disk structuring element
    J = imclose(J,SE);
    
    % Calculate centroid of the segmented region
    s2(ii) = regionprops(J,'centroid');
    
    % Add segmented region to the binary mask
    maskOut2 = maskOut2 + J;
end

%%
% Convert centroid information from structures to matrices
sm1 = struct2mat(s1); % Convert centroid data of the left calibration view to matrix
sm2 = struct2mat(s2); % Convert centroid data of the right calibration view to matrix

% Display segmented regions overlaid on the original images
figure, % Create a new figure
subplot(1,2,1), % Create subplot for the left calibration view
imshowpair(maskOut1,im1), % Display the binary mask overlaid on the left calibration view image
hold on, % Hold the current axis for further plotting
scatter(sm1(:,1),sm1(:,2),'green','filled'), % Scatter plot of centroid positions on the left calibration view
title('Mask right') % Set subplot title for the left calibration view

subplot(1,2,2), % Create subplot for the right calibration view
imshowpair(maskOut2,im2), % Display the binary mask overlaid on the right calibration view image
hold on, % Hold the current axis for further plotting
scatter(sm2(:,1),sm2(:,2),'green','filled'), % Scatter plot of centroid positions on the right calibration view
title('Mask left') % Set subplot title for the right calibration view


%%
% Estimate camera projection matrices and reprojection errors for the left calibration view
[camMatrix1,reprojectionErrors1] = estimateCameraProjection(sm1,markersPositions{:,:});
% [camMatrix1,reprojectionErrors1] = estimateCameraMatrix(sm1,markersPositions{:,:}); % up to Matlab 2022a
        
% Estimate camera projection matrices and reprojection errors for the right calibration view
[camMatrix2,reprojectionErrors2] = estimateCameraProjection(sm2,markersPositions{:,:});
% [camMatrix2,reprojectionErrors2] = estimateCameraMatrix(sm2,markersPositions{:,:}); % up to Matlab 2022a

% Display reprojection errors for the left calibration view
disp(reprojectionErrors1')

% Display reprojection errors for the right calibration view
disp(reprojectionErrors2')

% Calculate the mean reprojection error
meanReprError = mean([reprojectionErrors1;reprojectionErrors2]);

% Display the mean reprojection error
fprintf('Mean reprojection error: %f \n',meanReprError);

% Save calibration results to a MAT file
save('calibrationResults2.mat')


%% %%%%%%%%% Functions
% Function: struct2mat
% Description: Convert centroid data from a struct array to a matrix format.
% Input:
%   - s: Struct array containing centroid data with fields 'Centroid'.
% Output:
%   - sm: Matrix containing centroid coordinates (x,y) for each struct element.
function sm = struct2mat(s)
    % Initialize output matrix with zeros
    sm = zeros(length(s),2);
    % Loop through each struct element
    for ii = 1:length(s)
        % Extract x-coordinate (column 1) and y-coordinate (column 2) of the centroid
        sm(ii,1) = s(ii).Centroid(1);
        sm(ii,2) = s(ii).Centroid(2);
    end
end


