clear % Clear the workspace

% Load calibration results data
load('calibrationResults2.mat','camMatrix1','reprojectionErrors1','camMatrix2','reprojectionErrors2')

% Select the left and right view image files
[fileInL,path] = uigetfile('*.jpg','Select Left view file'); % Prompt user to select left view file
fileInL = [path,fileInL]; % Combine file path and name for left view file
fileInR = uigetfile('*.jpg','Select Right view file'); % Prompt user to select right view file
fileInR = [path,fileInR]; % Combine file path and name for right view file

% Read the left and right calibration view images and convert to grayscale
im1 = im2double(rgb2gray(imread(fileInR))); % Read and convert right view image to grayscale
im2 = im2double(rgb2gray(imread(fileInL))); % Read and convert left view image to grayscale

figure % Create a new figure
imshow(im2); % Display left view image
title('Click all the markers and hit Enter'); % Set title of the figure

% Prompt user to select markers on the left view image
[x1,y1] = getpts(); % Get user-selected points
x1 = round(x1); % Round x coordinates to nearest integer
y1 = round(y1); % Round y coordinates to nearest integer

figure % Create a new figure
imshow(im1); % Display right view image
title('Click all the markers in the same order and hit Enter'); % Set title of the figure

% Prompt user to select markers on the right view image
[x2,y2] = getpts(); % Get user-selected points
x2 = round(x2); % Round x coordinates to nearest integer
y2 = round(y2); % Round y coordinates to nearest integer

%% Perform region growing based segmentation to get the centers of each marker
% ######################################################### REQUIRED INPUT
% Define parameters for disk structuring element and region growing
r_disc = 6; % Radius of the disk structuring element
reg_maxdist = 0.1; % Maximum distance threshold for region growing
% ######################################################### 

SE = strel("disk",r_disc); % Create a disk structuring element

% Round the coordinates of markers to the nearest integer
x1 = round(x1); % Round x coordinates of markers on the left view image
y1 = round(y1); % Round y coordinates of markers on the left view image
x2 = round(x2); % Round x coordinates of markers on the right view image
y2 = round(y2); % Round y coordinates of markers on the right view image

maskOut1 = zeros(size(im2),'logical'); % Initialize binary mask for markers on the left view image
maskOut2 = maskOut1; % Initialize binary mask for markers on the right view image

% Loop through each marker
for ii = 1:length(x1)
    % Perform region growing segmentation on the left view image
    J = regiongrowing(im2,y1(ii),x1(ii),reg_maxdist);
    J = imclose(J,SE); % Perform morphological closing to fill gaps and smooth edges
    s1(ii) = regionprops(J,'centroid'); % Extract centroid of the segmented region
    maskOut1 = maskOut1 + J; % Add segmented region to the binary mask
    
    % Perform region growing segmentation on the right view image
    J = regiongrowing(im1,y2(ii),x2(ii),reg_maxdist);
    J = imclose(J,SE); % Perform morphological closing to fill gaps and smooth edges
    s2(ii) = regionprops(J,'centroid'); % Extract centroid of the segmented region
    maskOut2 = maskOut2 + J; % Add segmented region to the binary mask
end

% Convert regionprops struct to matrix format for easier manipulation
sm1 = struct2mat(s1); % Convert regionprops struct for left view image to matrix
sm2 = struct2mat(s2); % Convert regionprops struct for right view image to matrix

% Display the results: masks overlaid on original images with marker centroids
figure,subplot(1,2,1),imshowpair(maskOut1,im2),hold on, scatter(sm1(:,1),sm1(:,2),'green','filled'),title('Mask 1')
subplot(1,2,2),imshowpair(maskOut2,im1),hold on, scatter(sm2(:,1),sm2(:,2),'green','filled'),title('Mask 2')

%% Calculating the 3D real-world coordinates of the markers using triangulation method

% The coordinates of the markers in the photo from the left camera
coord_LC = [x1, y1];

% The coordinates of the markers in the photo from the right camera
coord_RC = [x2, y2];

% Perform triangulation to calculate 3D real-world coordinates of the markers
% The function triangulate takes input parameters:
% - coord_LC: Coordinates of the markers in the left camera image
% - coord_RC: Coordinates of the markers in the right camera image
% - camMatrix2: Camera projection matrix for the right camera
% - camMatrix1: Camera projection matrix for the left camera
% It returns:
% - Gmatrix: Matrix containing the 3D real-world coordinates of the markers
% - reprojectionErrors: Vector containing reprojection errors for each marker
[Gmatrix, reprojectionErrors] = triangulate(coord_LC, coord_RC, camMatrix2, camMatrix1);

% Display the triangulated 3D real-world coordinates of the markers
disp('Real world coordinates of the points:')
disp(Gmatrix)

% Display the reprojection errors
disp('Reprojection errors:')
disp(reprojectionErrors)


%% Calculate the coplanarity error (tolerance) of the plane defined by the points and the equation of the plane using the Gmatrix
% Extract the points from Gmatrix
points = Gmatrix(:, 1:3);

% Fit a plane to the points using singular value decomposition (SVD)
[U, S, V] = svd(points - mean(points, 1));
normal = V(:, 3)'; % Extract the normal vector of the plane from the right singular vectors

% Calculate the equation of the plane in the form ax + by + cz + d = 0
% where [a, b, c] is the normal vector and d is the negative distance from the origin
imagingPlane.d = -dot(normal, mean(points, 1));

% Define the plane equation coefficients
imagingPlane.a = normal(1);
imagingPlane.b = normal(2);
imagingPlane.c = normal(3);

% Calculate the coplanarity error (tolerance) of the plane
% For each point (x, y, z), the coplanarity error is given by |ax + by + cz + d|
% where (x, y, z) is the point coordinates
coplanarity_error = abs(imagingPlane.a * points(:, 1) + imagingPlane.b * points(:, 2) + imagingPlane.c * points(:, 3) + imagingPlane.d);

% Display the coplanarity error
disp('Coplanarity Error (Tolerance):');
disp(coplanarity_error);

%%
save('triangulationResults.mat',"Gmatrix","reprojectionErrors","imagingPlane","coplanarity_error")

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