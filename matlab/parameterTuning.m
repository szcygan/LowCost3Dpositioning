%% Routine for establihing preffered settings based on minimum reprojection error
% ######################################################### REQUIRED INPUT
% Define parameters for disk structuring element and region growing
% Define a range of disk radii and region growing threshold values
r_disc = 1:8; % Range of disk radii
regVals = 0.01:0.01:0.2; % Range of region growing threshold values

% Initialize matrix to store mean reprojection errors for different parameter combinations
meanReprError = zeros(length(regVals),length(r_disc));

% Loop over disk radii
for kk=1:length(r_disc)
    % Loop over region growing threshold values
    for jj=1:length(regVals)
        % Set region growing threshold value and create disk structuring element
        reg_maxdist = regVals(jj);
        SE = strel("disk",r_disc(kk));
        
        % Round marker coordinates to nearest integer values
        x1 = round(x1);
        y1 = round(y1);
        x2 = round(x2);
        y2 = round(y2);
        
        % Initialize logical masks for image segmentation
        maskOut1 = zeros(size(im1),'logical');
        maskOut2 = maskOut1;
        
        % Loop over each marker
        for ii = 1:length(x1)
            % Perform region growing segmentation on the first image
            J = regiongrowing(im1,y1(ii),x1(ii),reg_maxdist);
            J = imclose(J,SE); % Close gaps in segmented regions
            s1(ii) = regionprops(J,'centroid'); % Compute centroid of segmented region
            maskOut1 = maskOut1 + J; % Update mask
            
            % Perform region growing segmentation on the second image
            J = regiongrowing(im2,y2(ii),x2(ii),reg_maxdist);
            J = imclose(J,SE); % Close gaps in segmented regions
            s2(ii) = regionprops(J,'centroid'); % Compute centroid of segmented region
            maskOut2 = maskOut2 + J; % Update mask
        end
        
        % Convert centroid data to matrix format
        sm1 = struct2mat(s1);
        sm2 = struct2mat(s2);
        
        % Estimate camera projection matrices and compute reprojection errors
        % [camMatrix1,reprojectionErrors1] = estimateCameraMatrix(sm1,markersPositions{:,:}); % up to Matlab 2022a
        % [camMatrix2,reprojectionErrors2] = estimateCameraMatrix(sm2,markersPositions{:,:}); % up to Matlab 2022a
        [camMatrix1,reprojectionErrors1] = estimateCameraProjection(sm1,markersPositions{:,:}); % since Matlab 2022b
        [camMatrix2,reprojectionErrors2] = estimateCameraProjection(sm2,markersPositions{:,:}); % since Matlab 2022b
        
        % Compute and store mean reprojection error for the current parameter combination
        meanReprError(jj,kk) = mean([reprojectionErrors1;reprojectionErrors2]);
    end
end

%% 
% Plotting mean reprojection error values as a surface plot
figure,
surf(r_disc,regVals,meanReprError), % Create surface plot
ylabel('Region growing limit'), % Label y-axis with region growing limit
xlabel('Structuring disk size'), % Label x-axis with structuring disk size
zlabel('Mean reprojection error value'), % Label z-axis with mean reprojection error value
title('Reprojection error values') % Set title of the plot

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