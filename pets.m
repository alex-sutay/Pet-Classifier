%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 1600;  % 40x40 Input Images of Digits
hidden_layer_size = 50;   % 50 hidden units
num_labels = 3;          % 3 labels, one for each pet   

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('KaiaDatG.mat'); % The first file name that can be changed
m1 = size(X, 1); 
cutoff = .8 * m1;
tempX = X(1:cutoff, :);     % Grab the first 80% for the training set
test = X(cutoff + 1:m1, :); % The rest goes in the test set
m1 = size(tempX, 1);      % Re-evaluate the sizes
mtest1 = size(test, 1);     

load('ZoeyDatG.mat'); % The second file name that can be changed
m2 = size(X, 1);
cutoff = .8 * m2;  
test = [test; X(cutoff + 1:m2, :)];  % Grab the last 20% for the test set
tempX = [tempX; X(1:cutoff, :)];     % The first 80% goes in the training set
m2 = size(tempX, 1) - m1;      % Re-evaluate the sizes
mtest2 = size(test, 1) - mtest1;

load('IzzyDatG.mat'); % The third file name that can be changed
m3 = size(X, 1);
cutoff = .8 * m3;
test = [test; X(cutoff + 1:m3, :)];  % Grab the last 20% for the test set
tempX = [tempX; X(1:cutoff, :)];     % The first 80% goes in the training set
m3 = size(tempX, 1) - m1-m2;      % Re-evaluate the sizes
mtest3 = size(test, 1) - mtest1-mtest2;

% Put it all together
X = tempX;
m = size(X, 1);
mtest = size(test, 1);

% Set up the expected outputs for the training set
y(1:m1) = 1;
y(m1+1:m1+m2) = 2;
y(m1+m2+1:m) = 3;
y = y';  % The above code stores Y as a horizantal vecotr, we need it to be vertical

% Same as above, but for the test set
ytest(1:mtest1) = 1;
ytest(mtest1+1:mtest1+mtest2) = 2;
ytest(mtest1+mtest2+1:mtest) = 3;
ytest = ytest';

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

fprintf('\nVisualizing data... \n')

displayData(X(sel, :));

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 10000);

%  used for feature regularization
lambda = 100;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the neural network parameters)
[nn_params, cost] = fmincg(costFunction, nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

pred = predict(Theta1, Theta2, test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

save("-v7", "Theta_output.mat", "Theta1", "Theta2");

% Initial 600 photo test results (training set accuracy)
% Grayscale accuracy: 88.989442%, 95.022624%, 74.660633%
% Red channel Accuracy: 95.776772%, 77.224736%, 77.224736%
% Green Channel Accuracy: 70.286576%, 85.520362%, 78.129713%
% Blue Channel Accuracy: 97.737557%, 95.475113%, 94.419306%
% Conclusion: The algorithm is working, the results are slightly random. Blue performs quite well.

% 1200 photo test results (960 in training set, 240 in test set), lambda = 1
% Grayscale accuracy: 64.583333%, 65.416667%, 70.416667%
% Red channel Accuracy: 63.750000%, 66.250000%, 65.416667%
% Green Channel Accuracy: 71.250000%, 65.416667%, 68.750000%
% Blue Channel Accuracy:  45.833333%, 69.16666667%, 67.916667%
% Conclusion: The algorithm has a major overfitting problem. The training set accuracy is still high 90s generally,
%   yet it is not doing well on the test set

% 1200 photo test results (960 in training set, 240 in test set), lambda = 10
% Grayscale accuracy: 68.333333%, 75.000000%, 67.500000%
% Red channel Accuracy: 62.500000%, 59.583333%, 64.583333%
% Green Channel Accuracy: 69.583333%, 68.333333%, 67.916667%
% Blue Channel Accuracy: 56.666667%, 68.750000%, 65.000000%
% Conclusion: The algorithm is still over fitting. Time to try a higher value for lambda. Although, having more data
%   will probably also help

% 1200 photo test results (960 in training set, 240 in test set), lambda = 1000
% Grayscale accuracy: 56.666667%,  65.000000%, 70.833333%
% Red channel Accuracy: 59.166667%, 65.416667%, 67.916667%
% Green Channel Accuracy: 68.750000%, 71.250000%, 68.750000%
% Blue Channel Accuracy: 64.166667%, 65.416667%, 67.083333%
% Conclusion: It's still overfitting, but looking at the image output, you don't see specific animals in any nodes
%   any more. It probably just needs more data and then a smaller lambda will probably do fine, maybe 100

% 2025 photo test results (1620 in training set, 405 in test set), lambda = 100
% Grayscale accuracy: 70.370370%, 69.382716%, 68.395062%
% Red channel Accuracy: 65.185185%, 63.456790%, 62.222222%
% Green Channel Accuracy: 
% Blue Channel Accuracy: 
% Conclusion: 