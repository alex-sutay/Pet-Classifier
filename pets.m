%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 1600;  % 40x40 Input Images of Digits
hidden_layer_size = 50;   % 50 hidden units
num_labels = 3;          % 3 labels, one for each pet   

% Load Training Data
printf('Select data files:\n')

count = 0;
y = [];
ytest = [];
finalX = [];
finaltest = [];
while true
    count += 1;
    
    filename = uigetfile();
    if (filename == 0)
        break;
    endif
    
    disp(filename);
    load(filename);
    m = size(X, 1); 
    X = X(randperm(m), :);  % Shuffle the rows so that a different set is used for the test set each time
    cutoff = .8 * m;
    tempX = X(1:cutoff, :);     % Grab the first 80% for the training set
    test = X(cutoff + 1:m, :); % The rest goes in the test set
    m = size(tempX, 1);      % Re-evaluate the sizes
    mtest = size(test, 1); 
        
    % Set up the expected outputs for the training and test set
    y = [y; count*ones(m, 1)];
    ytest = [ytest; count*ones(mtest, 1)];
    
    % Put it together with the rest
    finalX = [finalX; tempX];
    finaltest = [finaltest; test];
    m = size(finalX, 1);
    mtest = size(finaltest, 1);

endwhile

X = finalX;
test = finaltest;
printf('Examples in training set: ');
printf(num2str(m));
printf('\n');
printf('Examples in test set: ');
printf(num2str(mtest));
printf('\n');
pause

% Randomly select 100 data points to display
fprintf('\nVisualizing data... \n')
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 5000);

%  used for feature regularization
lambda = 1000;

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
% Green Channel Accuracy: 69.382716%, 68.888889%, 71.111111%
% Blue Channel Accuracy: 68.641975%, 68.888889%, 69.876543%
% Conclusion: It is still overfitting quite a bit, despite the extra data. Although even more data will probably
%   help, I think it's time to change some ohter aspects. Maybe it would generalize better if it used less iters...

% 2025 photo test results (1620 in training set, 405 in test set), lambda = 1000, MaxIter = 5000
% Grayscale accuracy: 68.888889%, 68.395062%, 70.864198%
% Red channel Accuracy: 63.456790%, 64.444444%, 64.938272%
% Green Channel Accuracy: 63.209877%, 68.641975%, 68.888889%
% Blue Channel Accuracy: 67.901235%, 70.617284%, 68.641975%
% Conclusion: I got it to overfit a bit less, but it's still a problem. There isn't much else I can do other than 
%   collect more data. So for now that's the next step

% 2805 photo test results (2244 in training set, 561 in test set), lambda = 1000, MaxIter = 5000
% Grayscale accuracy: 
% Red channel Accuracy: 
% Green Channel Accuracy: 
% Blue Channel Accuracy: 
% Conclusion: