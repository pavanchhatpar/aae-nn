clear ; close all;

input_layer_size  = 17;  % number of input features
hidden_layer_size = 5;   % 5 hidden units
num_labels = 2;          % two labels 1 or 0 

%% ================ Loading data ================
fprintf('\nLoading training data ...\n')
load('aae-data.mat');
s = size(X_data, 1);
n_train = ceil(s*0.8);
%X = X_data(1:n_train,:);
%y = y_data(1:n_train,:);
%X_test = X_data(n_train+1:end,:);
%y_test = y_data(n_train+1:end,:);
for i=1:s
  indices(i) = i;
endfor
rand_indices = indices'(randperm(size(indices,2)),:);
for i=1:n_train
  X(i,:) = X_data(rand_indices(i),:);
  y(i,:) = y_data(rand_indices(i),:);
endfor
for i=1:s-n_train
  X_test(i,:) = X_data(rand_indices(i+n_train),:);
  y_test(i,:) = y_data(rand_indices(i+n_train),:);
endfor
%% ================ Initializing Pameters ================
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Training NN ===================
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 250);

lambda = 0.7;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%% ================= Part 10: Implement Predict =================

pred = predict(Theta1, Theta2, X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);