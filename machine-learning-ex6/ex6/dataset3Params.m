function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
x1 = [1 2 1]; x2 = [0 4 -1];
%load('ex6data3.mat');
fprintf('obteniendo los parámetros más optimos \n')
accuracy = 1;
bether_accuracy = 1;
for i=linspace(0.01,0.2,10)
  for j = linspace(0.01,0.1,10)
    model = svmTrain(X,y,i, @(x1, x2) gaussianKernel(x1, x2, j));
    predictions = svmPredict(model,Xval);
    bether_accuracy = min(accuracy, bether_accuracy);
    accuracy = mean(double(predictions ~= yval));
    
    if (accuracy < bether_accuracy)
      C = i;
      sigma = j;
    endif
  
  endfor
endfor




% =========================================================================

end
