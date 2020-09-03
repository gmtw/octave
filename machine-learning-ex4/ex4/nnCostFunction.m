function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% Part 1

 a1 = [ones(m,1) X]; %5000x401 y Theta1 -> 25x401
 z2 = Theta1*transpose(a1); %25x5000
 a2 = sigmoid(z2);
 a2 = [transpose(ones(size(a2,2),1)); a2]; %26x5000
 z3 = Theta2*a2; %10x5000
 a3 = sigmoid(z3); %10x5000 
 a3 = transpose(a3); %5000x10
 
 h = a3; %5000x10
 
 Y= transpose(unique(y)) == y; %matriz cuadrada %5000x10
 
 Theta1_reg = [zeros(size(Theta1,1),1), Theta1(:,2:end)];
 Theta2_reg = [zeros(size(Theta2,1),1), Theta2(:,2:end)];
 
 suma_K =-Y.*log(h) - (1-Y).*log(1-h);

 J=sum(sum(suma_K));  
 
 
 J = 1/m*J+ lambda/(2*m)*(sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2)));
 
 %Part 2
 
 %Step 1: Done

 %Step 2:
 delta3 = a3-Y; %5000x10
   
   %Step 3:
 delta2 = delta3*Theta2.*transpose(a2.*(1-a2)); %[5000x10*10x26] = 5000x26*5000x26
 delta2 = transpose(delta2);
 delta2 = delta2(2:end,:); %25x5000 
 %Step 4:
 Delta2 = a2*delta3; %26x5000*5000x10 = 26x10
 
 Delta1 = delta2*a1; %25x5000*5000x401 = 25x401
 %Step 5:
 %Agregar regularizacio
 %Theta1_reg = [zeros(size(Theta1,1),1), Theta1(:,2:end)];
 %Theta2_reg = [zeros(size(Theta2,1),1), Theta2(:,2:end)];
 %
 Theta1_grad = 1/m*Delta1 + lambda/m*Theta1_reg; %25x401
 Theta2_grad = 1/m*transpose(Delta2) + lambda/m*Theta2_reg; %10x26
 
 
 











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
