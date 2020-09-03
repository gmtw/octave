function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
%%%m = length(y); % number of training examples
[m, n] = size(X);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta



z=X*theta;
g = sigmoid(z);

suma=0;

for i=1:m 
  suma = (-y(i)*log(g(i)))-(1-y(i))*log(1-g(i)) + suma;
endfor
J = 1/m*suma;
for j=1:n
  suma2=0;
  for i = 1:m
    suma2 = (g(i)-y(i))*X(i,j) + suma2;
  endfor
  grad(j) = 1/(m)*suma2;
  
endfor


% =============================================================

end
