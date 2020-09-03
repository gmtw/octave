function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
[m, n] = size(X);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%%%%%%%%%%%%%%%%%%% Calculate J %%%%%%%%%%%%%%%%%%55
z=X*theta;
g = sigmoid(z);

suma=0;

for i=1:m 
  suma = (-y(i)*log(g(i)))-(1-y(i))*log(1-g(i))+ suma;
endfor

sumaR = 0;
for j=2:n
  sumaR = theta(j)**2 + sumaR;
endfor
J = round((1/m*suma + lambda/(2*m)*sumaR)*1000)/1000;

%%%%%%%%%%%%%%%%%%% partial derivate %%%%%%%%%%%%%%%%%

suma0 = 0;
for i= 1:m
  suma0 = (g(i)-y(i))*X(i,1) + suma0;
endfor
grad(1) = round((1/(m)*suma0)*10000)/10000;

%%%%%%%%%%%%%%
for j=2:n
  suma2=0;
  for i = 1:m
    suma2 = (g(i)-y(i))*X(i,j) + suma2;
  endfor
  grad(j) =round((1/(m)*suma2 + lambda/m*theta(j)).*10000)./10000;
  
endfor




% =============================================================

end
