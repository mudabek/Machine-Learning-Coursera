function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J = (1/m) * sum(-y .* log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta))) + ...
  + (lambda/2/m)*sum(theta.^2);
  
%first row elements

X_col1 = X(1:rows(X), [1]);
theta_row1 = theta(1, [1:columns(theta)]);
y_col1 = y(1:rows(y), [1]);

grad(1) = (1/m) * (X_col1' * (sigmoid(X_col1 * theta_row1) - y_col1)) ;

X_part = X(1:rows(X), [2:columns(X)]);
theta_part = theta(2:rows(theta),[1]);
y_part = y(1:rows(y), [1]);

bracket = (1/m) * (X_part' * (sigmoid(X_part*theta_part) - y_part));

grad(2:rows(grad),[1]) = bracket + (lambda/m)*theta_part;

% =============================================================

end