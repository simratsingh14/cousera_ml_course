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
sum = 0;
h = sigmoid(X(1,:)*theta);


for i = 1:m
    h = sigmoid(X(i,:)*theta);
    sum = sum +((-1*y(i))*(log(h)) - (1-y(i))*log(1-h)); 
    for j = 1:max(size(theta))
        if(i ~= 1)
            sum = sum + (lambda)/(2*m)*(theta(j))*(theta(j));
        end
        if(i > 1)
            grad(j) = grad(j) + (h-y(i))*X(i,j) + (lambda/m)*(theta(j));
        else
            grad(j) = grad(j) + (h-y(i))*X(i,j);
        end
    end
    
    
end
J = sum/(1*m);
grad = grad/(1*m);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
