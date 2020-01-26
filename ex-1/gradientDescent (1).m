function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    a = 0;
    b = 0;
%     c = 0;
   for i = 1: m; 
        f = (X(i,:))*theta;
        a = a + (f-y(i,:))*X(i,1); 
        b = b + (f-y(i,:))*X(i,2); 
%         c = c + (f-y(i,:))*X(i,3); 
        
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
   end
   
    
        theta(1,:)=theta(1,:) - (alpha/m)*a;                                   % one theta
        theta(2,:)=theta(2,:) - (alpha/m)*b;
%         theta(3,:)=theta(3,:) - (alpha/m)*c;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
