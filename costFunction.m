function [J, grad] = costFunction(theta, X, y)
m = length(y); % number of training examples
J = 0;
h=sigmoid(X*theta);
J=1/m*(-y'*log(h)-(1-y)'*log(1-h));
grad = zeros(size(theta));
grad =1/m*(X'*(h-y));

end
